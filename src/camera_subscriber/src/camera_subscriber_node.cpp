#include "camera_subscriber/camera_subscriber_node.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace camera_subscriber
{
namespace
{

/// 把 header.stamp 折成纳秒。只用于 9 路之间的相对比较，
/// 所以不关心这个时间戳是仿真时间还是墙上时间
std::int64_t stampToNanoseconds(const sensor_msgs::msg::Image & msg)
{
  return static_cast<std::int64_t>(msg.header.stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(msg.header.stamp.nanosec);
}

/// 调用前必须已持有 mutex_：判断所有相机是否都到齐且时间戳足够接近
bool allSynchronized(
  const std::vector<CameraSlot> & slots, double tolerance_sec, std::int64_t & min_stamp_ns)
{
  if (slots.empty()) {
    return false;
  }
  std::int64_t min_ns = std::numeric_limits<std::int64_t>::max();
  std::int64_t max_ns = std::numeric_limits<std::int64_t>::min();
  for (const auto & slot : slots) {
    if (!slot.ever_received) {
      return false;
    }
    min_ns = std::min(min_ns, slot.stamp_ns);
    max_ns = std::max(max_ns, slot.stamp_ns);
  }
  const double span_sec = static_cast<double>(max_ns - min_ns) * 1e-9;
  if (span_sec > tolerance_sec) {
    return false;
  }
  min_stamp_ns = min_ns;
  return true;
}

}  // namespace

CameraSubscriberNode::CameraSubscriberNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("camera_subscriber", options)
{
  camera_count_ = static_cast<std::size_t>(
    this->declare_parameter<std::int64_t>("camera_count", 9));
  topic_prefix_ = this->declare_parameter<std::string>("topic_prefix", "camera_");
  topic_suffix_ = this->declare_parameter<std::string>("topic_suffix", "/image_raw");
  qos_reliability_ = this->declare_parameter<std::string>("qos_reliability", "best_effort");
  qos_depth_ = this->declare_parameter<std::int64_t>("qos_depth", 5);
  stats_period_sec_ = this->declare_parameter<double>("stats_period_sec", 5.0);
  stale_timeout_sec_ = this->declare_parameter<double>("stale_timeout_sec", 1.0);
  sync_tolerance_sec_ = this->declare_parameter<double>("sync_tolerance_sec", 0.02);

  if (camera_count_ < 1 || camera_count_ > kMaxCameras) {
    RCLCPP_WARN(
      get_logger(), "camera_count=%zu 超出 1~%zu，已夹到范围内", camera_count_, kMaxCameras);
    camera_count_ = std::min(std::max<std::size_t>(camera_count_, 1), kMaxCameras);
  }
  if (qos_depth_ < 1) {
    qos_depth_ = 1;
  }
  if (stats_period_sec_ <= 0.0) {
    stats_period_sec_ = 5.0;
  }

  slots_.resize(camera_count_);
  last_stats_time_ = std::chrono::steady_clock::now();

  createSubscriptions();

  stats_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(stats_period_sec_),
    [this]() {publishStats();});

  RCLCPP_INFO(get_logger(), "已订阅 %zu 路相机画面：", camera_count_);
  for (std::size_t i = 0; i < camera_count_; ++i) {
    RCLCPP_INFO(
      get_logger(), "  [%zu] %s (row=%zu, col=%zu)", i, topicName(i).c_str(),
      rowOf(i), colOf(i));
  }
}

void CameraSubscriberNode::setFrameCallback(FrameCallback callback)
{
  std::lock_guard<std::mutex> lock(mutex_);
  frame_callback_ = std::move(callback);
}

void CameraSubscriberNode::setFrameSetCallback(FrameSetCallback callback)
{
  std::lock_guard<std::mutex> lock(mutex_);
  frame_set_callback_ = std::move(callback);
}

std::string CameraSubscriberNode::topicName(std::size_t index) const
{
  return topic_prefix_ + std::to_string(index) + topic_suffix_;
}

void CameraSubscriberNode::createSubscriptions()
{
  // 9 路回调互不阻塞，配合 MultiThreadedExecutor 可以并行处理
  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  // gazebo_ros_camera 用 SensorDataQoS（best_effort）发布。
  // 订阅端若用 reliable 会 QoS 不兼容，一帧都收不到，而且不会报错。
  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<std::size_t>(qos_depth_)));
  if (qos_reliability_ == "reliable") {
    qos.reliable();
  } else {
    if (qos_reliability_ != "best_effort") {
      RCLCPP_WARN(
        get_logger(), "未知的 qos_reliability='%s'，按 best_effort 处理",
        qos_reliability_.c_str());
    }
    qos.best_effort();
  }
  qos.durability_volatile();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = callback_group_;

  subscriptions_.reserve(camera_count_);
  for (std::size_t i = 0; i < camera_count_; ++i) {
    subscriptions_.push_back(
      this->create_subscription<sensor_msgs::msg::Image>(
        topicName(i), qos,
        [this, i](const ImagePtr & msg) {onImage(i, msg);},
        sub_options));
  }
}

void CameraSubscriberNode::onImage(std::size_t index, const ImagePtr & msg)
{
  FrameCallback frame_cb;
  FrameSetCallback set_cb;
  std::vector<ImagePtr> frame_set;
  bool emit_frame_set = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= slots_.size()) {
      return;
    }
    frame_cb = frame_callback_;
    set_cb = frame_set_callback_;

    CameraSlot & slot = slots_[index];
    slot.frame = msg;
    slot.stamp_ns = stampToNanoseconds(*msg);
    slot.arrival = std::chrono::steady_clock::now();
    slot.width = msg->width;
    slot.height = msg->height;
    slot.encoding = msg->encoding;
    ++slot.frame_count;
    ++slot.frames_in_window;
    slot.ever_received = true;

    if (set_cb) {
      std::int64_t min_stamp_ns = 0;
      if (allSynchronized(slots_, sync_tolerance_sec_, min_stamp_ns) &&
        min_stamp_ns > last_emitted_stamp_ns_)
      {
        last_emitted_stamp_ns_ = min_stamp_ns;
        frame_set.reserve(slots_.size());
        for (const auto & s : slots_) {
          frame_set.push_back(s.frame);
        }
        emit_frame_set = true;
      }
    }
  }

  // 用户回调（尤其是模型推理）在锁外执行，不能阻塞订阅回调
  if (frame_cb) {
    frame_cb(index, msg);
  }
  if (emit_frame_set && set_cb) {
    set_cb(frame_set);
  }
}

void CameraSubscriberNode::publishStats()
{
  const auto now = std::chrono::steady_clock::now();
  const double window_sec = std::chrono::duration<double>(now - last_stats_time_).count();
  last_stats_time_ = now;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  oss << "相机订阅状态（" << camera_count_ << " 路）：";

  std::vector<std::string> problems;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      CameraSlot & slot = slots_[i];
      slot.fps = (window_sec > 0.0) ?
        static_cast<double>(slot.frames_in_window) / window_sec : 0.0;
      slot.frames_in_window = 0;

      oss << "\n  [" << i << "] " << topicName(i);
      if (!slot.ever_received) {
        oss << " -> 还没收到任何帧";
        problems.push_back(topicName(i) + " 没收到数据");
        continue;
      }
      const double age_sec = std::chrono::duration<double>(now - slot.arrival).count();
      oss << " -> " << slot.fps << " Hz，累计 " << slot.frame_count << " 帧，"
          << slot.width << "x" << slot.height << " " << slot.encoding
          << "，距上一帧 " << age_sec << " s";
      if (age_sec > stale_timeout_sec_) {
        problems.push_back(topicName(i) + " 已停顿 " + std::to_string(age_sec) + " s");
      }
    }
  }

  RCLCPP_INFO_STREAM(get_logger(), oss.str());
  for (const auto & problem : problems) {
    RCLCPP_WARN(get_logger(), "  ! %s", problem.c_str());
  }
}

ImagePtr CameraSubscriberNode::latestFrame(std::size_t index) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (index >= slots_.size()) {
    return nullptr;
  }
  return slots_[index].frame;
}

bool CameraSubscriberNode::latestFrames(std::vector<ImagePtr> & out) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  out.clear();
  out.reserve(slots_.size());
  for (const auto & slot : slots_) {
    if (!slot.frame) {
      out.clear();
      return false;
    }
    out.push_back(slot.frame);
  }
  return !out.empty();
}

}  // namespace camera_subscriber
