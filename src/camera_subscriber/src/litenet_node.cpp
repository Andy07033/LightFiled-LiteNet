// 示例：把模型推理接到光场相机订阅上（独立推理线程版）
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "camera_subscriber/camera_subscriber_node.hpp"

namespace
{
constexpr int kRows = 3;
constexpr int kCols = 3;
constexpr int kCams = kRows * kCols;   // 9

// 布局一：每个视图的 BGR 三个平面依次排开 -> [1, 27, H, W]
// 通道顺序 = camera_0..camera_8 的 (B,G,R)，索引即光场 (u,v) 的行优先顺序
cv::Mat toChannelStack(const std::vector<camera_subscriber::ImagePtr> & frames)
{
  const int h = static_cast<int>(frames[0]->height);
  const int w = static_cast<int>(frames[0]->width);
  const int channels = kCams * 3;

  cv::Mat blob(std::vector<int>{1, channels, h, w}, CV_32F);
  float * base = blob.ptr<float>(0);

  for (int i = 0; i < kCams; ++i) {
    cv::Mat bgr = cv_bridge::toCvShare(frames[i], "bgr8")->image;
    cv::Mat f32;
    bgr.convertTo(f32, CV_32FC3, 1.0 / 255.0);       // 归一化到 [0,1]
    std::vector<cv::Mat> planes(3);
    cv::split(f32, planes);                          // B, G, R
    for (int c = 0; c < 3; ++c) {
      cv::Mat dst(h, w, CV_32F, base + static_cast<std::size_t>(i * 3 + c) * h * w);
      planes[c].copyTo(dst);
    }
  }
  return blob;
}

// 布局二：按 3x3 行列拼成一张大图（spatial-angular 排布）-> [1, 3, 3H, 3W]
// 用哪个取决于你模型训练时的输入定义，两者不能混
cv::Mat toMosaic(const std::vector<camera_subscriber::ImagePtr> & frames)
{
  const int h = static_cast<int>(frames[0]->height);
  const int w = static_cast<int>(frames[0]->width);
  cv::Mat mosaic(kRows * h, kCols * w, CV_8UC3);
  for (int i = 0; i < kCams; ++i) {
    cv::Mat bgr = cv_bridge::toCvShare(frames[i], "bgr8")->image;
    // i/3 是行，i%3 是列，与 xacro 里的 (u,v) 顺序一致，不能重排
    bgr.copyTo(mosaic(cv::Rect((i % kCols) * w, (i / kCols) * h, w, h)));
  }
  return cv::dnn::blobFromImage(
    mosaic, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
}
}  // namespace

class LiteNetRunner : public rclcpp::Node
{
public:
  LiteNetRunner()
  : rclcpp::Node("litenet_node")
  {
    const std::string model_path = declare_parameter<std::string>("model_path", "");
    infer_period_sec_ = declare_parameter<double>("infer_period_sec", 0.1);
    sync_tolerance_sec_ = declare_parameter<double>("sync_tolerance_sec", 0.05);
    result_pub_ = create_publisher<sensor_msgs::msg::Image>("~/result", rclcpp::QoS(1));

    if (!model_path.empty()) {
      net_ = cv::dnn::readNetFromONNX(model_path);
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);   // 有 CUDA 换 DNN_TARGET_CUDA
      RCLCPP_INFO(get_logger(), "模型已加载: %s", model_path.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "未设置 model_path，只跑预处理并打印输入形状");
    }
  }

  ~LiteNetRunner() override
  {
    stop_ = true;
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  /// 绑上订阅节点并启动推理线程
  void bind(const std::shared_ptr<camera_subscriber::CameraSubscriberNode> & camera)
  {
    camera_ = camera;
    worker_ = std::thread(&LiteNetRunner::loop, this);
  }

private:
  void loop()
  {
    const auto period = std::chrono::duration<double>(infer_period_sec_);
    while (rclcpp::ok() && !stop_) {
      std::this_thread::sleep_for(period);

      std::vector<camera_subscriber::ImagePtr> frames;
      if (!camera_->latestFrames(frames)) {
        continue;                      // 还没凑齐 9 路
      }
      if (!usable(frames)) {
        continue;                      // 某路卡住或尺寸不一致，宁可不推理
      }
      infer(frames, frames[kCams / 2]->header.stamp);   // 时间戳取中心相机
    }
  }

  /// 时间戳是否足够接近 + 各路尺寸是否一致
  bool usable(const std::vector<camera_subscriber::ImagePtr> & frames) const
  {
    std::int64_t lo = std::numeric_limits<std::int64_t>::max();
    std::int64_t hi = std::numeric_limits<std::int64_t>::min();
    const std::uint32_t h = frames[0]->height;
    const std::uint32_t w = frames[0]->width;
    for (const auto & f : frames) {
      if (f->height != h || f->width != w) {
        return false;
      }
      const std::int64_t ns =
        static_cast<std::int64_t>(f->header.stamp.sec) * 1000000000LL +
        static_cast<std::int64_t>(f->header.stamp.nanosec);
      lo = std::min(lo, ns);
      hi = std::max(hi, ns);
    }
    return static_cast<double>(hi - lo) * 1e-9 <= sync_tolerance_sec_;
  }

  void infer(
    const std::vector<camera_subscriber::ImagePtr> & frames,
    const builtin_interfaces::msg::Time & stamp)
  {
    cv::Mat blob = toChannelStack(frames);    // 换成 toMosaic(frames) 也可以
    if (net_.empty()) {
      RCLCPP_INFO_ONCE(
        get_logger(), "输入 blob: %d 维 [%d, %d, %d, %d]",
        blob.dims, blob.size[0], blob.size[1], blob.size[2], blob.size[3]);
      return;
    }
    net_.setInput(blob);
    const cv::Mat out = net_.forward();
    publish(out, stamp);
  }

  void publish(const cv::Mat & out, const builtin_interfaces::msg::Time & stamp)
  {
    // 模型输出一般是 [1,1,H,W] 浮点图（深度/显著性），归一化到 mono8 发出去
    cv::Mat m = out;
    if (m.dims == 4) {
      CV_Assert(m.size[0] == 1 && m.size[1] == 1);
      const std::vector<int> shape{m.size[2], m.size[3]};
      m = m.reshape(1, shape);
    }
    double lo = 0.0;
    double hi = 0.0;
    cv::minMaxLoc(m, &lo, &hi);
    cv::Mat u8;
    if (hi > lo) {
      m.convertTo(u8, CV_8U, 255.0 / (hi - lo), -lo * 255.0 / (hi - lo));
    } else {
      m.convertTo(u8, CV_8U);
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = "camera_4_optical_link";
    result_pub_->publish(*cv_bridge::CvImage(header, "mono8", u8).toImageMsg());
  }

  std::shared_ptr<camera_subscriber::CameraSubscriberNode> camera_;
  cv::dnn::Net net_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_pub_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  double infer_period_sec_{0.1};
  double sync_tolerance_sec_{0.05};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto camera = std::make_shared<camera_subscriber::CameraSubscriberNode>();
  auto runner = std::make_shared<LiteNetRunner>();
  runner->bind(camera);

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(camera);
  executor.add_node(runner);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}