#ifndef CAMERA_SUBSCRIBER__CAMERA_SUBSCRIBER_NODE_HPP_
#define CAMERA_SUBSCRIBER__CAMERA_SUBSCRIBER_NODE_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace camera_subscriber
{

/// 相机路数上限（数组容量，实际路数由 camera_count 参数决定）
constexpr std::size_t kMaxCameras = 32;

/// 3x3 光场阵列：index = row * kArrayCols + col，
/// 与 camera.urdf.xacro 里 (u, v) 的顺序严格对应，绝对不能重排
constexpr std::size_t kArrayRows = 3;
constexpr std::size_t kArrayCols = 3;

inline std::size_t rowOf(std::size_t index) {return index / kArrayCols;}
inline std::size_t colOf(std::size_t index) {return index % kArrayCols;}

using ImagePtr = sensor_msgs::msg::Image::ConstSharedPtr;

/// 单路相机的最新帧 + 运行统计
struct CameraSlot
{
  ImagePtr frame;                                  ///< 最新一帧（共享指针，不拷贝像素）
  std::int64_t stamp_ns{0};                        ///< header.stamp 折成纳秒，用于判断 9 路是否同步
  std::chrono::steady_clock::time_point arrival{}; ///< 本地收到该帧的时刻，用于算延迟/帧率
  std::size_t frame_count{0};                      ///< 累计帧数
  std::size_t frames_in_window{0};                 ///< 本统计周期内收到的帧数
  double fps{0.0};                                 ///< 最近一个统计周期测得的帧率
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::string encoding;
  bool ever_received{false};
};

/// 单路回调：每收到一帧调用一次
using FrameCallback = std::function<void (std::size_t camera_index, const ImagePtr & frame)>;

/// 整组回调：9 路都到齐且时间戳足够接近时调用一次，直接喂模型
using FrameSetCallback = std::function<void (const std::vector<ImagePtr> & frames)>;

class CameraSubscriberNode : public rclcpp::Node
{
public:
  explicit CameraSubscriberNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CameraSubscriberNode() override = default;

  /// 注册单路回调
  void setFrameCallback(FrameCallback callback);
  /// 注册整组回调（同一时刻的 camera_count 路画面到齐时调用一次）
  void setFrameSetCallback(FrameSetCallback callback);

  std::size_t cameraCount() const noexcept {return camera_count_;}
  /// 第 index 路的话题名，例如 /camera_3/image_raw
  std::string topicName(std::size_t index) const;

  /// 线程安全取第 index 路最新帧，没有则返回 nullptr
  ImagePtr latestFrame(std::size_t index) const;
  /// 线程安全取所有路最新帧；全部有帧才返回 true（out 大小为 camera_count）
  bool latestFrames(std::vector<ImagePtr> & out) const;

private:
  void createSubscriptions();
  void onImage(std::size_t index, const ImagePtr & msg);
  void publishStats();

  std::size_t camera_count_{9};
  std::string topic_prefix_{"camera_"};
  std::string topic_suffix_{"/image_raw"};
  std::string qos_reliability_{"best_effort"};
  std::int64_t qos_depth_{5};
  double stats_period_sec_{5.0};
  double stale_timeout_sec_{1.0};
  double sync_tolerance_sec_{0.02};

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscriptions_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  mutable std::mutex mutex_;
  std::vector<CameraSlot> slots_;
  std::int64_t last_emitted_stamp_ns_{-1};
  std::chrono::steady_clock::time_point last_stats_time_{};

  FrameCallback frame_callback_;
  FrameSetCallback frame_set_callback_;
};

}  // namespace camera_subscriber

#endif  // CAMERA_SUBSCRIBER__CAMERA_SUBSCRIBER_NODE_HPP_
