#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "camera_subscriber/camera_subscriber_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<camera_subscriber::CameraSubscriberNode>();

  // ========================= 模型接入点 =========================
  // 每当 9 路画面凑齐一组时间戳同步的帧，就回调一次。
  // frames[0..8] 依次对应 camera_0 .. camera_8（行优先，见 rowOf/colOf），
  // 元素是 ConstSharedPtr，不拷贝像素数据，回调结束即可释放。
  node->setFrameSetCallback(
    [logger = node->get_logger()](const std::vector<camera_subscriber::ImagePtr> & frames) {
      RCLCPP_INFO_ONCE(
        logger, "首次收到同步的 %zu 路画面，这里可以送进模型", frames.size());
      // TODO(模型): 例如
      //   auto input  = buildTensor(frames);   // 9 路 -> NCHW / NHWC
      //   auto output = model_->infer(input);
      //   publishResult(output);
    });
  // ==============================================================

  // 订阅回调 + 模型推理都在这个线程池里跑，9 路互不阻塞
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
