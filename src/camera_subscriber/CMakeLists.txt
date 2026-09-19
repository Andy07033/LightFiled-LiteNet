cmake_minimum_required(VERSION 3.8)
project(camera_subscriber)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

# ---- 订阅逻辑做成库：后面写模型节点时直接 link camera_subscriber_core ----
add_library(${PROJECT_NAME}_core SHARED
  src/camera_subscriber_node.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
)

# ---- 可执行节点 ----
add_executable(${PROJECT_NAME}_node
  src/camera_subscriber_main.cpp
)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_core
)

install(
  TARGETS ${PROJECT_NAME}_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 与 robot_description 保持一致：跳过版权检查
  set(ament_cmake_copyright_FOUND TRUE)
  # 跳过 cpplint（本工作区不是 git 仓库）
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_export_dependencies(rclcpp sensor_msgs)
ament_export_targets(export_${PROJECT_NAME})

# =====================================================================
# 光场推理（Python 侧）
#   camera_subscriber/            Python 包：calibration / geometry / inference / lightfield_node
#   scripts/lightfield_node       入口，安装成 lib/camera_subscriber/lightfield_node
#   models/best.onnx              ONNX 权重（best.pt 是训练权重，留在源码树不进安装空间）
# 用 onnxruntime 而不是 cv::dnn：本网络输入 [1,27,512,512]，单帧约 10 GFLOP，
# CPU 只能跑到零点几 Hz，只有 onnxruntime-gpu 这条路能走到实时。
# =====================================================================
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS scripts/lightfield_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES models/best.onnx
  DESTINATION share/${PROJECT_NAME}/models
)

# ---- 已停用：C++/cv::dnn 支线 ----
# src/litenet_node.cpp 是没编译过的草稿，且通道顺序有误
# （cv_bridge 按 "bgr8" 拆出来是 B,G,R，而模型要 view-major RGB），
# 它输出的是归一化 mono8 图，也不是点云。推理已由 Python 侧的
# lightfield_node 接管，所以这里不再构建它。要用就取消下面 5 行注释。
# add_executable(litenet_node src/litenet_node.cpp)
# target_include_directories(litenet_node PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(litenet_node
#   ${PROJECT_NAME}_core          # 复用订阅逻辑，不用重写 9 路订阅
#   ${OpenCV_LIBRARIES}
# )
# ament_target_dependencies(litenet_node rclcpp sensor_msgs cv_bridge)
# install(TARGETS litenet_node DESTINATION lib/${PROJECT_NAME})

ament_package()
