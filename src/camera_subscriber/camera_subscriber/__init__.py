"""光场相机阵列的推理侧代码。

由 lightfield_bridge/lightfield_litenet/lfnav 提取进本包，改动见各模块 docstring。
对外只需要关心 lightfield_node（ROS 节点）与 calibration/geometry/inference 三层。
"""

__all__ = ["calibration", "geometry", "inference", "lightfield_node"]
