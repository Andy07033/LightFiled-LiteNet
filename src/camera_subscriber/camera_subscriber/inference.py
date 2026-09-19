"""ONNX 推理封装。

由 lightfield_bridge/lightfield_litenet/lfnav/runtime.py 提取，OnnxRunner 逐字等价；
差别在输入侧：原版吃一张已经拼好的 1536x1536 大图，这里直接吃 9 路 sensor_msgs/Image，
自己负责取 RGB、拼图、展开成模型要的 [1,27,H,W]。
"""

import numpy as np

# 相机插件实际会发的编码；其他编码不做猜测，直接报错
_ENCODING_CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def image_msg_to_rgb(msg):
    """sensor_msgs/Image → (H, W, 3) uint8，RGB 顺序。

    按 msg.step 取有效像素（相机的行末可能有填充），
    与 lfnav/runtime.py:unpack_grid 的处理方式一致。
    """
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    channels = _ENCODING_CHANNELS.get(msg.encoding)
    if channels is None:
        raise ValueError("不支持的图像编码: %s" % msg.encoding)
    if step < width * channels or len(msg.data) < step * height:
        raise ValueError("图像 step 或缓冲区长度不合法")

    raw = np.frombuffer(msg.data, dtype=np.uint8, count=step * height).reshape(height, step)
    image = raw[:, : width * channels].reshape(height, width, channels)
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    if msg.encoding == "bgr8":
        return image[..., ::-1]
    return image


def build_mosaic(frames, height, width):
    """9 张 (H, W, 3) RGB → (3H, 3W, 3) uint8 拼图。

    行优先：第 i 张放在 (i // 3) 行、(i % 3) 列。
    顺序必须和 camera.urdf.xacro 里的 (u, v) 一致，重排会让子孔径几何关系出错。
    """
    if len(frames) != 9:
        raise ValueError("拼图需要恰好 9 路画面，收到 %d 路" % len(frames))
    mosaic = np.empty((3 * height, 3 * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, col = divmod(index, 3)
        mosaic[row * height:(row + 1) * height, col * width:(col + 1) * width] = frame
    return mosaic


def mosaic_to_input(mosaic, height, width):
    """(3H, 3W, 3) RGB uint8 → [1, 27, H, W] float32，范围 [0, 1]。

    与 lfnav/runtime.py:unpack_grid 的 reshape/transpose 完全一致：
    先按 (3, H, 3, W, 3) 解释成 [行, 高, 列, 宽, 通道]，
    再转成 [行, 列, 通道, 高, 宽]，最后按「视角优先、每视角 RGB」展平成 27 通道。
    模型内部自己做 x*2-1，所以这里只除 255。
    """
    views = mosaic.reshape(3, height, 3, width, 3).transpose(0, 2, 4, 1, 3)
    return np.ascontiguousarray(views.reshape(1, 27, height, width), dtype=np.float32) / 255.0


class OnnxRunner:
    """与 lfnav/runtime.py 的 OnnxRunner 等价（同样的 provider 选择与形状校验）。"""

    def __init__(self, path, provider="auto"):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider not in ("auto", "cpu", "cuda"):
            raise ValueError("provider 只能是 auto / cpu / cuda")
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider 不可用；请安装匹配 CUDA/cuDNN 的 onnxruntime-gpu")

        providers = ["CPUExecutionProvider"]
        if provider != "cpu" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        if provider == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError("CUDA provider 初始化失败")
        self.shape = tuple(self.session.get_inputs()[0].shape)

    def __call__(self, x):
        if tuple(x.shape) != self.shape:
            raise ValueError("模型期望 %s，实际 %s" % (self.shape, tuple(x.shape)))
        return self.session.run(
            None, {"light_field": np.ascontiguousarray(x, dtype=np.float32)})[0]
