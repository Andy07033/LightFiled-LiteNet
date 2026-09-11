import numpy as np


def unpack_grid(msg, height, width):
    """Decode a ROS-like Image buffer; pure NumPy for testing without ROS installed."""
    if msg.height != 3*height or msg.width != 3*width or msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError("Expected a 3H x 3W rgb8/bgr8 tile grid of synchronized rectified subaperture views")
    if msg.step < msg.width*3 or len(msg.data) != msg.step*msg.height:
        raise ValueError("Invalid image row stride/buffer")
    rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3]
    rgb = rgb.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        rgb = rgb[..., ::-1]
    views = rgb.reshape(3, height, 3, width, 3).transpose(0, 2, 4, 1, 3)
    return np.ascontiguousarray(views.reshape(1, 27, height, width), dtype=np.float32)/255


class OnnxRunner:
    def __init__(self, path, provider='auto'):
        import onnxruntime as ort
        available = ort.get_available_providers()
        if provider not in ('auto', 'cpu', 'cuda'):
            raise ValueError('provider must be auto, cpu or cuda')
        if provider == 'cuda' and 'CUDAExecutionProvider' not in available:
            raise RuntimeError('CUDAExecutionProvider unavailable; install compatible onnxruntime-gpu')
        providers = ['CPUExecutionProvider']
        if provider != 'cpu' and 'CUDAExecutionProvider' in available:
            providers.insert(0, 'CUDAExecutionProvider')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        if provider == 'cuda' and 'CUDAExecutionProvider' not in self.providers:
            raise RuntimeError('CUDA provider failed to initialize')
        self.shape = tuple(self.session.get_inputs()[0].shape)

    def __call__(self, x):
        if x.shape != self.shape:
            raise ValueError(f"Expected {self.shape}, got {x.shape}")
        return self.session.run(None, {"light_field": np.ascontiguousarray(x, dtype=np.float32)})[0]


