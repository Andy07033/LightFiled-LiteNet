"""Optional training-only sensor noise / shared contrast augmentation."""
import torch


def augment_robustness(x, mode):
    if mode not in ("none", "noise", "contrast", "noise_contrast"):
        raise ValueError(f"Unknown robustness mode: {mode}")
    shape = (x.shape[0], 1, 1, 1)
    if mode in ("contrast", "noise_contrast"):
        # One factor per light field, shared by all cameras and colors.
        factor = torch.empty(shape, device=x.device).uniform_(.3, 1.)
        active = torch.rand(shape, device=x.device) < .5
        factor = torch.where(active, factor, torch.ones_like(factor))
        x = ((x-.5)*factor+.5).clamp(0, 1)
    if mode in ("noise", "noise_contrast"):
        # Sensor samples differ across cameras, but noise strength is shared.
        sigma = torch.empty(shape, device=x.device).uniform_(0., .12)
        active = torch.rand(shape, device=x.device) < .5
        x = (x+torch.randn_like(x)*sigma*active).clamp(0, 1)
    return x
