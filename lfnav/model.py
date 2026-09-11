import torch
from torch import nn
import torch.nn.functional as F


def conv(cin, cout, kernel=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride,
                                   padding=dilation*(kernel//2), dilation=dilation,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def separable(cin, cout, stride=1, dilation=1):
    return nn.Sequential(conv(cin, cin, stride=stride, dilation=dilation, groups=cin),
                         conv(cin, cout, kernel=1))


class LFNavNet(nn.Module):
    """Original small engineering baseline inspired by EPINET, not its reproduction.

    Four 3-view RGB directional stems cover all nine views. Only standard 2D
    operators are used, with an unbounded signed disparity and Laplace log b.
    Input: [N,27,H,W], view-major RGB in [0,1]. H,W must be multiples of 8.
    Output: [N,2,H,W], channels = disparity_px, log_laplace_scale_px.
    """
    groups = ((3, 4, 5), (1, 4, 7), (0, 4, 8), (2, 4, 6))

    def __init__(self, width=24, stem_depth=1):
        super().__init__()
        if width < 8 or width % 2:
            raise ValueError("width must be even and >=8")
        self.width = width
        self.epinet_grayscale = False
        self.stems = nn.ModuleList([conv(9, width//2) for _ in self.groups])
        if stem_depth not in (1, 2, 3):
            raise ValueError("stem_depth must be 1, 2 or 3")
        self.stem_depth = stem_depth
        if stem_depth >= 2:
            # Identity initialization preserves pretrained nonnegative stem features.
            # A full 3x3 convolution per direction, not a separable convolution.
            self.stem_second = nn.ModuleList([
                nn.Sequential(nn.Conv2d(width//2, width//2, 3, padding=1), nn.ReLU(inplace=True))
                for _ in self.groups])
            for layer in self.stem_second:
                nn.init.dirac_(layer[0].weight)
                nn.init.zeros_(layer[0].bias)
        if stem_depth == 3:
            self.stem_third = nn.ModuleList([
                nn.Sequential(nn.Conv2d(width//2, width//2, 3, padding=1), nn.ReLU(inplace=True))
                for _ in self.groups])
        self.fuse = separable(width*2, width)
        self.down1 = nn.Sequential(separable(width, width*2, 2), separable(width*2, width*2))
        self.down2 = nn.Sequential(separable(width*2, width*4, 2), separable(width*4, width*4))
        self.down3 = nn.Sequential(separable(width*4, width*4, 2), separable(width*4, width*4, dilation=2),
                                   separable(width*4, width*4, dilation=4))
        self.up2 = separable(width*8, width*2)
        self.up1 = separable(width*4, width)
        self.up0 = nn.Sequential(separable(width*2, width), separable(width, width))
        self.head = nn.Conv2d(width, 2, 3, padding=1)

    def features(self, x):
        if self.epinet_grayscale:
            # Embedded preprocessing keeps evaluation, ONNX and RGB deployment identical.
            views = x.reshape(x.shape[0], 9, 3, *x.shape[-2:])
            gray = views[:, :, 0:1]*.299 + views[:, :, 1:2]*.587 + views[:, :, 2:3]*.114
            x = gray.expand(-1, -1, 3, -1, -1).reshape_as(x)
        x = x*2-1
        directions = [stem(torch.cat([x[:, v*3:v*3+3] for v in ids], dim=1))
                      for stem, ids in zip(self.stems, self.groups)]
        if self.stem_depth >= 2:
            directions = [layer(z) for layer, z in zip(self.stem_second, directions)]
        if self.stem_depth == 3:
            directions = [layer(z) for layer, z in zip(self.stem_third, directions)]
        a = self.fuse(torch.cat(directions, dim=1))
        b = self.down1(a)
        c = self.down2(b)
        z = self.down3(c)
        z = self.up2(torch.cat([F.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        z = self.up1(torch.cat([F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        z = self.up0(torch.cat([F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return z

    def forward(self, x):
        z = self.head(self.features(x))
        return torch.cat([z[:, :1], z[:, 1:2].clamp(-5, 3)], 1)


class LocalMatch(nn.Module):
    """Half-resolution RGB costs; hypotheses remain in full-resolution pixels.

    HCI convention: sample view (u,v) at (x-u*d, y-v*d). Opposite-view
    minimum and mean costs retain both one-sided and two-sided evidence.
    No learned 3D volume or extra feature encoder is needed.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", torch.linspace(-.5, .5, 9), persistent=False)
        self.register_buffer("angles", torch.tensor([
            [-1., -1.], [0., -1.], [1., -1.], [-1., 0.],
            [1., 0.], [-1., 1.], [0., 1.], [1., 1.]]), persistent=False)

    def forward(self, x, disparity):
        # FP32 coordinates are essential when resolving <0.07 px at 512x512.
        x = F.avg_pool2d(x.float(), 2)
        n, _, h, w = x.shape
        views = x.reshape(n, 9, 3, h, w)
        sides = torch.cat([views[:, :4], views[:, 5:]], 1)
        d = F.interpolate(disparity.detach().float(), size=(h, w), mode="bilinear", align_corners=False)
        hypotheses = (d + self.offsets.reshape(1, 9, 1, 1)) * .5
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                                torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
        gx = xx[None, None, None] - self.angles[:, 0].reshape(1, 8, 1, 1, 1)*hypotheses[:, None]
        gy = yy[None, None, None] - self.angles[:, 1].reshape(1, 8, 1, 1, 1)*hypotheses[:, None]
        grid = torch.stack([2*(gx+.5)/w-1, 2*(gy+.5)/h-1], -1).reshape(n*8, 9*h, w, 2)
        warped = F.grid_sample(sides.reshape(n*8, 3, h, w), grid,
                               mode="bilinear", padding_mode="border", align_corners=False)
        costs = (warped.reshape(n, 8, 3, 9, h, w)-views[:, 4:5, :, None]).abs().mean(2)
        left, right = costs[:, :4], costs[:, 4:].flip(1)
        return torch.cat([torch.minimum(left, right), (left+right)*.5], 1).reshape(n, 72, h, w)


class LFNavMatchNet(LFNavNet):
    """Warm-startable baseline plus local matching and a small residual head."""
    def __init__(self, width=96, matching=True):
        super().__init__(width)
        self.matching = matching
        self.match = LocalMatch()
        self.cost_encoder = nn.Sequential(conv(72, 32), separable(32, 32, dilation=2),
                                          separable(32, 32, dilation=4))
        self.refine = nn.Sequential(separable(width+32+2, 32), separable(32, 32))
        self.residual_head = nn.Conv2d(32, 2, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x):
        features = self.features(x)
        prior = self.head(features)
        if self.matching:
            costs = self.match(x, prior[:, :1])
        else:
            # Parameter-matched ablation: no angular matching evidence.
            costs = x.new_zeros(x.shape[0], 72, x.shape[2]//2, x.shape[3]//2)
        cost_features = self.cost_encoder(costs)
        cost_features = F.interpolate(cost_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.residual_head(self.refine(torch.cat([features, cost_features, prior], 1)))
        output = prior + residual
        return torch.cat([output[:, :1], output[:, 1:2].clamp(-5, 3)], 1)


class LFNavDetailNet(LFNavNet):
    """Full-resolution RGB residual; zero head preserves the source prediction."""
    def __init__(self, width=96):
        super().__init__(width, stem_depth=2)
        self.detail = nn.Sequential(conv(27+width+2, 24), conv(24, 24),
                                    conv(24, 24, dilation=2))
        self.detail_head = nn.Conv2d(24, 2, 3, padding=1)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)

    def forward(self, x):
        features = self.features(x)
        prior = self.head(features)
        out = prior + self.detail_head(self.detail(torch.cat([x*2-1, features, prior], 1)))
        return torch.cat([out[:, :1], out[:, 1:2].clamp(-5, 3)], 1)


def build_model(config):
    architecture = config.get("architecture", "baseline")
    if architecture == "two_layer_detail":
        if config.get("input_color", "rgb") != "rgb":
            raise ValueError("two_layer_detail supports RGB input only")
        return LFNavDetailNet(config["width"])
    if architecture == "baseline":
        return LFNavNet(config["width"])
    if architecture == "two_layer_stem":
        model = LFNavNet(config["width"], stem_depth=2)
        model.epinet_grayscale = config.get("input_color") == "epinet_gray"
        return model
    if architecture == "three_layer_stem":
        model = LFNavNet(config["width"], stem_depth=3)
        model.epinet_grayscale = config.get("input_color") == "epinet_gray"
        return model
    if architecture in ("local_match", "refine_only"):
        if architecture == "local_match" and config.get("angular_step", 1) != 1:
            raise ValueError("local_match currently supports angular_step=1 only")
        return LFNavMatchNet(config["width"], matching=architecture == "local_match")
    raise ValueError(f"Unknown architecture: {architecture}")


def initialize_model(model, state, config):
    """Strict loading, except the explicit baseline -> residual-model migration."""
    source = state["config"].get("architecture", "baseline")
    target = config.get("architecture", "baseline")
    if source == target:
        model.load_state_dict(state["model"], strict=True)
    elif source == "baseline" and target in ("local_match", "refine_only", "two_layer_stem"):
        expected = {k for k in model.state_dict() if k.startswith(("cost_encoder.", "refine.", "residual_head.", "stem_second."))}
        # Check BEFORE loading so malformed source checkpoints cannot silently transfer.
        if set(state["model"]) != set(model.state_dict()) - expected:
            raise ValueError("Source is not a complete baseline checkpoint")
        model.load_state_dict(state["model"], strict=False)
    else:
        raise ValueError(f"Unsupported architecture transfer: {source} -> {target}")
    factor = state["config"].get("angular_step", 1)/config.get("angular_step", 1)
    if factor != 1:
        if source not in ("baseline", "two_layer_stem") or target not in ("baseline", "two_layer_stem"):
            raise ValueError("Angular transfer scaling supports baseline and two_layer_stem only")
        # Larger input spacing produces larger image motion; retain original GT
        # units by scaling the disparity head at initialization. Uncertainty is
        # not presumed better and is left for supervised recalibration.
        with torch.no_grad():
            model.head.weight[0].mul_(factor)
            model.head.bias[0].mul_(factor)
    return dict(source_architecture=source, architecture=target,
                transferred_tensors=len(state["model"]), disparity_head_transfer_factor=factor)
