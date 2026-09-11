import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# HCI row-major 9x9 array: angular coordinates (3..5, 3..5).
VIEW_IDS = (30, 31, 32, 39, 40, 41, 48, 49, 50)


def view_ids_for_step(step):
    if step not in (1, 2, 3):
        raise ValueError("angular_step must be 1, 2 or 3")
    return tuple(9*r+c for r in (4-step, 4, 4+step) for c in (4-step, 4, 4+step))


def load_views(scene, angular_step=1):
    images = []
    for i in view_ids_for_step(angular_step):
        with Image.open(Path(scene) / f"input_Cam{i:03d}.png") as im:
            images.append(np.asarray(im.convert("RGB")))
    return np.stack(images).transpose(0, 3, 1, 2).copy()


def augment_geometry(x, d, flip_x=False, flip_y=False, transpose=False):
    """Flip BOTH angular and spatial axes; disparity sign remains unchanged."""
    a = x.reshape(3, 3, 3, *x.shape[-2:])  # angular_y, angular_x, RGB, H, W
    if flip_x:
        a, d = a[:, ::-1, :, :, ::-1], d[:, ::-1]
    if flip_y:
        a, d = a[::-1, :, :, ::-1, :], d[::-1, :]
    if transpose:
        a, d = a.transpose(1, 0, 2, 4, 3), d.T
    return a.reshape(9, 3, *d.shape).copy(), d.copy()


def epinet_scaled_crop(x, d, y, z, crop, scale):
    """Spatial decimation changes disparity in pixels by the SAME factor."""
    return (x[..., y:y+scale*crop:scale, z:z+scale*crop:scale],
            d[y:y+scale*crop:scale, z:z+scale*crop:scale].copy()/scale)


def epinet_geometry(x, d, operation):
    if operation == 4:
        return augment_geometry(x, d, transpose=True)
    a = x.reshape(3, 3, 3, *d.shape)
    a = np.rot90(np.rot90(a, operation, axes=(0, 1)), operation, axes=(3, 4))
    return a.reshape(9, 3, *d.shape).copy(), np.rot90(d, operation).copy()


class AngularMix(Dataset):
    """Train with step 1/2 views in canonical step-2 disparity units."""
    def __init__(self, primary, secondary):
        manifests = [dict(dataset.manifest) for dataset in (primary, secondary)]
        for manifest, step in zip(manifests, (2, 1)):
            if manifest.pop("angular_step") != step or manifest.pop("view_ids") != list(view_ids_for_step(step)):
                raise ValueError("Angular mixing requires step-2 and step-1 caches")
        if manifests[0] != manifests[1]:
            raise ValueError("Angular mixing must preserve scene splits, GT and calibration")
        if any(getattr(primary, key) != getattr(secondary, key) for key in
               ("crop", "samples", "seed", "augmentation", "scenes")) or primary.crop is None:
            raise ValueError("Angular mixing requires matching training samplers")
        self.primary, self.secondary = primary, secondary
        self.scenes = primary.scenes
        self.epoch = 0

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, index):
        self.primary.epoch = self.secondary.epoch = self.epoch
        use_small = np.random.default_rng(742 + 1000003*self.epoch + index).random() < .5
        x, y, scene = (self.secondary if use_small else self.primary)[index]
        return x, y*.5 if use_small else y, scene


class HCI3x3(Dataset):
    def __init__(self, manifest, split, crop=None, samples=4000, seed=42, augmentation="legacy"):
        self.base = Path(manifest).resolve().parent
        self.manifest = json.loads(Path(manifest).read_text())
        self.scenes = [s for s in self.manifest["scenes"] if s["split"] == split]
        if not self.scenes:
            raise ValueError(f"Empty split: {split}")
        if any(not s["has_gt"] for s in self.scenes):
            raise ValueError("Hidden test GT is unavailable; use infer to export predictions")
        self.crop, self.samples, self.seed, self.epoch = crop, samples, seed, 0
        self.augmentation = augmentation
        if augmentation not in ("legacy", "epinet", "rgb_masked"):
            raise ValueError("Unknown augmentation")
        if crop and augmentation == "epinet" and any(3*crop > min(s['height'], s['width']) for s in self.scenes):
            raise ValueError("EPINET scale-3 sampling requires 3*crop to fit the source")
        if crop is not None and (crop < 32 or crop % 8 or any(crop > min(s['height'], s['width']) for s in self.scenes)):
            raise ValueError("Crop must be a multiple of 8, >=32, and fit each scene")
        self.cache = {}
        self.masks = {}

    def __len__(self):
        return self.samples if self.crop else len(self.scenes)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch*1000003 + index)
        if self.crop and self.augmentation == "epinet":
            return self.epinet_sample(rng)
        si = int(rng.integers(len(self.scenes))) if self.crop else index
        s = self.scenes[si]
        if si not in self.cache:
            p = self.base / s["cache"]
            self.cache[si] = (np.load(p/"views.npy", mmap_mode="r"),
                              np.load(p/"disp.npy", mmap_mode="r"))
            if self.crop and self.augmentation == "rgb_masked":
                self.masks[si] = np.load(p/"invalid.npy", mmap_mode="r") if s.get("invalid_mask") else None
        x, d = self.cache[si]
        if self.crop:
            h, w = d.shape
            for _ in range(128):
                y, z = int(rng.integers(h-self.crop+1)), int(rng.integers(w-self.crop+1))
                target = np.array(d[y:y+self.crop, z:z+self.crop], copy=True)
                mask = self.masks.get(si)
                if mask is not None:
                    target[mask[y:y+self.crop, z:z+self.crop]] = np.nan
                if np.isfinite(target[8:-8, 8:-8]).any():
                    break
            else:
                raise RuntimeError(f"No valid training crop in {s['name']}")
            x, d = x[..., y:y+self.crop, z:z+self.crop], target
            x, d = augment_geometry(x, d, *rng.integers(0, 2, 3).astype(bool))
        x = np.array(x, dtype=np.float32)/255
        if self.crop:
            # Identical photometric change for every view preserves correspondence.
            x = np.clip(x**rng.uniform(0.8, 1.2)*rng.uniform(0.85, 1.15), 0, 1)
        return torch.from_numpy(x.reshape(27, *d.shape).copy()), torch.from_numpy(np.array(d, copy=True))[None], si

    def epinet_sample(self, rng):
        weights = np.array([1 if s["name"] in ("kitchen", "museum", "vinyl") else 4 for s in self.scenes])
        for _ in range(128):
            si = int(rng.choice(len(self.scenes), p=weights/weights.sum()))
            s = self.scenes[si]
            p = self.base/s["cache"]
            if si not in self.cache:
                self.cache[si] = (np.load(p/"views.npy", mmap_mode="r"), np.load(p/"disp.npy", mmap_mode="r"))
                self.masks[si] = np.load(p/"invalid.npy", mmap_mode="r") if s.get("invalid_mask") else None
            x, d = self.cache[si]
            scale = int(rng.choice((1, 2, 3), p=np.array((8, 6, 3))/17))
            y, z = (int(rng.integers(n-scale*self.crop+1)) for n in d.shape)
            x, d = epinet_scaled_crop(x, d, y, z, self.crop, scale)
            mask = self.masks[si]
            if mask is not None:
                d[mask[y:y+scale*self.crop:scale, z:z+scale*self.crop:scale]] = np.nan
            if not np.isfinite(d[8:-8, 8:-8]).any():
                continue
            color = (0.05+rng.random(3)).astype(np.float32)
            color /= color.sum()
            gray = (np.asarray(x, dtype=np.float32)*color[None, :, None, None]).sum(1)/255
            if np.abs(gray[4]-gray[4, self.crop//2, self.crop//2]).mean() < .01:
                continue
            gray = gray**rng.uniform(.8, 1.2)
            x = np.repeat(gray[:, None], 3, axis=1)
            x, d = epinet_geometry(x, d, int(rng.integers(5)))
            return torch.from_numpy(x.reshape(27, *d.shape)), torch.from_numpy(d)[None], si
        raise RuntimeError("Unable to sample a textured patch with valid EPINET training labels")
