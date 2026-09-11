import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import view_ids_for_step, load_views
from .geometry import read_pfm, hci_calibration, disparity_to_depth


def main(argv=None):
    p = argparse.ArgumentParser(description="Audit HCI and cache central 3x3 views at the selected angular step")
    p.add_argument("--root", default="hclnew")
    p.add_argument("--output", default="data/hci3x3")
    p.add_argument("--angular-step", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--protocol", choices=("legacy", "epinet"), default="legacy")
    args = p.parse_args(argv)
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    if (out/"manifest.json").exists():
        previous = json.loads((out/"manifest.json").read_text())
        if previous["angular_step"] != args.angular_step or previous.get("protocol", "legacy") != args.protocol:
            raise ValueError("Use a separate output directory when changing angular step")
    scenes = []
    for category in ("additional", "training", "stratified", "test"):
        dirs = sorted(x for x in (root/category).iterdir() if x.is_dir())
        if not dirs:
            raise ValueError(f"No scenes in {root/category}")
        for scene in dirs:
            split = {"training": "holdout", "stratified": "stress", "test": "test"}.get(category)
            if category == "additional":
                split = "val" if scene.name in ("museum", "vinyl") else "train"
            if args.protocol == "epinet":
                split = "train" if category == "additional" else "test" if category == "test" else "val"
            x, c = load_views(scene, args.angular_step), hci_calibration(scene/"parameters.cfg")
            if x.shape != (9, 3, c["height"], c["width"]):
                raise ValueError(f"Unexpected image shape: {scene}")
            dest = out/category/scene.name
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest/"views.npy", x)
            has_gt = (scene/"gt_disp_lowres.pfm").exists()
            if split != "test" and not has_gt:
                raise ValueError(f"Missing GT: {scene}")
            record = dict(name=scene.name, category=category, split=split, cache=f"{category}/{scene.name}",
                          has_gt=has_gt, width=c["width"], height=c["height"], calibration=c)
            if args.protocol == "epinet" and category == "additional" and scene.name in ("kitchen", "museum", "vinyl"):
                mask_path = Path(__file__).resolve().parent.parent/"assets/epinet_masks"/(scene.name+".png")
                with Image.open(mask_path) as im:
                    mask = np.asarray(im.convert("RGBA"))[..., 3] > 0
                if mask.shape != x.shape[-2:]:
                    raise ValueError(f"Invalid reflection mask shape: {mask_path}")
                np.save(dest/"invalid.npy", mask)
                record["invalid_mask"] = True
            if has_gt:
                d, z = read_pfm(scene/"gt_disp_lowres.pfm"), read_pfm(scene/"gt_depth_lowres.pfm")
                if d.shape != x.shape[-2:] or z.shape != d.shape or not np.isfinite(d).any():
                    raise ValueError(f"Invalid ground truth shape/content: {scene}")
                np.save(dest/"disp.npy", d)
                np.save(dest/"depth.npy", z)
                pred = disparity_to_depth(d, c)
                valid = np.isfinite(pred) & np.isfinite(z) & (z > 0)
                err = float(np.median(np.abs(pred[valid]-z[valid])/z[valid])) if valid.any() else float("inf")
                record.update(disp_min=float(np.nanmin(d)), disp_max=float(np.nanmax(d)),
                              depth_formula_median_relative_error=err)
                if err > 1e-3:
                    raise ValueError(f"Depth unit/formula mismatch: {scene}, relative error={err}")
            scenes.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    report = dict(version=1, angular_shape=[3, 3], view_ids=view_ids_for_step(args.angular_step), angular_step=args.angular_step,
                  input_layout="N,27,H,W; row-major views, RGB per view; float32 [0,1]",
                  validation_scenes=["museum", "vinyl"], scenes=scenes)
    if args.protocol == "epinet":
        report.update(protocol="epinet", validation_scenes=[s["name"] for s in scenes if s["split"] == "val"])
    (out/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Manifest:", out/"manifest.json")


if __name__ == "__main__":
    main()
