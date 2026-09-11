import math

import numpy as np

from .geometry import disparity_to_depth, navigation_outputs


def training_loss(output, target, border=8, mode="legacy"):
    import torch
    import torch.nn.functional as F
    output, target = output.float(), target.float()
    if border:
        output = output[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    d, s = output[:, :1], output[:, 1:2]
    mask = torch.isfinite(target)
    if not mask.any():
        raise ValueError("No valid training pixels")
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    error = d-safe_target
    if mode not in ("legacy", "balanced"):
        raise ValueError(f"Unknown loss mode: {mode}")
    # In balanced mode uncertainty cannot attenuate the disparity-head gradient.
    nll_error = error.abs().detach() if mode == "balanced" else error.abs()
    nll = (nll_error*torch.exp(-s)+s+math.log(2))[mask].mean()
    robust = F.smooth_l1_loss(d[mask], safe_target[mask], beta=0.02 if mode == "balanced" else 0.1)
    soft_badpix = torch.sigmoid((error[mask].abs()-0.07)/0.02).mean()
    grad = d.new_zeros(())
    for axis in (-1, -2):
        e = torch.diff(error, dim=axis)
        pair = mask.narrow(axis, 0, mask.shape[axis]-1) & mask.narrow(axis, 1, mask.shape[axis]-1)
        if pair.any():
            grad = grad + e[pair].abs().mean()
    loss = (robust+0.1*soft_badpix+0.1*grad+0.05*nll if mode == "balanced"
            else nll+0.5*robust+0.1*grad)
    return loss, dict(loss=float(loss.detach()), nll=float(nll.detach()),
                      disparity_loss=float(robust.detach()), soft_badpix007=float(soft_badpix.detach()),
                      gradient_loss=float(grad.detach()), train_epe_px=float(error[mask].abs().mean().detach()))


def evaluate_maps(output, gt, calibration, border=11, near_m=2., max_scale=0.15):
    if border < 0 or border*2 >= min(gt.shape):
        raise ValueError("Invalid evaluation border")
    disp, s = output
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite network prediction; do not silently exclude model failures")
    mask = np.isfinite(gt)
    if border:
        mask[:border] = mask[-border:] = False
        mask[:, :border] = mask[:, -border:] = False
    if not mask.any():
        raise ValueError("No valid evaluation pixels")
    error_map = np.abs(disp-gt)
    e, b = error_map[mask].astype(np.float64), np.exp(s[mask].astype(np.float64))
    result = dict(epe_px=float(e.mean()), rmse_disp_px=float(np.sqrt((e*e).mean())),
                  mse_x100=float(100*(e*e).mean()),
                  nll=float(np.mean(e/b+np.log(2*b))),
                  laplace_3b_coverage_pct=float(100*np.mean(e <= 3*b)),
                  mean_scale_px=float(b.mean()), pixels=int(mask.sum()))
    for t in (0.01, 0.03, 0.07, 0.1, 1.0):
        result[f"badpix_{t:g}_pct"] = float(100*np.mean(e > t))
    edge = np.zeros_like(mask)
    edge[:, 1:] |= np.abs(np.diff(gt, axis=1)) > 0.1
    edge[1:, :] |= np.abs(np.diff(gt, axis=0)) > 0.1
    result["edge_epe_px"] = float(error_map[edge & mask].mean()) if (edge & mask).any() else None
    z, zg = disparity_to_depth(disp, calibration), disparity_to_depth(gt, calibration)
    gt_ok = mask & np.isfinite(zg) & (zg > 0)
    ok = gt_ok & np.isfinite(z) & (z > 0)
    result["depth_valid_pct"] = float(100*ok.sum()/max(gt_ok.sum(), 1))
    if ok.any():
        result.update(depth_absrel=float(np.mean(np.abs(z[ok]-zg[ok])/zg[ok])),
                      depth_rmse_m=float(np.sqrt(np.mean((z[ok]-zg[ok])**2))),
                      depth_delta1_pct=float(100*np.mean(np.maximum(z[ok]/zg[ok], zg[ok]/z[ok]) < 1.25)))
    else:
        result.update(depth_absrel=None, depth_rmse_m=None, depth_delta1_pct=None)
    _, accepted, _ = navigation_outputs(disp, s, calibration, max_scale=max_scale, border=border)
    accepted &= mask
    result["accepted_pct"] = float(100*accepted.sum()/mask.sum())
    result["accepted_epe_px"] = float(error_map[accepted].mean()) if accepted.any() else None
    # Pixel-level near-surface proxy, NOT ground-filtered obstacle/trajectory success.
    truth_near = gt_ok & (zg < near_m)
    predicted_near = gt_ok & accepted & (z < near_m)
    tp = int((truth_near & predicted_near).sum())
    result["near_surface_recall_pct"] = 100*tp/int(truth_near.sum()) if truth_near.any() else None
    result["near_surface_precision_pct"] = 100*tp/int(predicted_near.sum()) if predicted_near.any() else None
    return result


def macro_average(rows):
    keys = set().union(*(r.keys() for r in rows))
    return {k: float(np.mean(v)) if (v := [r[k] for r in rows if isinstance(r.get(k), (int, float))]) else None
            for k in sorted(keys) if k not in ("scene", "pixels")}
