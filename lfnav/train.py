import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import AngularMix, HCI3x3, view_ids_for_step
from .metrics import training_loss, evaluate_maps, macro_average
from .model import build_model, initialize_model
from .robustness import augment_robustness
from .synthetic import random_geometry


def log(path, row):
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text+"\n")


def checkpoint_improvements(metrics, best_epe, best_badpix, selection_metric):
    """Keep both objectives; best.pt follows the explicitly selected objective."""
    if selection_metric not in ("epe_px", "badpix_0.07_pct"):
        raise ValueError("selection_metric must be epe_px or badpix_0.07_pct")
    epe_better = metrics["epe_px"] < best_epe
    badpix_better = metrics["badpix_0.07_pct"] < best_badpix
    names = ["last.pt"]
    if epe_better:
        names.append("best_epe.pt")
    if badpix_better:
        names.append("best_badpix007.pt")
    primary_better = epe_better if selection_metric == "epe_px" else badpix_better
    if primary_better:
        names.append("best.pt")
    return names


def check_transfer_manifest(state, config, target_hash):
    if state["manifest_sha256"] == target_hash:
        return
    # Only angular sampling may change. Scene splits, GT statistics and physical
    # calibration must remain identical, and the source manifest must be intact.
    source_path = Path(state["config"]["manifest"])
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != state["manifest_sha256"]:
        raise ValueError("Source manifest is missing or has changed since checkpoint")
    source = json.loads(source_bytes)
    target = json.loads(Path(config["manifest"]).read_text())
    old_step, new_step = source["angular_step"], target["angular_step"]
    if old_step == new_step or {old_step, new_step} != {1, 2}:
        raise ValueError("Fine-tuning manifest differs beyond supported angular transfer")
    for manifest in (source, target):
        if manifest["view_ids"] != list(view_ids_for_step(manifest["angular_step"])):
            raise ValueError("Unexpected view selection in transfer manifest")
        manifest.pop("view_ids")
        manifest.pop("angular_step")
    if source != target:
        raise ValueError("Angular transfer must preserve all scene splits, GT and calibration")


@torch.inference_mode()
def validate(model, dataset, device, border=11):
    model.eval()
    rows = []
    for x, y, i in dataset:
        x = x[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        row = evaluate_maps(pred[0].float().cpu().numpy(), y[0].numpy(),
                            dataset.scenes[i]["calibration"], border=border)
        row.update(scene=dataset.scenes[i]["name"], single_pass_model_ms=elapsed)
        rows.append(row)
    return macro_average(rows), rows


def main(default_config="configs/train_3090.yaml"):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--resume", help="Resume identical config, optimizer and scheduler")
    start.add_argument("--init-from", help="Initialize learned weights only, then start a new fine-tuning run")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    robustness_mode = cfg.get("robustness_mode", "none")
    if robustness_mode not in ("none", "noise", "contrast", "noise_contrast", "synthetic_combined"):
        raise ValueError("Unknown robustness_mode")
    loss_mode = cfg.get("loss_mode", "legacy")
    selection_metric = cfg.get("selection_metric", "epe_px")
    checkpoint_improvements({"epe_px": 0., "badpix_0.07_pct": 0.}, float("inf"), float("inf"), selection_metric)
    if loss_mode not in ("legacy", "balanced"):
        raise ValueError("loss_mode must be legacy or balanced")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install the matching server PyTorch or explicitly use --device cpu")
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = device.type == "cuda"
    out = Path(cfg["run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out/"last.pt").exists():
        raise ValueError("Run already exists; use --resume or a new run_dir")
    if not Path(cfg["manifest"]).exists() and cfg.get("dataset_root"):
        from .prepare import main as prepare_cache
        prepare_cache(["--root", cfg["dataset_root"], "--output", str(Path(cfg["manifest"]).parent),
                       "--angular-step", str(cfg.get("angular_step", 1)),
                       "--protocol", cfg.get("data_protocol", "legacy")])
    manifest_hash = hashlib.sha256(Path(cfg["manifest"]).read_bytes()).hexdigest()
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    angular_step = cfg.get("angular_step", 1)
    if manifest["angular_step"] != angular_step or manifest["view_ids"] != list(view_ids_for_step(angular_step)):
        raise ValueError("Config angular_step does not match cached view selection")
    if manifest.get("protocol", "legacy") != cfg.get("data_protocol", "legacy"):
        raise ValueError("Config data_protocol does not match the cache")
    if cfg.get("data_protocol") == "epinet":
        expected_train = set("antinous boardgames dishes greek kitchen medieval2 museum pens pillows platonic rosemary table tomb tower town vinyl".split())
        expected_val = set("backgammon dots pyramids stripes boxes cotton dino sideboard".split())
        if ({s["name"] for s in manifest["scenes"] if s["split"] == "train"} != expected_train or
                {s["name"] for s in manifest["scenes"] if s["split"] == "val"} != expected_val):
            raise ValueError("EPINET requires the official 16 training and 8 public evaluation scenes")
        if (cfg.get("architecture") not in ("two_layer_stem", "three_layer_stem") or angular_step not in (1, 2, 3) or
                (cfg.get("augmentation"), cfg.get("input_color")) not in (("epinet", "epinet_gray"), ("rgb_masked", "rgb"))):
            raise ValueError("Use two/three-layer stems, step 1/2/3 and paired epinet/epinet_gray or rgb_masked/rgb processing")
    train = HCI3x3(cfg["manifest"], "train", cfg["crop_size"], cfg["samples_per_epoch"], cfg["seed"],
                   augmentation=cfg.get("augmentation", "legacy"))
    val = HCI3x3(cfg["manifest"], "val")
    mix_hash = None
    if cfg.get("angular_mix_manifest"):
        if angular_step != 2:
            raise ValueError("Angular mixing requires step-2 validation and inference")
        mix_path = Path(cfg["angular_mix_manifest"])
        if not mix_path.exists() and cfg.get("dataset_root"):
            from .prepare import main as prepare_cache
            prepare_cache(["--root", cfg["dataset_root"], "--output", str(mix_path.parent),
                           "--angular-step", "1", "--protocol", cfg.get("data_protocol", "legacy")])
        mix_hash = hashlib.sha256(mix_path.read_bytes()).hexdigest()
        secondary = HCI3x3(mix_path, "train", cfg["crop_size"], cfg["samples_per_epoch"], cfg["seed"],
                           augmentation=cfg.get("augmentation", "legacy"))
        train = AngularMix(train, secondary)
    model = build_model(cfg).to(device)
    if not args.resume and not args.init_from and cfg.get("architecture") in ("two_layer_stem", "three_layer_stem"):
        # Identity stems are useful for transfer only; scratch training initializes
        # the additional convolutions randomly like the rest of the network.
        for layer in model.stem_second:
            layer[0].reset_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=cfg["min_lr"])
    amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(cfg["seed"])
    start_epoch, best, best_badpix, global_step = 0, float("inf"), float("inf"), 0
    initialization = dict(mode="scratch", pretrained_weights=False)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("Dataset manifest changed since checkpoint")
        if state.get("angular_mix_manifest_sha256") != mix_hash:
            raise ValueError("Angular mixing manifest changed since checkpoint")
        if state["config"] != cfg:
            raise ValueError("Resume requires the saved config, including scheduler length; start a new run to change it")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        generator.set_state(state["loader_rng"])
        start_epoch, best, global_step = state["epoch"]+1, state["best_epe"], state["global_step"]
        best_badpix = state.get("best_badpix007", float("inf"))
        initialization = state.get("initialization")
    elif args.init_from:
        state = torch.load(args.init_from, map_location="cpu", weights_only=True)
        check_transfer_manifest(state, cfg, manifest_hash)
        if state["config"]["width"] != cfg["width"]:
            raise ValueError("Fine-tuning width must match the source checkpoint")
        transfer = initialize_model(model, state, cfg)
        initialization = dict(checkpoint=str(args.init_from), source_epoch=state["epoch"]+1,
                              source_angular_step=state["config"].get("angular_step", 1), **transfer)
    # Workers restart each epoch, so deterministic epoch-dependent augmentation is propagated.
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], pin_memory=device.type == "cuda",
                        generator=generator, persistent_workers=False)
    (out/"config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = dict(torch=str(torch.__version__), python=platform.python_version(), platform=platform.platform(),
               device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               parameters=sum(x.numel() for x in model.parameters()), architecture=cfg.get("architecture", "baseline"),
               manifest_sha256=manifest_hash,
               angular_mix_manifest_sha256=mix_hash,
               angular_mix=bool(mix_hash), angular_mix_step1_probability=.5 if mix_hash else 0.,
               angular_mix_step1_target_scale=.5 if mix_hash else None,
               angular_step=angular_step, view_ids=manifest["view_ids"], disparity_units="original HCI adjacent-view pixels",
               data_protocol=cfg.get("data_protocol", "legacy"), augmentation=cfg.get("augmentation", "legacy"),
               input_color=cfg.get("input_color", "rgb"),
               robustness_mode=robustness_mode,
               train_scenes=[s["name"] for s in train.scenes], val_scenes=[s["name"] for s in val.scenes])
    (out/"environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(out/"metrics.jsonl", dict(event="start", **env))
    log(out/"metrics.jsonl", dict(event="objective", loss_mode=loss_mode,
                                   selection_metric=selection_metric, initialization=initialization))

    def save_progress(epoch, metrics):
        nonlocal best, best_badpix
        names = checkpoint_improvements(metrics, best, best_badpix, selection_metric)
        best = min(best, metrics["epe_px"])
        best_badpix = min(best_badpix, metrics["badpix_0.07_pct"])
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), config=cfg, epoch=epoch, best_epe=best,
                     best_badpix007=best_badpix, initialization=initialization, val_metrics=metrics,
                     global_step=global_step, manifest_sha256=manifest_hash,
                     angular_mix_manifest_sha256=mix_hash,
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     loader_rng=generator.get_state())
        for name in names:
            temporary = out/(name+".tmp")
            torch.save(state, temporary)
            temporary.replace(out/name)

    if args.init_from:
        # Validate and save the starting point so a worse fine-tune cannot replace best.pt.
        initial, rows = validate(model, val, device, cfg["eval_border"])
        for row in rows:
            log(out/"metrics.jsonl", dict(event="initial_val_scene", epoch=0, **row))
        log(out/"metrics.jsonl", dict(event="initial_val", epoch=0, **initial))
        save_progress(-1, initial)
    writer = None
    if cfg.get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out/"tensorboard"))
    for epoch in range(start_epoch, cfg["epochs"]):
        train.epoch = epoch
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start, last = time.perf_counter(), time.perf_counter()
        sums, seen, skipped_steps = {}, 0, 0
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if robustness_mode == "synthetic_combined":
                x, y = random_geometry(x, y, step=angular_step)
            x = augment_robustness(x, "noise_contrast" if robustness_mode == "synthetic_combined" else robustness_mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(x)
                loss, values = training_loss(output, y, cfg["loss_border"], mode=loss_mode)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = bool(scaler.get_scale() < old_scale)
            skipped_steps += int(skipped)
            if not amp and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            batch = x.shape[0]
            seen += batch
            for k, v in values.items():
                sums[k] = sums.get(k, 0.)+v*batch
            global_step += 1
            if step % cfg["log_every"] == 0 or time.perf_counter()-last > 30:
                log(out/"metrics.jsonl", dict(event="train_step", epoch=epoch+1, step=step,
                    global_step=global_step, lr=optimizer.param_groups[0]["lr"],
                    grad_norm=float(norm) if torch.isfinite(norm) else None,
                    amp_step_skipped=skipped, **values))
                last = time.perf_counter()
        train_seconds = time.perf_counter()-start
        metrics, scene_rows = validate(model, val, device, cfg["eval_border"])
        for r in scene_rows:
            log(out/"metrics.jsonl", dict(event="val_scene", epoch=epoch+1, **r))
        if cfg.get("data_protocol") == "epinet":
            for category in ("training", "stratified"):
                names = {s["name"] for s in val.scenes if s["category"] == category}
                group_metrics = macro_average([r for r in scene_rows if r["scene"] in names])
                log(out/"metrics.jsonl", dict(event="val_group", epoch=epoch+1, group=category, **group_metrics))
        summary = dict(event="epoch", epoch=epoch+1, **{f"train/{k}": v/seen for k, v in sums.items()},
                       **{f"val/{k}": v for k, v in metrics.items()},
                       lr=optimizer.param_groups[0]["lr"], selection_metric=selection_metric,
                       amp_skipped_steps=skipped_steps, optimizer_updates=len(loader)-skipped_steps,
                       train_seconds=train_seconds, train_patches_per_second=seen/train_seconds,
                       peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0)
        log(out/"metrics.jsonl", summary)
        if writer:
            for k, v in summary.items():
                if isinstance(v, (float, int)):
                    writer.add_scalar(k, v, epoch+1)
            writer.flush()
        scheduler.step()
        save_progress(epoch, metrics)
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
