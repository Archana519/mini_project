"""
Unified training entry point for EVERY experiment. Training loop, optimiser,
LR schedule, grad clipping and evaluation are identical to your original
StrongDINOv2Seg script; only the model (and, for ablations A-D, the loss) change.

Usage:
    python train.py --experiment unet
    python train.py --experiment ablation_A
    python train.py --experiment ablation_A --resume          # continue after a crash / shutdown
    python train.py --experiment unet --dry_run               # 5 train + 5 val batches, results_dryrun/
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import LoveDADataset, get_splits
from losses import TotalLoss, SimpleCELoss
from metrics import validate, print_metrics
from experiments import EXPERIMENTS, ABLATION_SETTINGS
from models.baselines import build_smp_baseline
from models.fcn8s import FCN8s
from models.hrnet import HRNetSeg
from models.proposed import ConfigurableSegModel

# every parameter whose name starts with one of these is "encoder" (0.1x LR)
ENCODER_PREFIXES = ("encoder.",            # DINOv2 (proposed / ablations)
                    "model.encoder.",      # smp baselines (inside PadToMultiple)
                    "backbone.",           # HRNet, FCN-8s (VGG + fc6/fc7)
                    "stem.", "layer1.", "layer2.", "layer3.", "layer4.")  # ResNet-50 ablation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_and_loss(entry):
    if entry["kind"] == "baseline":
        arch = entry["arch"]
        if arch == "fcn8s":
            model = FCN8s(cfg.NUM_CLASSES)
        elif arch == "hrnet":
            model = HRNetSeg(cfg.NUM_CLASSES)
        else:
            model = build_smp_baseline(arch, cfg.NUM_CLASSES,
                                       encoder_name=cfg.BASELINE_ENCODER,
                                       encoder_weights=cfg.BASELINE_ENCODER_WEIGHTS)
        return model, TotalLoss()          # same composite loss as the proposed model

    s = ABLATION_SETTINGS[entry["ablation"]]
    model = ConfigurableSegModel(num_classes=cfg.NUM_CLASSES,
                                 encoder_type=s["encoder_type"],
                                 use_fpn=s["use_fpn"], use_aspp=s["use_aspp"],
                                 use_residual_decoder=s["use_residual_decoder"])
    return model, (TotalLoss() if s["composite_loss"] else SimpleCELoss())


def build_optimizer(model, lr):
    enc, dec = [], []
    for n, p in model.named_parameters():
        (enc if n.startswith(ENCODER_PREFIXES) else dec).append(p)
    n_enc = sum(p.numel() for p in enc)
    n_dec = sum(p.numel() for p in dec)
    print(f"Param groups: encoder {n_enc/1e6:.1f}M @ {lr*cfg.ENCODER_LR_MULT:.1e} | "
          f"decoder/head {n_dec/1e6:.1f}M @ {lr:.1e}")
    if not enc:
        print("[WARN] no encoder params matched ENCODER_PREFIXES -- single LR used")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    return torch.optim.AdamW([
        {"params": enc, "lr": lr * cfg.ENCODER_LR_MULT},
        {"params": dec, "lr": lr},
    ], weight_decay=cfg.WEIGHT_DECAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="5 train + 5 val batches, 1 epoch, writes to results_dryrun/")
    args = ap.parse_args()

    if args.experiment == "proposed":
        raise SystemExit("'proposed' is not retrained -- run: python import_original.py")

    max_steps = max_val = None
    root = cfg.RESULTS_DIR
    if args.dry_run:
        args.epochs, max_steps, max_val, root = 1, 5, 5, "results_dryrun"

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    exp = args.experiment
    entry = EXPERIMENTS[exp]
    exp_dir = os.path.join(root, exp)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n{'#'*70}\n# EXPERIMENT: {exp}   {entry}\n{'#'*70}\nDevice: {cfg.DEVICE}")

    (tr_i, tr_m), (va_i, va_m) = get_splits()
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(LoveDADataset(tr_i, tr_m, augment=True),
                              batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(LoveDADataset(va_i, va_m, augment=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print(f"Train images: {len(tr_i)} | Val images: {len(va_i)}")

    model, criterion = build_model_and_loss(entry)
    model = model.to(cfg.DEVICE)
    optimizer = build_optimizer(model, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)

    total_p = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_p:,} | Trainable: {trainable:,}")

    history, best_miou, best_metrics, start_epoch, prev_time = [], 0.0, None, 1, 0.0
    last_path = os.path.join(exp_dir, "last.pth")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        history, best_miou, best_metrics = st["history"], st["best_miou"], st["best_metrics"]
        start_epoch, prev_time = st["epoch"] + 1, st["elapsed"]
        print(f"Resumed from epoch {st['epoch']} (best mIoU so far {best_miou:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n_steps, te = 0.0, 0, time.time()
        pbar = tqdm(train_loader, desc=f"[{exp}] Epoch {epoch}/{args.epochs}")
        for step, (images, masks) in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                break
            images, masks = images.to(cfg.DEVICE), masks.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.AMP):
                outputs = model(images)
            loss = criterion(outputs, masks)      # loss always in fp32 (as original)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item(); n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_min = (time.time() - te) / 60

        tv = time.time()
        metrics = validate(model, val_loader, max_batches=max_val)
        val_min = (time.time() - tv) / 60
        print_metrics(metrics, epoch=epoch)
        print(f"  epoch time: train {train_min:.1f} min + val {val_min:.1f} min")
        scheduler.step()

        history.append(dict(epoch=epoch, train_loss=running / max(n_steps, 1),
                            train_min=train_min, val_min=val_min,
                            **{k: metrics[k] for k in ["accuracy", "precision", "recall",
                                                       "f1", "mIoU", "mDice"]}))
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            best_metrics = {k: v for k, v in metrics.items() if k != "cm"}
            best_metrics["cm"] = metrics["cm"].tolist()
            best_metrics["epoch"] = epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "mIoU": best_miou},
                       os.path.join(exp_dir, "best_model.pth"))
            print(f"  -> best checkpoint saved (mIoU={best_miou:.4f})")

        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, history=history,
                        best_miou=best_miou, best_metrics=best_metrics,
                        elapsed=prev_time + time.time() - t0), last_path)

    elapsed = prev_time + time.time() - t0
    bm = best_metrics or {}
    result = dict(experiment=exp, config=entry, seed=args.seed, epochs=args.epochs,
                  total_params=total_p, trainable_params=trainable, train_time_sec=elapsed,
                  best_epoch=bm.get("epoch"), best_mIoU=best_miou,
                  best_mDice=bm.get("mDice"), best_accuracy=bm.get("accuracy"),
                  best_precision=bm.get("precision"), best_recall=bm.get("recall"),
                  best_f1=bm.get("f1"), best_class_iou=bm.get("class_iou"),
                  best_class_dice=bm.get("class_dice"), best_cm=bm.get("cm"),
                  history=history, dry_run=args.dry_run)
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    if os.path.exists(last_path) and not args.dry_run:
        os.remove(last_path)               # ~1 GB for DINOv2 runs; best_model.pth is kept
    print(f"\n[{exp}] DONE in {elapsed/60:.1f} min | best mIoU={best_miou:.4f} "
          f"(epoch {bm.get('epoch')})")


if __name__ == "__main__":
    main()
