"""
STEP 2 -- measure the REAL training time of every experiment on YOUR GPU
before committing ~1.5 days of compute. Takes ~10-15 minutes.

For each model: 5 warm-up + 15 timed train steps and 15 timed val steps at
batch 2 / 560 px (same as real training), plus the real data-loading cost
(num_workers=0 means loading happens serially inside every step).
It then extrapolates to 1261 train steps + 835 val steps per epoch.
Also reports peak GPU memory -> tells you in advance if anything will OOM.

Usage:
    python estimate_time.py                  # all runs in RUN_ORDER
    python estimate_time.py --only unet hrnet
"""
import argparse
import math
import time

import torch

from config import cfg
from dataset import LoveDADataset, get_splits
from experiments import EXPERIMENTS, RUN_ORDER
from train import build_model_and_loss, build_optimizer

WARMUP, TIMED = 5, 15


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def data_time_per_step(ds, n=20):
    t = time.time()
    for i in range(n):
        ds[i % len(ds)]
    return (time.time() - t) / n * cfg.BATCH_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- run this on the training GPU.")

    (tr_i, tr_m), (va_i, va_m) = get_splits()
    n_train = math.ceil(len(tr_i) / cfg.BATCH_SIZE)
    n_val = math.ceil(len(va_i) / cfg.BATCH_SIZE)
    tr_load = data_time_per_step(LoveDADataset(tr_i, tr_m, augment=True))
    va_load = data_time_per_step(LoveDADataset(va_i, va_m, augment=False))
    print(f"steps/epoch: train {n_train}, val {n_val} | data loading "
          f"{tr_load:.2f}s/train-step, {va_load:.2f}s/val-step\n")

    x = torch.rand(cfg.BATCH_SIZE, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=cfg.DEVICE)
    y = torch.randint(0, cfg.NUM_CLASSES, (cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                      device=cfg.DEVICE)
    rows, total_h = [], 0.0
    for name in (args.only or RUN_ORDER):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            model, crit = build_model_and_loss(EXPERIMENTS[name])
            model = model.to(cfg.DEVICE)
            opt = build_optimizer(model, cfg.LR)
            model.train()
            for i in range(WARMUP + TIMED):
                if i == WARMUP:
                    sync(); t = time.time()
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            sync(); tr_step = (time.time() - t) / TIMED
            model.eval()
            with torch.no_grad():
                for i in range(WARMUP + TIMED):
                    if i == WARMUP:
                        sync(); t = time.time()
                    model(x).argmax(1)
            sync(); va_step = (time.time() - t) / TIMED
            mem = torch.cuda.max_memory_allocated() / 1024**3
            ep_min = (n_train * (tr_step + tr_load) + n_val * (va_step + va_load)) / 60
            run_h = ep_min * args.epochs / 60
            total_h += run_h
            rows.append((name, tr_step + tr_load, ep_min, run_h, mem, "OK"))
        except torch.cuda.OutOfMemoryError:
            rows.append((name, 0, 0, 0, 0, "OOM"))
        except Exception as ex:
            rows.append((name, 0, 0, 0, 0, f"ERR {type(ex).__name__}: {ex}"[:60]))
        finally:
            model = opt = None
            torch.cuda.empty_cache()
        r = rows[-1]
        print(f"  {r[0]:<20} {r[1]:5.2f} s/it  {r[2]:6.1f} min/epoch  "
              f"{r[3]:5.1f} h/run  peak {r[4]:4.1f} GB  {r[5]}")

    print(f"\nTOTAL for {len(rows)} runs x {args.epochs} epochs: {total_h:.1f} h "
          f"(+ ~10 min for import_original.py)")


if __name__ == "__main__":
    main()
