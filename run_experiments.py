"""
Runs experiments one after another, each in its own subprocess (GPU memory is
fully released between models).

  * Finished runs (results/<exp>/result.json exists) are skipped automatically,
    so you can simply re-launch the same command after a crash / shutdown.
  * An interrupted run (results/<exp>/last.pth exists) is resumed from its
    last finished epoch.

Usage:
    python run_experiments.py --dry_run            # every model, 5+5 batches (~15-25 min)
    python run_experiments.py --group fast         # 7 CNN baselines + ablation_no_dinov2
    python run_experiments.py --group dino         # 7 DINOv2 ablations
    python run_experiments.py                      # everything in RUN_ORDER
    python run_experiments.py --only ablation_A ablation_B
"""
import argparse
import json
import os
import subprocess
import sys
import time

from config import cfg
from experiments import RUN_ORDER, FAST_RUNS, DINO_RUNS


def is_done(exp):
    p = os.path.join(cfg.RESULTS_DIR, exp, "result.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return not json.load(f).get("dry_run", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["fast", "dino", "all"], default="all")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if result.json exists")
    args = ap.parse_args()

    todo = args.only or {"fast": FAST_RUNS, "dino": DINO_RUNS, "all": RUN_ORDER}[args.group]
    todo = [e for e in todo if e not in args.skip]
    if not args.dry_run and not args.force:
        done = [e for e in todo if is_done(e)]
        if done:
            print(f"Already finished, skipping: {done}")
        todo = [e for e in todo if e not in done]
    print(f"Will run {len(todo)} experiments: {todo}\n")

    log = []
    for exp in todo:
        cmd = [sys.executable, "train.py", "--experiment", exp]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.dry_run:
            cmd += ["--dry_run"]
        elif os.path.exists(os.path.join(cfg.RESULTS_DIR, exp, "last.pth")):
            cmd += ["--resume"]
        print(f"\n{'='*70}\nRUNNING: {' '.join(cmd)}\n{'='*70}", flush=True)
        t0 = time.time()
        ret = subprocess.run(cmd)
        dt = (time.time() - t0) / 60
        status = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        log.append((exp, status, dt))
        with open("runs_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}  {exp:<22} {status:<18} {dt:7.1f} min"
                    f"{'  [dry-run]' if args.dry_run else ''}\n")
        if ret.returncode != 0:
            print(f"[WARN] {exp} {status} -- continuing with the next experiment.")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for exp, status, dt in log:
        print(f"  {exp:<22} {status:<18} {dt:7.1f} min")
    if any(s != "OK" for _, s, _ in log):
        sys.exit(1)


if __name__ == "__main__":
    main()
