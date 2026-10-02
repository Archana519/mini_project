"""
STEP 3 -- Import your already-trained StrongDINOv2Seg checkpoint as the
"Proposed" row (Comment 4) and the "E / Full" row (Comment 5).

It (1) loads the checkpoint into ConfigurableSegModel with strict=True, which
proves the refactored model is layer-for-layer your original model, and
(2) re-evaluates it with the shared validate(), writing results/proposed/result.json.

Expected: mIoU = 0.5380 (your epoch-4 best), params = 93,835,616.
Runtime: ~6-10 min on GPU (one pass over 1,669 val images).
"""
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset import LoveDADataset, get_splits
from metrics import validate, print_metrics
from models.proposed import ConfigurableSegModel


def main():
    if not os.path.exists(cfg.ORIGINAL_CKPT):
        raise SystemExit(f"Checkpoint not found: {cfg.ORIGINAL_CKPT}\n"
                         f"Edit ORIGINAL_CKPT in config.py")
    ckpt = torch.load(cfg.ORIGINAL_CKPT, map_location=cfg.DEVICE, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model = ConfigurableSegModel(cfg.NUM_CLASSES, encoder_type="dinov2",
                                 use_fpn=True, use_aspp=True, use_residual_decoder=True)
    model.load_state_dict(state, strict=True)
    model.to(cfg.DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"strict load OK | params = {n:,} (expected 93,835,616)")

    _, (va_i, va_m) = get_splits()
    loader = DataLoader(LoveDADataset(va_i, va_m), batch_size=cfg.BATCH_SIZE,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    t = time.time()
    m = validate(model, loader)
    print_metrics(m, tag="Imported original checkpoint")
    print(f"validation took {(time.time()-t)/60:.1f} min | "
          f"mIoU = {m['mIoU']:.4f} (expected 0.5380 from your notebook)")

    out_dir = os.path.join(cfg.RESULTS_DIR, "proposed")
    os.makedirs(out_dir, exist_ok=True)
    res = dict(experiment="proposed", config={"kind": "proposed", "ablation": "E"},
               seed=None, epochs=10, total_params=n, trainable_params=n,
               train_time_sec=None, best_epoch=ckpt.get("epoch"),
               best_mIoU=m["mIoU"], best_mDice=m["mDice"], best_accuracy=m["accuracy"],
               best_precision=m["precision"], best_recall=m["recall"], best_f1=m["f1"],
               best_class_iou=m["class_iou"], best_class_dice=m["class_dice"],
               best_cm=m["cm"].tolist(), history=[], imported_from=cfg.ORIGINAL_CKPT)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"wrote {os.path.join(out_dir, 'result.json')}")


if __name__ == "__main__":
    main()
