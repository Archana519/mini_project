"""
Evaluation protocol for EVERY model -- produces the same numbers as the
original sklearn-based validate(), but accumulates a confusion matrix on the
GPU per batch instead of building Python lists of ~500M pixels
(much faster, and uses almost no RAM).

Semantics reproduced exactly from the original:
  * pixels whose ground truth is 0 (no-data) are ignored
  * accuracy  = correct / all valid pixels            (sklearn accuracy_score)
  * precision = macro over classes 1..7               (sklearn precision_score, labels=1..7)
  * recall    = macro over classes 1..7; a pixel predicted as 0 counts as a miss
  * F1        = macro mean of per-class F1            (sklearn f1_score, average="macro")
  * IoU/Dice  = from the 7x7 confusion matrix over labels 1..7 (as before)
"""
import numpy as np
import torch

from config import cfg


def compute_iou_from_cm(cm):
    ious = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp) / float(denom) if denom > 0 else 0.0)
    return float(np.mean(ious)), ious


def compute_dice_from_cm(cm):
    dices = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        dices.append(float(2 * tp) / float(denom) if denom > 0 else 0.0)
    return float(np.mean(dices)), dices


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


@torch.no_grad()
def validate(model, loader, max_batches=None):
    model.eval()
    C = cfg.NUM_CLASSES
    cm_full = torch.zeros(C, C, dtype=torch.int64, device=cfg.DEVICE)  # rows = true, cols = pred

    for b, (images, masks) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        out = model(images.to(cfg.DEVICE))
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = out.argmax(1).flatten()
        true = masks.to(cfg.DEVICE).flatten()
        valid = true != cfg.IGNORE_INDEX
        cm_full += torch.bincount(true[valid] * C + pred[valid],
                                  minlength=C * C).view(C, C)

    full = cm_full.cpu().numpy().astype(np.float64)
    cm = full[1:, 1:]                      # labels 1..7 only (original IoU/Dice matrix)
    tp = np.diag(cm)
    pred_count = full[1:, 1:].sum(0)       # predictions of class c among valid pixels
    true_count = full[1:, :].sum(1)        # all valid pixels of class c (incl. predicted as 0)

    per_prec = _safe_div(tp, pred_count)
    per_rec = _safe_div(tp, true_count)
    per_f1 = _safe_div(2 * per_prec * per_rec, per_prec + per_rec)

    total_valid = full[1:, :].sum()
    accuracy = float(tp.sum() / total_valid) if total_valid > 0 else 0.0
    miou, class_iou = compute_iou_from_cm(cm)
    mdice, class_dice = compute_dice_from_cm(cm)

    return dict(accuracy=accuracy,
                precision=float(per_prec.mean()),
                recall=float(per_rec.mean()),
                f1=float(per_f1.mean()),
                mIoU=miou, mDice=mdice,
                class_iou=class_iou, class_dice=class_dice,
                cm=cm.astype(np.int64))


def print_metrics(m, epoch=None, tag=None):
    label = tag or (f"Epoch {epoch}" if epoch else "Final")
    bar = "=" * 56
    print(f"\n{bar}\n  {label} -- Validation Metrics\n{bar}")
    for k in ["accuracy", "precision", "recall", "f1", "mIoU", "mDice"]:
        print(f"  {k:<10}: {m[k]:.4f}")
    print(f"\n  {'Class':<14} {'IoU':>7}  {'Dice':>7}")
    print(f"  {'-'*32}")
    for i, name in enumerate(cfg.CLASS_NAMES[1:]):
        print(f"  {name:<14} {m['class_iou'][i]:>7.4f}  {m['class_dice'][i]:>7.4f}")
    print(bar + "\n")
