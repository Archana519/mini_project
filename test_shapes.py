"""
STEP 1 -- structural smoke test at the REAL image size (560x560).
No dataset needed. Runs on CPU in ~2-4 minutes.

Checks for every model that will be trained:
  * output shape == (1, 8, 560, 560)          (the old version tested 224 px and
                                                hid the smp /32 crash)
  * the optimiser actually finds encoder parameters (0.1x LR group)
DINOv2 variants load facebook/dinov2-base from your HF cache (downloaded once
by your original run); if it cannot be loaded they are reported as SKIP.
"""
import torch

from config import cfg
from experiments import EXPERIMENTS, ABLATION_SETTINGS
from models.baselines import build_smp_baseline
from models.fcn8s import FCN8s
from models.hrnet import HRNetSeg
from models.proposed import ConfigurableSegModel
from train import ENCODER_PREFIXES

S = cfg.IMAGE_SIZE
DUMMY = torch.randn(1, 3, S, S)
EXPECTED = (1, cfg.NUM_CLASSES, S, S)


def build(name):
    e = EXPERIMENTS[name]
    if e["kind"] == "baseline":
        if e["arch"] == "fcn8s":
            return FCN8s(cfg.NUM_CLASSES, pretrained_backbone=False)
        if e["arch"] == "hrnet":
            return HRNetSeg(cfg.NUM_CLASSES, pretrained=False)
        return build_smp_baseline(e["arch"], cfg.NUM_CLASSES, cfg.BASELINE_ENCODER, None)
    s = ABLATION_SETTINGS[e["ablation"]]
    return ConfigurableSegModel(cfg.NUM_CLASSES, encoder_type=s["encoder_type"],
                                use_fpn=s["use_fpn"], use_aspp=s["use_aspp"],
                                use_residual_decoder=s["use_residual_decoder"],
                                pretrained_cnn=False)


def main():
    print(f"Smoke test at {tuple(DUMMY.shape)}\n")
    ok_all, skipped = True, []
    for name in EXPERIMENTS:
        try:
            m = build(name).eval()
        except Exception as ex:
            if "dinov2" in str(ex).lower() or "huggingface" in str(ex).lower() \
                    or "connection" in str(ex).lower():
                print(f"  [SKIP] {name:<22} cannot load DINOv2 weights: {type(ex).__name__}")
                skipped.append(name)
                continue
            print(f"  [FAIL] {name:<22} build error: {type(ex).__name__}: {ex}")
            ok_all = False
            continue
        try:
            with torch.no_grad():
                out = m(DUMMY)
            out = out[0] if isinstance(out, (list, tuple)) else out
            enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith(ENCODER_PREFIXES))
            tot = sum(p.numel() for p in m.parameters())
            shape_ok = tuple(out.shape) == EXPECTED
            enc_ok = enc > 0
            ok = shape_ok and enc_ok
            ok_all &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<22} out={tuple(out.shape)} "
                  f"params={tot/1e6:6.1f}M encoder={enc/1e6:6.1f}M"
                  + ("" if enc_ok else "  <-- no encoder params matched!"))
        except Exception as ex:
            print(f"  [FAIL] {name:<22} forward error: {type(ex).__name__}: {ex}")
            ok_all = False
        del m

    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED -- fix before training")
    if skipped:
        print(f"Skipped (DINOv2 not loadable here): {skipped}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
