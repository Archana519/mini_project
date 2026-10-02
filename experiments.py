"""
Single source of truth for every experiment.

REDESIGN vs. the first version
  * Cumulative table now changes exactly ONE thing per row:
        A  DINOv2 only                       (plain CE)
        B  + FPN                             (plain CE)
        C  + ASPP                            (plain CE)
        D  + Residual Decoder                (plain CE)
        E  + Composite loss  == Full model   (your original checkpoint)
  * Duplicates removed: the old C == minus_resdec and D == minus_loss pairs
    trained the same model twice. Now:
        "- Composite Loss" row  <- ablation_D   (identical config)
        "Full" / "E" row        <- proposed     (your original checkpoint, imported)
"""

ABLATION_SETTINGS = {
    # cumulative build-up, one change per row
    "A": dict(encoder_type="dinov2", use_fpn=False, use_aspp=False, use_residual_decoder=False, composite_loss=False),
    "B": dict(encoder_type="dinov2", use_fpn=True,  use_aspp=False, use_residual_decoder=False, composite_loss=False),
    "C": dict(encoder_type="dinov2", use_fpn=True,  use_aspp=True,  use_residual_decoder=False, composite_loss=False),
    "D": dict(encoder_type="dinov2", use_fpn=True,  use_aspp=True,  use_residual_decoder=True,  composite_loss=False),
    "E": dict(encoder_type="dinov2", use_fpn=True,  use_aspp=True,  use_residual_decoder=True,  composite_loss=True),

    # single-component removal from the FULL model (all use the composite loss)
    "minus_dinov2": dict(encoder_type="resnet50", use_fpn=True,  use_aspp=True,  use_residual_decoder=True,  composite_loss=True),
    "minus_fpn":    dict(encoder_type="dinov2",   use_fpn=False, use_aspp=True,  use_residual_decoder=True,  composite_loss=True),
    "minus_aspp":   dict(encoder_type="dinov2",   use_fpn=True,  use_aspp=False, use_residual_decoder=True,  composite_loss=True),
    "minus_resdec": dict(encoder_type="dinov2",   use_fpn=True,  use_aspp=True,  use_residual_decoder=False, composite_loss=True),
}

EXPERIMENTS = {
    # ---------------- Comment 4: baselines ----------------
    "unet":          dict(kind="baseline", arch="unet"),
    "unetplusplus":  dict(kind="baseline", arch="unetplusplus"),
    "pspnet":        dict(kind="baseline", arch="pspnet"),
    "linknet":       dict(kind="baseline", arch="linknet"),
    "deeplabv3plus": dict(kind="baseline", arch="deeplabv3plus"),
    "hrnet":         dict(kind="baseline", arch="hrnet"),
    "fcn8s":         dict(kind="baseline", arch="fcn8s"),
    "proposed":      dict(kind="proposed", ablation="E"),   # imported, not retrained

    # ---------------- Comment 5: cumulative ----------------
    "ablation_A": dict(kind="proposed", ablation="A"),
    "ablation_B": dict(kind="proposed", ablation="B"),
    "ablation_C": dict(kind="proposed", ablation="C"),
    "ablation_D": dict(kind="proposed", ablation="D"),      # also the "- Composite Loss" row

    # ---------------- Comment 5: component removal --------
    "ablation_no_dinov2": dict(kind="proposed", ablation="minus_dinov2"),
    "ablation_no_fpn":    dict(kind="proposed", ablation="minus_fpn"),
    "ablation_no_aspp":   dict(kind="proposed", ablation="minus_aspp"),
    "ablation_no_resdec": dict(kind="proposed", ablation="minus_resdec"),
}

# Groups used by run_experiments.py. Fast CNN runs first, DINOv2 runs after.
FAST_RUNS = ["unet", "unetplusplus", "pspnet", "linknet", "deeplabv3plus",
             "hrnet", "fcn8s", "ablation_no_dinov2"]
DINO_RUNS = ["ablation_A", "ablation_B", "ablation_C", "ablation_D",
             "ablation_no_fpn", "ablation_no_aspp", "ablation_no_resdec"]
RUN_ORDER = FAST_RUNS + DINO_RUNS          # "proposed" is imported by import_original.py

# Display names used in the report tables
DISPLAY = {
    "unet": "U-Net", "unetplusplus": "UNet++", "pspnet": "PSPNet", "linknet": "LinkNet",
    "deeplabv3plus": "DeepLabV3+", "hrnet": "HRNet-W18", "fcn8s": "FCN-8s",
    "proposed": "Proposed (StrongDINOv2Seg)",
    "ablation_A": "A: DINOv2 only (CE)",
    "ablation_B": "B: + FPN (CE)",
    "ablation_C": "C: + ASPP (CE)",
    "ablation_D": "D: + Residual Decoder (CE)",
    "ablation_no_dinov2": "- DINOv2 (ResNet-50 encoder)",
    "ablation_no_fpn": "- FPN",
    "ablation_no_aspp": "- ASPP",
    "ablation_no_resdec": "- Residual Decoder",
}
