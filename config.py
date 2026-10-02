"""
Central configuration shared by EVERY experiment (baselines, proposed model,
all ablation variants). Values below match your original StrongDINOv2Seg
training script exactly, so the imported original checkpoint and every new
run are trained/evaluated under identical conditions.

FIXES vs. the first version of this suite
  * CLASS_NAMES corrected to the official LoveDA label ids
    (0 = no-data/ignored, 1 = background, 2 = building, ... 7 = agriculture).
  * SEED added (the original script had none).
"""
import os
import torch


class CFG:
    # ---- Data -------------------------------------------------------------
    DATA_ROOT    = r"C:\Users\Dell\Downloads\archive"   # Train/Train/{Rural,Urban}, Val/Val/{Rural,Urban}
    DOMAINS      = ["Rural", "Urban"]   # same as the original run -- do NOT change unless you retrain everything
    IMAGE_SIZE   = 560                   # 560 = 40 x 14 (DINOv2 patch size); kept identical for ALL models
    BATCH_SIZE   = 2
    NUM_WORKERS  = 0                     # Windows + Jupyter safe. From a normal terminal you can try 2-4.
    NUM_CLASSES  = 8
    IGNORE_INDEX = 0                     # LoveDA "no-data"
    # Official LoveDA ids: index i in this list == mask value i
    CLASS_NAMES  = ["No-data", "Background", "Building", "Road",
                    "Water", "Barren", "Forest", "Agriculture"]

    # ---- Optimisation (identical across ALL models, == original script) ---
    EPOCHS       = 10
    LR           = 5e-5
    WEIGHT_DECAY = 1e-4
    ENCODER_LR_MULT = 0.1
    GRAD_CLIP    = 1.0
    SEED         = 42
    AMP          = False   # original run was fp32; keep False so all rows stay comparable

    # ---- Composite loss (== original script) ------------------------------
    DICE_SMOOTH  = 1e-6
    CE_W         = 0.5
    DICE_W       = 0.4
    EDGE_W       = 0.1
    AUX_W        = 0.4

    # ---- Baselines --------------------------------------------------------
    BASELINE_ENCODER         = "resnet34"
    BASELINE_ENCODER_WEIGHTS = "imagenet"

    # ---- Your already-trained mini-project checkpoint ---------------------
    ORIGINAL_CKPT = r"C:\Users\Dell\checkpoints\best_model.pth"   # <-- EDIT to where your notebook saved it

    # ---- Output -----------------------------------------------------------
    RESULTS_DIR  = "results"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
