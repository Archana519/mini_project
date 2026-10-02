"""
STEP 7 -- build every table/figure for the rebuttal and the paper.
Safe to run at any time; unfinished experiments show as "N/A".

Outputs (results/summary/):
  table1_baselines.csv / .md / .tex         Comment 4: overall metrics + params + train time
  table2_classwise_iou.csv / .md / .tex     Comment 4: class-wise IoU of every model
  table3_ablation_cumulative.csv/.md/.tex   Comment 5: A -> B -> C -> D -> E with delta mIoU
  table4_ablation_removal.csv / .md / .tex  Comment 5: Full vs. -X with delta mIoU
  fig_*.png                                 bar charts, class-IoU heatmap, training curves
  REPORT.md                                 everything assembled
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import cfg
from experiments import DISPLAY

OUT = os.path.join(cfg.RESULTS_DIR, "summary")
os.makedirs(OUT, exist_ok=True)
CLASSES = cfg.CLASS_NAMES[1:]          # Background ... Agriculture (labels 1..7)
NOISE = 1.5                            # mIoU points; epoch-to-epoch spread seen in the original run


def load(exp):
    p = os.path.join(cfg.RESULTS_DIR, exp, "result.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = json.load(f)
    return None if r.get("dry_run") else r


def pct(v):
    return None if v is None else round(100 * v, 2)


def metrics_row(exp, name=None):
    r = load(exp)
    row = {"Model": name or DISPLAY.get(exp, exp)}
    if r is None:
        row.update({k: None for k in ["Params (M)", "Acc", "Prec", "Rec", "F1",
                                      "mIoU", "mDice", "Train (h)"]})
        return row, None
    row.update({"Params (M)": round(r["total_params"] / 1e6, 1),
                "Acc": pct(r["best_accuracy"]), "Prec": pct(r["best_precision"]),
                "Rec": pct(r["best_recall"]), "F1": pct(r["best_f1"]),
                "mIoU": pct(r["best_mIoU"]), "mDice": pct(r["best_mDice"]),
                "Train (h)": None if not r.get("train_time_sec") else round(r["train_time_sec"] / 3600, 2)})
    return row, r


def to_latex(df, caption, label):
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\hline",
             " & ".join(cols) + r" \\", r"\hline"]
    for _, row in df.iterrows():
        cells = ["--" if (v is None or (isinstance(v, float) and np.isnan(v))) else
                 (f"{v:+.2f}" if isinstance(v, float) and c.startswith("Δ")
                  else f"{v:.2f}" if isinstance(v, float) else str(v))
                 for c, v in row.items()]
        lines.append(" & ".join(cells).replace("%", r"\%") + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return ("\n".join(lines).replace("Δ", r"$\Delta$")
            .replace("✓", r"$\checkmark$").replace("✗", r"$\times$"))


def save(df, stem, caption):
    df.to_csv(os.path.join(OUT, stem + ".csv"), index=False)
    md = df.to_markdown(index=False, floatfmt=".2f").replace(" nan ", " N/A ")
    with open(os.path.join(OUT, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(OUT, stem + ".tex"), "w", encoding="utf-8") as f:
        f.write(to_latex(df, caption, "tab:" + stem) + "\n")
    return md


def bar(df, col, title, path):
    d = df.dropna(subset=[col])
    if d.empty:
        return
    plt.figure(figsize=(max(6, 1.1 * len(d)), 4.5))
    bars = plt.bar(d["Model"], d[col], color="#4C72B0")
    for b, v in zip(bars, d[col]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    lo = max(0, d[col].min() - 10)
    plt.ylim(lo, d[col].max() + 4)
    plt.ylabel(col + " (%)"); plt.title(title)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def main():
    # ---------- Table 1: baselines ----------
    base = ["unet", "unetplusplus", "pspnet", "linknet", "deeplabv3plus",
            "hrnet", "fcn8s", "proposed"]
    rows, results = [], {}
    for e in base:
        row, r = metrics_row(e)
        rows.append(row); results[e] = r
    t1 = pd.DataFrame(rows)
    md1 = save(t1, "table1_baselines",
               "Comparison with segmentation baselines on LoveDA val (identical training protocol).")
    bar(t1, "mIoU", "Baselines vs. proposed -- mIoU", os.path.join(OUT, "fig_baselines_miou.png"))

    # ---------- Table 2: class-wise IoU ----------
    rows = []
    for e in base:
        r = results[e]
        row = {"Model": DISPLAY[e]}
        for i, c in enumerate(CLASSES):
            row[c] = None if r is None else pct(r["best_class_iou"][i])
        row["mIoU"] = None if r is None else pct(r["best_mIoU"])
        rows.append(row)
    t2 = pd.DataFrame(rows)
    md2 = save(t2, "table2_classwise_iou", "Class-wise IoU (\\%) on LoveDA val.")
    d = t2.dropna(subset=["mIoU"])
    if not d.empty:
        mat = d[CLASSES].to_numpy(dtype=float)
        plt.figure(figsize=(9, 0.55 * len(d) + 1.5))
        plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=100)
        plt.colorbar(label="IoU (%)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                         color="white" if mat[i, j] < 50 else "black", fontsize=8)
        plt.xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
        plt.yticks(range(len(d)), d["Model"]); plt.title("Class-wise IoU")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_classwise_iou.png"), dpi=200); plt.close()

    # ---------- Table 3: cumulative ablation ----------
    cum = [("ablation_A", "A: DINOv2 only", "✓", "✗", "✗", "✗", "✗"),
           ("ablation_B", "B: A + FPN", "✓", "✓", "✗", "✗", "✗"),
           ("ablation_C", "C: B + ASPP", "✓", "✓", "✓", "✗", "✗"),
           ("ablation_D", "D: C + Residual Decoder", "✓", "✓", "✓", "✓", "✗"),
           ("proposed", "E: D + Composite Loss (Full)", "✓", "✓", "✓", "✓", "✓")]
    rows, prev = [], None
    for e, name, *flags in cum:
        row, r = metrics_row(e, name)
        full = {"Model": name, "DINOv2": flags[0], "FPN": flags[1], "ASPP": flags[2],
                "ResDec": flags[3], "Comp. loss": flags[4],
                "mIoU": row["mIoU"], "mDice": row["mDice"], "F1": row["F1"],
                "Δ mIoU": None if (row["mIoU"] is None or prev is None) else round(row["mIoU"] - prev, 2)}
        prev = row["mIoU"] if row["mIoU"] is not None else prev
        rows.append(full)
    t3 = pd.DataFrame(rows)
    md3 = save(t3, "table3_ablation_cumulative",
               "Cumulative ablation: one component added per row (mIoU, \\%).")
    bar(t3, "mIoU", "Cumulative ablation A -> E", os.path.join(OUT, "fig_ablation_cumulative.png"))

    # ---------- Table 4: component removal ----------
    rem = [("proposed", "Full model"),
           ("ablation_no_dinov2", "- DINOv2 (ResNet-50)"),
           ("ablation_no_fpn", "- FPN"),
           ("ablation_no_aspp", "- ASPP"),
           ("ablation_no_resdec", "- Residual Decoder"),
           ("ablation_D", "- Composite Loss (plain CE)")]
    full_miou = pct(results["proposed"]["best_mIoU"]) if results.get("proposed") else None
    rows = []
    for e, name in rem:
        row, r = metrics_row(e, name)
        rows.append({"Model": name, "Params (M)": row["Params (M)"], "mIoU": row["mIoU"],
                     "mDice": row["mDice"], "F1": row["F1"],
                     "Δ mIoU": None if (row["mIoU"] is None or full_miou is None or e == "proposed")
                     else round(row["mIoU"] - full_miou, 2)})
    t4 = pd.DataFrame(rows)
    md4 = save(t4, "table4_ablation_removal",
               "Component-removal ablation: each row removes one component from the full model.")
    bar(t4, "mIoU", "Component removal", os.path.join(OUT, "fig_ablation_removal.png"))

    # ---------- training curves ----------
    plt.figure(figsize=(8, 5)); n = 0
    for e in base[:-1] + [x for x, *_ in cum[:-1]] + ["ablation_no_dinov2", "ablation_no_fpn",
                                                       "ablation_no_aspp", "ablation_no_resdec"]:
        r = load(e)
        if r and r.get("history"):
            h = r["history"]
            plt.plot([x["epoch"] for x in h], [100 * x["mIoU"] for x in h], marker="o",
                     ms=3, label=DISPLAY.get(e, e)); n += 1
    if n:
        plt.xlabel("Epoch"); plt.ylabel("Val mIoU (%)"); plt.title("Validation mIoU per epoch")
        plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig_training_curves.png"), dpi=200)
    plt.close()

    # ---------- report ----------
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# LoveDA -- Baseline comparison & ablation study\n\n")
        f.write("Protocol (all rows): LoveDA Train (Rural+Urban, 2522 img) -> Val (1669 img), "
                "560x560, batch 2, AdamW lr 5e-5 (encoder x0.1), cosine, 10 epochs, grad-clip 1.0, "
                "seed 42, same augmentation; no-data pixels (label 0) ignored. "
                "Baselines use the same composite loss (CE+Dice+Boundary) as the proposed model. "
                "Best-epoch validation scores are reported.\n\n")
        f.write("## Table 1 -- Baselines (Comment 4)\n\n" + md1 + "\n\n![](fig_baselines_miou.png)\n\n")
        f.write("## Table 2 -- Class-wise IoU (Comment 4)\n\n" + md2 + "\n\n![](fig_classwise_iou.png)\n\n")
        f.write("## Table 3 -- Cumulative ablation (Comment 5)\n\n" + md3 + "\n\n![](fig_ablation_cumulative.png)\n\n")
        f.write("## Table 4 -- Component removal (Comment 5)\n\n" + md4 + "\n\n![](fig_ablation_removal.png)\n\n")
        f.write(f"> Note: single-seed runs; the proposed model's val mIoU varied by about ±{NOISE} "
                "points between epochs, so differences smaller than that should be described as "
                "comparable rather than as an improvement.\n\n")
        f.write("![](fig_training_curves.png)\n")
    print(f"Wrote tables, figures and REPORT.md to {OUT}")
    print(md1); print(); print(md3); print(); print(md4)


if __name__ == "__main__":
    main()
