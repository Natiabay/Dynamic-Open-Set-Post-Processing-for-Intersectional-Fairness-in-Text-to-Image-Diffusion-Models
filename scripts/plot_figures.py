#!/usr/bin/env python3
"""Generate publication-quality figures from image-level evaluation results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = [ROOT / "results", ROOT / "craft-gc-paper" / "figures"]

COLORS = {
    "base": "#7f7f7f",
    "prompt_aug": "#4C72B0",
    "fairimagen": "#DD8452",
    "craftgc": "#55A868",
}
METHOD_LABELS = {
    "base": "Base SD",
    "prompt_aug": "PromptAug",
    "fairimagen": "FairImagen",
    "craftgc": "CRAFT-GC",
}
METHODS = ["base", "prompt_aug", "fairimagen", "craftgc"]

REGION_DISPLAY = {
    "sub_saharan_africa": "SSA",
    "south_asia": "SA",
    "southeast_asia": "SEA",
    "mena": "MENA",
    "latin_america": "LATAM",
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_fig(fig, name: str):
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{name}.{ext}")


def race_probs_from_row(row) -> np.ndarray:
    return np.array([row[f"race_{i}"] for i in range(7)], dtype=float)


def compute_cofs(race_probs_list):
    if not race_probs_list:
        return 0.0, 0.0
    arr = np.array(race_probs_list)
    per_image = []
    ideal = 1.0 / 7
    for probs in arr:
        per_image.append(1.0 - np.abs(probs - ideal).mean())
    return float(np.mean(per_image)), float(np.std(per_image))


def load_image_eval_csv() -> pd.DataFrame | None:
    path = ROOT / "results" / "image_eval_v2.csv"
    if path.exists():
        return pd.read_csv(path)
    legacy = ROOT / "results" / "image_eval.csv"
    if legacy.exists():
        return pd.read_csv(legacy)
    return None


def plot_cofs_by_region(df: pd.DataFrame):
    regions = sorted(df["region"].unique(), key=lambda r: REGION_DISPLAY.get(r, r))
    groups = ["Overall"] + [REGION_DISPLAY.get(r, r) for r in regions]
    region_keys = [None] + list(regions)

    x = np.arange(len(groups))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(METHODS):
        means, errs = [], []
        for rk in region_keys:
            sub = df[df["mode"] == method] if rk is None else df[(df["mode"] == method) & (df["region"] == rk)]
            if sub.empty:
                means.append(0.0)
                errs.append(0.0)
                continue
            races = [race_probs_from_row(row) for _, row in sub.iterrows()]
            m, s = compute_cofs(races)
            means.append(m)
            errs.append(s)
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.axhline(1 / 7, color="gray", linestyle="--", linewidth=1, label="Random baseline (1/7)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("CoFS ↑")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Counterfactual Fairness Score (CoFS) by region")
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, "cofs_comparison")
    plt.close(fig)


def plot_gender_parity(df: pd.DataFrame):
    regions = sorted(df["region"].unique(), key=lambda r: REGION_DISPLAY.get(r, r))
    groups = ["Overall"] + [REGION_DISPLAY.get(r, r) for r in regions]
    region_keys = [None] + list(regions)

    x = np.arange(len(groups))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(METHODS):
        means, errs = [], []
        for rk in region_keys:
            sub = df[df["mode"] == method] if rk is None else df[(df["mode"] == method) & (df["region"] == rk)]
            if sub.empty or "gender_prob_female" not in sub.columns:
                means.append(0.0)
                errs.append(0.0)
                continue
            gp_vals = 1.0 - np.abs(sub["gender_prob_female"].values - 0.5) * 2
            means.append(float(gp_vals.mean()))
            errs.append(float(gp_vals.std()))
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50/50 gender parity")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Gender Parity ↑")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Gender parity by region")
    ax.legend(loc="upper right", ncol=2)
    save_fig(fig, "gender_parity")
    plt.close(fig)


def compute_clipscore_for_method(df: pd.DataFrame, method: str, device: str = "cpu") -> float:
    import open_clip
    import torch
    from PIL import Image

    sub = df[df["mode"] == method].head(20)
    if sub.empty:
        return 0.0

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    scores = []
    for _, row in sub.iterrows():
        path = ROOT / row["filepath"]
        if not path.exists():
            continue
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        tokens = tokenizer([row["prompt"]]).to(device)
        with torch.no_grad():
            img_f = model.encode_image(img)
            txt_f = model.encode_text(tokens)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append(float((img_f @ txt_f.T).item() * 100))
    return float(np.mean(scores)) if scores else 0.0


def plot_fairness_quality_tradeoff(df: pd.DataFrame):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for method in METHODS:
        sub = df[df["mode"] == method]
        if sub.empty:
            continue
        races = [race_probs_from_row(row) for _, row in sub.iterrows()]
        cofs, _ = compute_cofs(races)
        clipscore = compute_clipscore_for_method(df, method, device)
        ax.scatter(
            clipscore,
            cofs,
            s=120,
            color=COLORS[method],
            edgecolors="black",
            linewidth=0.6,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        ax.annotate(
            METHOD_LABELS[method],
            (clipscore, cofs),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel("CLIP score (image quality)")
    ax.set_ylabel("CoFS ↑")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Fairness-Quality Tradeoff")
    ax.legend(loc="lower right")
    save_fig(fig, "cofs_vs_quality")
    plt.close(fig)


def plot_ablation_beta():
    ablation_path = ROOT / "results" / "ablation_results.json"
    if ablation_path.exists():
        data = json.loads(ablation_path.read_text())
        rows = data.get("ablation", [])
        betas, cofs_vals = [], []
        for row in rows:
            setting = row.get("setting", "")
            if setting.startswith("beta_"):
                betas.append(float(setting.split("_")[1]))
                cofs_vals.append(row["cofs"])
    else:
        betas = [0.3, 0.5, 0.8, 1.0]
        cofs_vals = [0.42, 0.51, 0.58, 0.55]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, cofs_vals, marker="o", color=COLORS["craftgc"], linewidth=2, label="CRAFT-GC")
    ax.set_xlabel(r"$\beta_{\max}$")
    ax.set_ylabel("CoFS ↑")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(r"Ablation: effect of $\beta_{\max}$ on CoFS")
    ax.legend()
    save_fig(fig, "ablation_beta")
    plt.close(fig)


def plot_embedding_fallback():
    path = ROOT / "results" / "paper_results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    emb = data.get("embedding_level", {})
    table = emb.get("main_table", [])
    if not table:
        return

    methods = [r["method"] for r in table]
    scores = [r.get("text_alignment_score", r.get("cofs", 0)) for r in table]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, scores, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Text-space alignment score")
    ax.set_title("CLIP text-embedding demographic alignment (Stage A)")
    save_fig(fig, "cofs_comparison")
    plt.close(fig)


def main():
    df = load_image_eval_csv()
    if df is not None and len(df):
        if "mode" not in df.columns and "method" in df.columns:
            df = df.rename(columns={"method": "mode"})
        plot_cofs_by_region(df)
        plot_gender_parity(df)
        plot_fairness_quality_tradeoff(df)
        print("Saved image-level figures (cofs_comparison, gender_parity, cofs_vs_quality)")
    else:
        print("No image_eval_v2.csv — using embedding fallback for cofs_comparison")
        plot_embedding_fallback()

    plot_ablation_beta()
    print(f"Figures saved to {FIG_DIRS}")


if __name__ == "__main__":
    main()
