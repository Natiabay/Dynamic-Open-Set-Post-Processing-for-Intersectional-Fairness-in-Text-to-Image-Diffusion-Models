#!/usr/bin/env python3
"""Plot embedding evaluation summary for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHOD_LABELS = {
    "base": "Base",
    "prompt_aug": "PromptAug",
    "fairimagen_e": "FairImagen-E",
    "craftgc_e": "CRAFT-GC-E",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/embedding_eval.csv")
    parser.add_argument("--out", default="craft-gc-paper/figures/cofs_comparison.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["label"] = df["method"].map(METHOD_LABELS)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.barplot(data=df, x="label", y="cofs", ax=axes[0], errorbar="sd", palette="muted")
    axes[0].set_title("CoFS (embedding proxy)")
    axes[0].set_ylabel("CoFS ↑")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=df, x="label", y="cfs", ax=axes[1], errorbar="sd", palette="muted")
    axes[1].set_title("CFS (cultural fidelity delta)")
    axes[1].set_ylabel("CFS ↑")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
