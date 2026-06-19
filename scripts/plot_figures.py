#!/usr/bin/env python3
"""Generate CoFS comparison figure from paper_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results" / "paper_results.json").read_text())
fig_dir = ROOT / "craft-gc-paper" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

methods = [r["method"] for r in data["main_table"]]
cofs = [r["cofs"] for r in data["main_table"]]
stds = [r["cofs_std"] for r in data["main_table"]]
cfs = [r["cfs"] for r in data["main_table"]]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
x = np.arange(len(methods))
axes[0].bar(x, cofs, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
axes[0].set_ylabel("CoFS (CLIP embedding) ↑")
axes[0].set_title("(a) Demographic parity (text space)")
ymin, ymax = min(cofs) - 0.002, max(cofs) + 0.002
axes[0].set_ylim(max(0.99, ymin), min(1.0, ymax + 0.001))

axes[1].bar(x, cfs, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
axes[1].set_ylabel("CFS delta")
axes[1].set_title("(b) Cultural fidelity delta")
axes[1].axhline(0, color="gray", linewidth=0.8)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(fig_dir / f"cofs_comparison.{ext}", dpi=150, bbox_inches="tight")
print(f"Saved figures to {fig_dir}")
