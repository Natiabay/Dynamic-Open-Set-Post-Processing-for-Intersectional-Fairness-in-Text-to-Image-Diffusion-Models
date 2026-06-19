#!/usr/bin/env python3
"""Update main.tex Stage B table from results/image_eval.csv."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "image_eval.csv"
TEX = ROOT / "craft-gc-paper" / "main.tex"

LABELS = {
    "base": "Base SD",
    "prompt_aug": "PromptAug",
    "fairimagen": "FairImagen",
    "craftgc": "CRAFT-GC",
}


def main():
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}; run Colab Stage B first.")
    df = pd.read_csv(CSV)
    rows = []
    for mode, label in LABELS.items():
        sub = df[df.mode == mode]
        if sub.empty:
            continue
        rows.append(
            f"    {label} & {sub.cofs.mean():.3f} & {sub.cultural_sim.mean():.3f} & "
            f"{sub.clipscore.mean():.1f} & -- \\\\"
        )
    block = "\n".join(rows)
    tex = TEX.read_text()
    pattern = r"(\\midrule\n)(.*?)(\\bottomrule)"
    replacement = rf"\1{block}\n    \3"
    new_tex, n = re.subn(pattern, replacement, tex, count=1, flags=re.S)
    if n == 0:
        raise SystemExit("Could not find tab:image table in main.tex")
    TEX.write_text(new_tex)
    print(f"Updated {TEX} with image results from {CSV}")


if __name__ == "__main__":
    main()
