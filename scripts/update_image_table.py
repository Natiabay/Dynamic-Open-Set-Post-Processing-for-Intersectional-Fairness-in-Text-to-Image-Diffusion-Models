#!/usr/bin/env python3
"""Update Stage B table in main.tex from results/paper_results_full.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "results" / "paper_results_full.json"
CSV = ROOT / "results" / "image_eval.csv"
TEX = ROOT / "craft-gc-paper" / "main.tex"

LABELS = {
    "base": "Base SD",
    "prompt_aug": "PromptAug",
    "fairimagen": "FairImagen",
    "craftgc": "CRAFT-GC",
}


def main():
    rows = []
    if JSON.exists():
        data = json.loads(JSON.read_text())
        for row in data.get("stage_b", {}).get("main_table", []):
            key = row.get("method_key", "")
            label = LABELS.get(key, row.get("method", key))
            rows.append(
                f"    {label} & {row['cofs']:.3f} & {row.get('cfs', 0):.3f} & "
                f"{row.get('clipscore', 0):.1f} & -- \\\\"
            )
    elif CSV.exists():
        import pandas as pd

        df = pd.read_csv(CSV)
        for mode, label in LABELS.items():
            sub = df[df["mode"] == mode]
            if sub.empty:
                continue
            cfs = 0.0
            if "cultural_sim" in sub.columns:
                base = df[df["mode"] == "base"].groupby("id")["cultural_sim"].mean()
                m = sub.groupby("id")["cultural_sim"].mean()
                common = base.index.intersection(m.index)
                if len(common):
                    cfs = float((m.loc[common] - base.loc[common]).mean())
            rows.append(
                f"    {label} & {sub.cofs.mean():.3f} & {cfs:.3f} & "
                f"{sub.clipscore.mean():.1f} & -- \\\\"
            )
    else:
        raise SystemExit("Run Colab Stage B and summarize_results.py first.")

    tex = TEX.read_text()
    pattern = r"(\\label\{tab:image\}[\s\S]*?\\midrule\n)(.*?)(\\bottomrule)"
    block = "\n".join(rows) + "\n    "
    new_tex, n = re.subn(pattern, rf"\1{block}\3", tex, count=1)
    if n == 0:
        raise SystemExit("Could not find tab:image in main.tex")
    TEX.write_text(new_tex)
    print(f"Updated {TEX}")


if __name__ == "__main__":
    main()
