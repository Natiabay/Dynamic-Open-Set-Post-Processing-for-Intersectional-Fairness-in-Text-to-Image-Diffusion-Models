#!/usr/bin/env python3
"""Generate LaTeX table rows and paper results JSON from evaluation CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METHOD_LABELS = {
    "base": "Base (embedding)",
    "prompt_aug": "PromptAug",
    "fairimagen_e": "FairImagen-E",
    "craftgc_e": "CRAFT-GC-E",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/embedding_eval.csv")
    parser.add_argument("--out", default="results/paper_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    results = {"main_table": [], "by_region": {}}

    for method, label in METHOD_LABELS.items():
        sub = df[df.method == method]
        if sub.empty:
            continue
        results["main_table"].append({
            "method": label,
            "cofs": round(sub.cofs.mean(), 3),
            "cofs_std": round(sub.cofs.std(), 3),
            "cfs": round(sub.cfs.mean(), 3),
            "cfs_std": round(sub.cfs.std(), 3),
            "n": len(sub),
        })

    for region in df.region.unique():
        results["by_region"][region] = {}
        for method in METHOD_LABELS:
            sub = df[(df.region == region) & (df.method == method)]
            if len(sub):
                results["by_region"][region][method] = round(sub.cofs.mean(), 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
