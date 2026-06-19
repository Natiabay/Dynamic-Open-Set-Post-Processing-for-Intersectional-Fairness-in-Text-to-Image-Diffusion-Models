#!/usr/bin/env python3
"""Merge CLIP embedding + image eval JSON into paper_results.json and LaTeX snippets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "base": "Base",
    "prompt_aug": "PromptAug",
    "fairimagen_e": "FairImagen-E",
    "craftgc_e": "CRAFT-GC-E",
    "craftgc": "CRAFT-GC",
}


def from_embedding_csv(csv_path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    main_table = []
    for method in df.method.unique():
        sub = df[df.method == method]
        main_table.append({
            "method_key": method,
            "method": LABELS.get(method, method),
            "cofs": round(float(sub.cofs.mean()), 4),
            "cofs_std": round(float(sub.cofs.std()), 4),
            "cfs": round(float(sub.cfs.mean()), 4),
            "cfs_std": round(float(sub.cfs.std()), 4),
            "n": len(sub),
        })
    by_region = {}
    for region in sorted(df.region.unique()):
        by_region[region] = {}
        for method in df.method.unique():
            sub = df[(df.region == region) & (df.method == method)]
            by_region[region][method] = round(float(sub.cofs.mean()), 4)
    return {
        "evaluation_type": "clip_embedding",
        "main_table": main_table,
        "by_region": by_region,
        "note": "Real CLIP ViT-B/32 text-embedding evaluation on GCFairBench n=500.",
    }


def from_embedding_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    main_table = []
    for key, vals in data.items():
        main_table.append({
            "method_key": key,
            "method": LABELS.get(key, key),
            "cofs": round(vals["cofs_mean"], 3),
            "cofs_std": round(vals["cofs_std"], 3),
            "cfs": round(vals["cfs_mean"], 3),
            "n": 500,
        })
    return {
        "evaluation_type": "clip_embedding",
        "main_table": main_table,
        "note": "Real CLIP ViT-B/32 text-embedding evaluation on GCFairBench n=500.",
    }


def main():
    csv = ROOT / "results" / "embedding_eval.csv"
    emb = ROOT / "results" / "embedding_summary.json"
    out = ROOT / "results" / "paper_results.json"
    if csv.exists():
        results = from_embedding_csv(csv)
    elif emb.exists():
        results = from_embedding_summary(emb)
    else:
        raise SystemExit("Run scripts/evaluate_embeddings.py first.")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
