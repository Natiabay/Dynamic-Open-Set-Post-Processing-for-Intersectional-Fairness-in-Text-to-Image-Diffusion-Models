#!/usr/bin/env python3
"""Summarize Stage A/B results into paper_results.json and LaTeX-friendly tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = {
    "base": "Base SD",
    "prompt_aug": "PromptAug",
    "fairimagen": "FairImagen",
    "craftgc": "CRAFT-GC",
    "craftgc_e": "CRAFT-GC-E",
    "fairimagen_e": "FairImagen-E",
}


def summarize_embedding():
    csv = ROOT / "results" / "embedding_eval.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    rows = []
    for method in df.method.unique():
        sub = df[df.method == method]
        rows.append(
            {
                "method_key": method,
                "method": LABELS.get(method, method),
                "cofs": round(float(sub.cofs.mean()), 4),
                "cofs_std": round(float(sub.cofs.std()), 4),
                "cfs": round(float(sub.cfs.mean()), 4),
                "n": len(sub),
            }
        )
    return {"stage": "A", "main_table": rows}


def summarize_images():
    csv = ROOT / "results" / "image_eval.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    rows = []
    for mode in ["base", "prompt_aug", "fairimagen", "craftgc"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        cfs = None
        if "cultural_sim" in sub.columns:
            base_by_id = df[df["mode"] == "base"].groupby("id")["cultural_sim"].mean()
            m_by_id = sub.groupby("id")["cultural_sim"].mean()
            common = base_by_id.index.intersection(m_by_id.index)
            if len(common):
                cfs = round(float((m_by_id.loc[common] - base_by_id.loc[common]).mean()), 4)
        rows.append(
            {
                "method_key": mode,
                "method": LABELS.get(mode, mode),
                "cofs": round(float(sub.cofs.mean()), 4),
                "cofs_std": round(float(sub.cofs.std()), 4),
                "clipscore": round(float(sub.clipscore.mean()), 2),
                "cfs": cfs if cfs is not None else 0.0,
                "n": len(sub),
            }
        )

    by_region = {}
    for region in sorted(df.region.unique()):
        by_region[region] = {}
        for mode in ["base", "craftgc"]:
            sub = df[(df["region"] == region) & (df["mode"] == mode)]
            if len(sub):
                by_region[region][mode] = round(float(sub.cofs.mean()), 4)

    return {"stage": "B", "main_table": rows, "by_region": by_region, "n_prompts": df.id.nunique()}


def main():
    out = {"trial_note": "Trial 2 uses SD-native 768-d CAS steering and stratified sampling."}
    emb = summarize_embedding()
    img = summarize_images()
    if emb:
        out["stage_a"] = emb
    if img:
        out["stage_b"] = img

    path = ROOT / "results" / "paper_results_full.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
