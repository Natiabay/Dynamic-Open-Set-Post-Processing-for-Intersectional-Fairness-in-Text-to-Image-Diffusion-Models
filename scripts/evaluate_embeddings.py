#!/usr/bin/env python3
"""
Embedding-level evaluation on full GCFairBench (500 prompts).
Runs on CPU without Stable Diffusion — measures demographic neutrality of
steered CLIP prompt embeddings (text-space proxy; not image-level CoFS).

Usage:
  python scripts/evaluate_embeddings.py --output results/embedding_eval.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark, write_benchmark
from craft_gc.evaluation.cofs_metric import CoFSMetric
from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline
from craft_gc.steering.embedding_steering import steer_embedding

METHODS = ["base", "prompt_aug", "fairimagen_e", "craftgc_e"]

METRIC_NOTE = (
    "This measures CLIP text-embedding-space demographic parity of prompt encodings, "
    "NOT image-level demographic classification. See image_eval_v2 for image-level results."
)


@torch.no_grad()
def evaluate_prompt(pipeline, cofs, entry, method, beta) -> dict:
    prompt = entry["prompt"]
    if method == "prompt_aug":
        prompt = f"{prompt}, diverse inclusive multicultural representation"

    detected, directions, lambdas = pipeline.prepare(prompt)

    tokens = pipeline.clip_tokenize([prompt]).to(pipeline.device)
    emb = pipeline.clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    vec = emb.squeeze(0)

    if method == "base":
        steered = vec
    elif method == "fairimagen_e":
        steered = steer_embedding(vec, directions, {k: 0.85 * v for k, v in lambdas.items()}, beta=beta)
    elif method == "craftgc_e":
        steered = steer_embedding(vec, directions, lambdas, beta=beta)
    else:
        steered = vec

    race_sim = (steered.unsqueeze(0) @ cofs._race_text.T).softmax(dim=-1).squeeze().cpu().numpy()
    gender_sim = (steered.unsqueeze(0) @ cofs._gender_text.T).softmax(dim=-1).squeeze().cpu().numpy()
    align_race = cofs._cofs_from_probs(race_sim)
    align_gender = cofs._cofs_from_probs(gender_sim)
    text_alignment = 0.5 * (align_race + align_gender)

    cult_tokens = pipeline.clip_tokenize([entry["cultural_caption"]]).to(pipeline.device)
    cult = pipeline.clip_model.encode_text(cult_tokens)
    cult = cult / cult.norm(dim=-1, keepdim=True)
    base_sim = float((vec.unsqueeze(0) @ cult.T).item())
    steer_sim = float((steered.unsqueeze(0) @ cult.T).item())
    cfs = steer_sim - base_sim

    return {
        "id": entry["id"],
        "region": entry["region"],
        "category": entry["category"],
        "method": method,
        "text_alignment_score": round(text_alignment, 4),
        "text_alignment_race": round(align_race, 4),
        "text_alignment_gender": round(align_gender, 4),
        "cfs": round(cfs, 4),
        "detected_region": detected.get("region"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/embedding_eval.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 500")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    write_benchmark()
    entries = load_benchmark()
    if args.limit > 0:
        entries = entries[: args.limit]

    pipeline = CRAFTGCPipeline(device=args.device, load_sd=False)
    cofs = CoFSMetric(
        pipeline.clip_model, pipeline.clip_preprocess, pipeline.clip_tokenize, args.device
    )

    rows = []
    for method in METHODS:
        beta = 0.8 if method in ("fairimagen_e", "craftgc_e") else 1.0
        for entry in entries:
            rows.append(evaluate_prompt(pipeline, cofs, entry, method, beta))

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary_dict = {"metric_note": METRIC_NOTE, "methods": {}}
    for method in METHODS:
        sub = df[df["method"] == method]
        summary_dict["methods"][method] = {
            "text_alignment_score_mean": float(sub["text_alignment_score"].mean()),
            "text_alignment_score_std": float(sub["text_alignment_score"].std()),
            "cfs_mean": float(sub["cfs"].mean()),
            "cfs_std": float(sub["cfs"].std()),
        }

    summary_path = out.parent / "embedding_summary.json"
    summary_path.write_text(json.dumps(summary_dict, indent=2))

    print("Text-space alignment score (NOT image-level CoFS):")
    print(
        df.groupby("method")[["text_alignment_score", "cfs"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(f"Saved {out} and {summary_path}")


if __name__ == "__main__":
    main()
