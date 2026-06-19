#!/usr/bin/env python3
"""
Embedding-level evaluation on full GCFairBench (500 prompts).
Runs on CPU without Stable Diffusion — measures demographic neutrality of
steered CLIP prompt embeddings (validated proxy; image eval in run_pilot_images.py).

Usage:
  python scripts/evaluate_embeddings.py --output results/embedding_eval.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark, write_benchmark
from craft_gc.evaluation.cofs_metric import CoFSMetric
from craft_gc.evaluation.stochastic_variance import fairness_variance
from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline, load_clip
from craft_gc.steering.embedding_steering import default_lambdas, steer_embedding


METHODS = ["base", "prompt_aug", "fairimagen_e", "craftgc_e"]


@torch.no_grad()
def embed_prompt(model, tokenize, prompt: str, device: str) -> torch.Tensor:
    tokens = tokenize([prompt]).to(device)
    feat = model.encode_text(tokens)
    return feat / feat.norm(dim=-1, keepdim=True)


def apply_method(
    pipeline: CRAFTGCPipeline,
    prompt: str,
    method: str,
    beta: float,
) -> str:
    if method == "prompt_aug":
        return f"{prompt}, diverse inclusive multicultural representation"
    return prompt


@torch.no_grad()
def evaluate_prompt(
    pipeline: CRAFTGCPipeline,
    cofs: CoFSMetric,
    entry: dict,
    method: str,
    beta: float,
) -> dict:
    prompt = apply_method(pipeline, entry["prompt"], method, beta)
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

    # Score steered embedding via race/gender text probes
    race_sim = (steered.unsqueeze(0) @ cofs._race_text.T).softmax(dim=-1).squeeze().cpu().numpy()
    gender_sim = (steered.unsqueeze(0) @ cofs._gender_text.T).softmax(dim=-1).squeeze().cpu().numpy()
    cofs_race = cofs._cofs_from_probs(race_sim)
    cofs_gender = cofs._cofs_from_probs(gender_sim)
    cofs_val = 0.5 * (cofs_race + cofs_gender)

    # CFS proxy: similarity to cultural caption
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
        "cofs": round(cofs_val, 4),
        "cofs_race": round(cofs_race, 4),
        "cofs_gender": round(cofs_gender, 4),
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

    summary = df.groupby("method")[["cofs", "cfs"]].agg(["mean", "std"]).round(4)
    summary_path = out.parent / "embedding_summary.json"
    summary_dict = {}
    for method in METHODS:
        sub = df[df.method == method]
        summary_dict[method] = {
            "cofs_mean": float(sub.cofs.mean()),
            "cofs_std": float(sub.cofs.std()),
            "cfs_mean": float(sub.cfs.mean()),
            "cfs_std": float(sub.cfs.std()),
        }
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(summary)
    print(f"Saved {out} and {summary_path}")


if __name__ == "__main__":
    main()
