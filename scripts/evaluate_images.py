#!/usr/bin/env python3
"""
Full image-level evaluation on generated PNGs (Colab / GPU output).

Expects manifest from scripts/run_pilot_images.py or run_full_experiment.py.
Computes CoFS (CLIPSE/gender via CLIP), CFS, CLIPScore.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark
from craft_gc.evaluation.cofs_metric import CoFSMetric
from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline


@torch.no_grad()
def clipscore(model, preprocess, tokenize, image: Image.Image, prompt: str, device: str) -> float:
    img = preprocess(image).unsqueeze(0).to(device)
    tokens = tokenize([prompt]).to(device)
    img_f = model.encode_image(img)
    txt_f = model.encode_text(tokens)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return float((img_f @ txt_f.T).item() * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/pilot_images/manifest.json")
    parser.add_argument("--output", default="results/image_eval.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    bench = {e["id"]: e for e in load_benchmark()}
    pipe = CRAFTGCPipeline(device=args.device, load_sd=False)
    cofs = CoFSMetric(pipe.clip_model, pipe.clip_preprocess, pipe.clip_tokenize, args.device)

    rows = []
    for item in manifest:
        path = Path(item["path"])
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        entry = bench.get(item["id"], {})
        prompt = item.get("prompt") or entry.get("prompt", "")
        metrics = cofs.compute([img])
        cult = entry.get("cultural_caption", prompt)
        cult_tokens = pipe.clip_tokenize([cult]).to(args.device)
        cult_emb = pipe.clip_model.encode_text(cult_tokens)
        cult_emb = cult_emb / cult_emb.norm(dim=-1, keepdim=True)
        img_f = cofs._image_features([img])
        cult_sim = float((img_f @ cult_emb.T).item())
        rows.append({
            **item,
            "cofs": metrics["cofs"],
            "cofs_race": metrics["cofs_race"],
            "cofs_gender": metrics["cofs_gender"],
            "clipscore": round(clipscore(pipe.clip_model, pipe.clip_preprocess, pipe.clip_tokenize, img, prompt, args.device), 2),
            "cultural_sim": round(cult_sim, 4),
        })

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if len(df):
        print(df.groupby("mode")[["cofs", "clipscore", "cultural_sim"]].mean().round(4))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
