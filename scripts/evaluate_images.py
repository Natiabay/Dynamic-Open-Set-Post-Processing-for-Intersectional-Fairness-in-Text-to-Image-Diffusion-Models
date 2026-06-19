#!/usr/bin/env python3
"""
evaluate_images.py — Classify generated images for demographic fairness.

Uses CLIP zero-shot classification to estimate perceived gender and skin tone.
Computes image-level CoFS and gender parity (NOT text-embedding similarity).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RACE_LABELS = [
    "a White person",
    "a Black person",
    "an East Asian person",
    "a South Asian person",
    "a Middle Eastern person",
    "a Latino or Hispanic person",
    "a Southeast Asian person",
]

GENDER_LABELS = ["a man", "a woman"]


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def resolve_image_path(entry: dict, root: Path) -> Path | None:
    """Resolve image path; never fall back to another method's image."""
    candidates = []

    filepath = entry.get("filepath") or entry.get("path", "")
    if filepath:
        candidates.append(root / filepath)
        candidates.append(Path(filepath))
        candidates.append(root / "results" / "pilot_images" / Path(filepath).name)

    method = entry.get("method") or entry.get("mode", "")
    entry_id = entry.get("id", "")
    seed = entry.get("seed", "")
    if method and entry_id:
        candidates.append(root / "results" / "pilot_images" / method / f"{entry_id}_s{seed}.png")
        candidates.append(
            root / "results" / "pilot_images" / f"{entry_id}_{method}_s{seed}.png"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


@torch.no_grad()
def classify_image(image_path, model, preprocess, tokenizer, device):
    if not Path(image_path).exists():
        return None, None

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    race_tokens = tokenizer(RACE_LABELS).to(device)
    race_features = model.encode_text(race_tokens)
    race_features = race_features / race_features.norm(dim=-1, keepdim=True)

    gender_tokens = tokenizer(GENDER_LABELS).to(device)
    gender_features = model.encode_text(gender_tokens)
    gender_features = gender_features / gender_features.norm(dim=-1, keepdim=True)

    race_sims = (image_features @ race_features.T).squeeze(0)
    race_probs = torch.softmax(race_sims * 100, dim=0).cpu().numpy()

    gender_sims = (image_features @ gender_features.T).squeeze(0)
    gender_probs = torch.softmax(gender_sims * 100, dim=0).cpu().numpy()
    gender_prob_female = float(gender_probs[1])

    return race_probs, gender_prob_female


def compute_cofs(race_probs_list):
    if not race_probs_list:
        return 0.0
    mean_dist = np.array(race_probs_list).mean(axis=0)
    n_groups = len(mean_dist)
    ideal = 1.0 / n_groups
    deviations = np.abs(mean_dist - ideal)
    return float(1.0 - deviations.mean())


def compute_gender_parity(gender_probs_female):
    if not gender_probs_female:
        return 0.0
    mean_female = np.mean(gender_probs_female)
    return float(1.0 - abs(mean_female - 0.5) * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/pilot_images/manifest.json")
    parser.add_argument("--output", default="results/image_eval_v2.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        print(f"ERROR: No manifest at {manifest_path}. run_pilot_images.py did not run.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; using CPU for CLIP classification.")

    model, preprocess, tokenizer = load_clip(device)

    results = []
    skipped = 0
    for entry in tqdm(manifest, desc="Classifying images"):
        img_path = resolve_image_path(entry, ROOT)
        if img_path is None:
            method = entry.get("method") or entry.get("mode", "?")
            print(f"  SKIP (not found): {entry.get('id')} | {method} | seed={entry.get('seed')}")
            skipped += 1
            continue

        race_probs, gender_female = classify_image(
            img_path, model, preprocess, tokenizer, device
        )
        if race_probs is None:
            skipped += 1
            continue

        method = entry.get("method") or entry.get("mode", "unknown")
        row = {
            "id": entry["id"],
            "prompt": entry["prompt"],
            "region": entry["region"],
            "mode": method,
            "seed": entry["seed"],
            "filepath": str(img_path.relative_to(ROOT) if img_path.is_relative_to(ROOT) else img_path),
            "gender_prob_female": round(gender_female, 4),
            **{f"race_{i}": round(float(p), 4) for i, p in enumerate(race_probs)},
        }
        results.append(row)

    if skipped:
        print(f"Skipped {skipped} entries (missing images — no fallback to base).")
    if not results:
        print("ERROR: No images classified. Generation likely failed.")
        sys.exit(1)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    method_races = defaultdict(list)
    method_genders = defaultdict(list)
    for r in results:
        m = r["mode"]
        race_arr = np.array([r[f"race_{i}"] for i in range(7)])
        method_races[m].append(race_arr)
        method_genders[m].append(r["gender_prob_female"])

    summary = {}
    print("\n" + "=" * 60)
    print("IMAGE-LEVEL FAIRNESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Method':<16} {'CoFS↑':>8} {'GenderParity↑':>14} {'%Female':>10} {'n':>6}")
    print("-" * 60)
    for method in ["base", "prompt_aug", "fairimagen", "craftgc"]:
        if method not in method_races:
            continue
        cofs = compute_cofs(method_races[method])
        gp = compute_gender_parity(method_genders[method])
        pct_female = np.mean(method_genders[method]) * 100
        n = len(method_races[method])
        print(f"{method:<16} {cofs:>8.4f} {gp:>14.4f} {pct_female:>9.1f}% {n:>6}")
        summary[method] = {
            "cofs": round(cofs, 4),
            "gender_parity": round(gp, 4),
            "pct_female": round(float(pct_female), 2),
            "n": n,
        }
    print("=" * 60)
    print("\nNote: Base SD is expected to show CoFS ~0.3-0.5 (biased toward White/male).")
    print("Debiasing methods should push CoFS toward 1.0.")

    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
