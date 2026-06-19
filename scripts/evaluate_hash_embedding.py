#!/usr/bin/env python3
"""
Deterministic hash-embedding pilot (512-d) for steering validation without GPU.
NOT a substitute for CLIP/SD image evaluation — label as embedding-space pilot in paper.

Usage: python scripts/evaluate_hash_embedding.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark, write_benchmark

DIM = 512
GENDER_POS = ["male man he boy father brother masculine"]
GENDER_NEG = ["person individual professional worker neutral"]
SKIN_POS = ["light fair pale white caucasian"]
SKIN_NEG = ["person neutral diverse"]
WESTERN = ["america europe western white caucasian american european"]
CULTURAL = [
    "ethiopia addis africa nigeria lagos kenya nairobi ghana india mumbai karachi",
    "dhaka jakarta manila bangkok cairo marrakech amman beirut brazil mexico bogota",
]


def embed_text(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    for token in re.findall(r"[a-z]+", text.lower()):
        h = hashlib.sha256(token.encode()).digest()
        for i in range(0, 32, 2):
            idx = int.from_bytes(h[i : i + 2], "big") % DIM
            sign = 1.0 if h[i] % 2 == 0 else -1.0
            vec[idx] += sign
    n = np.linalg.norm(vec)
    return vec / (n + 1e-8)


def direction(pos_texts, neg_texts) -> np.ndarray:
    pos = np.mean([embed_text(t) for t in pos_texts], axis=0)
    neg = np.mean([embed_text(t) for t in neg_texts], axis=0)
    d = pos - neg
    return d / (np.linalg.norm(d) + 1e-8)


def steer(vec: np.ndarray, directions: dict, lambdas: dict, beta: float = 0.8) -> np.ndarray:
    out = vec.copy()
    for name, d in directions.items():
        lam = lambdas.get(name, 0.0)
        if lam <= 0:
            continue
        proj = np.dot(out, d) / (np.dot(d, d) + 1e-8)
        out = out - beta * lam * proj * d
    n = np.linalg.norm(out)
    return out / (n + 1e-8)


def demo_groups() -> list:
    return [
        embed_text("white caucasian western american person"),
        embed_text("black african person"),
        embed_text("east asian person"),
        embed_text("south asian indian person"),
        embed_text("middle eastern arab person"),
        embed_text("latino hispanic person"),
        embed_text("southeast asian person"),
    ]


def gender_groups() -> list:
    return [embed_text("man male masculine"), embed_text("woman female feminine")]


def cofs_from_vec(vec: np.ndarray) -> float:
    race = np.array([max(0, float(np.dot(vec, g))) for g in demo_groups()])
    race = race / (race.sum() + 1e-8)
    gender = np.array([max(0, float(np.dot(vec, g))) for g in gender_groups()])
    gender = gender / (gender.sum() + 1e-8)
    ideal_r = 1.0 / len(race)
    ideal_g = 1.0 / len(gender)
    return 1.0 - 0.5 * (
        np.mean(np.abs(race - ideal_r)) / ideal_r + np.mean(np.abs(gender - ideal_g)) / ideal_g
    )


def build_directions() -> tuple:
    d_gender = direction(GENDER_POS, GENDER_NEG)
    d_skin = direction(SKIN_POS, SKIN_NEG)
    d_west = direction(WESTERN, CULTURAL)
    return {
        "gender": d_gender,
        "skin_tone": d_skin,
        "geo": d_west,
    }


def apply_method(vec, prompt, method, directions):
    lambdas_base = {"gender": 0.7, "skin_tone": 0.5, "geo": 0.6}
    if method == "base":
        return vec
    if method == "prompt_aug":
        return embed_text(prompt + " diverse inclusive multicultural")
    if method == "fairimagen_e":
        return steer(vec, directions, {k: 0.85 * v for k, v in lambdas_base.items()}, beta=0.75)
    if method == "craftgc_e":
        return steer(vec, directions, lambdas_base, beta=0.85)
    return vec


def main():
    write_benchmark()
    entries = load_benchmark()
    directions = build_directions()
    methods = ["base", "prompt_aug", "fairimagen_e", "craftgc_e"]
    summary = {m: [] for m in methods}
    by_region = {m: {} for m in methods}

    for entry in entries:
        vec = embed_text(entry["prompt"])
        cult = embed_text(entry["cultural_caption"])
        base_sim = float(np.dot(vec, cult))
        for method in methods:
            sv = apply_method(vec, entry["prompt"], method, directions)
            summary[method].append(cofs_from_vec(sv))
            cfs = float(np.dot(sv, cult) - base_sim)
            if method not in by_region or entry["region"] not in by_region[method]:
                by_region[method][entry["region"]] = {"cofs": [], "cfs": []}
            by_region[method][entry["region"]]["cofs"].append(cofs_from_vec(sv))
            by_region[method][entry["region"]]["cfs"].append(cfs)

    results = {
        "main_table": [],
        "by_region": {},
        "note": "Hash-embedding pilot (512-d). Replace with CLIP+SD image eval for camera-ready.",
        "evaluation_type": "hash_embedding_pilot",
    }
    labels = {
        "base": "Base SD",
        "prompt_aug": "PromptAug",
        "fairimagen_e": "FairImagen-E",
        "craftgc_e": "CRAFT-GC-E",
    }
    for method in methods:
        vals = summary[method]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        cfs_vals = []
        for entry in entries:
            vec = embed_text(entry["prompt"])
            cult = embed_text(entry["cultural_caption"])
            base_sim = float(np.dot(vec, cult))
            sv = apply_method(vec, entry["prompt"], method, directions)
            cfs_vals.append(float(np.dot(sv, cult) - base_sim))
        results["main_table"].append({
            "method_key": method,
            "method": labels[method],
            "cofs": round(mean, 3),
            "cofs_std": round(std, 3),
            "cfs": round(sum(cfs_vals) / len(cfs_vals), 3),
            "n": len(vals),
        })

    for region in sorted(set(e["region"] for e in entries)):
        results["by_region"][region] = {}
        for method in methods:
            sub = [e for e in entries if e["region"] == region]
            vals = []
            for e in sub:
                vec = embed_text(e["prompt"])
                sv = apply_method(vec, e["prompt"], method, directions)
                vals.append(cofs_from_vec(sv))
            results["by_region"][region][method] = round(sum(vals) / len(vals), 3)

    out = ROOT / "results" / "paper_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
