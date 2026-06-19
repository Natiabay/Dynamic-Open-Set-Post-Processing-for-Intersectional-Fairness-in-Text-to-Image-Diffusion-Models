#!/usr/bin/env python3
"""Ablation study on hash-embedding pilot (same proxy as main eval)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark
from scripts.evaluate_hash_embedding import (
    apply_method,
    build_directions,
    cofs_from_vec,
    embed_text,
    steer,
)


def cofs_for_steering(vec, directions, lambdas, beta):
    sv = steer(vec, directions, lambdas, beta=beta)
    return cofs_from_vec(sv)


def main():
    entries = load_benchmark()
    directions = build_directions()
    lambdas_full = {"gender": 0.7, "skin_tone": 0.5, "geo": 0.6}
    lambdas_no_geo = {"gender": 0.7, "skin_tone": 0.5, "geo": 0.0}

    settings = [
        ("beta_0.3", lambdas_full, 0.3),
        ("beta_0.8", lambdas_full, 0.8),
        ("beta_1.0", lambdas_full, 1.0),
        ("no_geo", lambdas_no_geo, 0.85),
    ]

    rows = []
    for name, lambdas, beta in settings:
        vals = []
        for entry in entries:
            vec = embed_text(entry["prompt"])
            vals.append(cofs_for_steering(vec, directions, lambdas, beta))
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        rows.append({"setting": name, "cofs": round(mean, 3), "cofs_std": round(std, 3)})

    out = ROOT / "results" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ablation": rows}, indent=2))
    print(json.dumps({"ablation": rows}, indent=2))


if __name__ == "__main__":
    main()
