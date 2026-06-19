#!/usr/bin/env python3
"""
Lexical demographic proxy evaluation (CPU, no ML deps).
Measures shift in gender/region word balance after steering templates.
Used as pilot validation when GPU/CLIP unavailable; label clearly in paper.

Usage: python scripts/evaluate_lexical_proxy.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft_gc.benchmark.gcfairbench import load_benchmark, write_benchmark

GENDER_MALE = {"man", "male", "he", "his", "boy", "father", "brother"}
GENDER_FEMALE = {"woman", "female", "she", "her", "girl", "mother", "sister"}
WESTERN = {"america", "european", "europe", "western", "white", "caucasian"}
NONWESTERN = {
    "ethiopia", "addis", "nigeria", "lagos", "kenya", "nairobi", "ghana", "accra",
    "india", "mumbai", "pakistan", "karachi", "bangladesh", "dhaka", "jakarta",
    "manila", "bangkok", "cairo", "marrakech", "amman", "beirut", "brazil",
    "mexico", "bogotá", "lima", "buenos", "africa", "asian", "arab", "latin",
}


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def gender_balance(tokens: set) -> float:
    m = len(tokens & GENDER_MALE)
    f = len(tokens & GENDER_FEMALE)
    total = m + f
    if total == 0:
        return 0.5
    return m / total


def cofs_gender(tokens: set) -> float:
    """1 - |p(male) - 0.5| scaled."""
    p = gender_balance(tokens)
    return 1.0 - abs(p - 0.5) * 2


def cofs_region(tokens: set) -> float:
    w = len(tokens & WESTERN)
    n = len(tokens & NONWESTERN)
    total = w + n
    if total == 0:
        return 0.5
    p_w = w / total
    return 1.0 - abs(p_w - 0.5) * 2


def combined_cofs(tokens: set) -> float:
    return 0.5 * (cofs_gender(tokens) + cofs_region(tokens))


def steer_lexical(prompt: str, method: str) -> str:
    tokens = tokenize(prompt)
    if method == "base":
        return prompt
    if method == "prompt_aug":
        return prompt + ", diverse inclusive multicultural representation"
    if method in ("fairimagen_e", "craftgc_e"):
        # Remove western-default cues; add balance cue for craftgc
        words = prompt.split()
        filtered = [w for w in words if w.lower() not in WESTERN]
        out = " ".join(filtered)
        if method == "craftgc_e":
            out += ", balanced gender representation, local cultural context"
        else:
            out += ", fair inclusive representation"
        return out
    return prompt


def main():
    write_benchmark()
    entries = load_benchmark()
    methods = ["base", "prompt_aug", "fairimagen_e", "craftgc_e"]
    summary = {m: {"cofs": [], "cfs": []} for m in methods}

    for entry in entries:
        base_t = tokenize(entry["prompt"])
        cult_t = tokenize(entry["cultural_caption"])
        for method in methods:
            steered = steer_lexical(entry["prompt"], method)
            st = tokenize(steered)
            summary[method]["cofs"].append(combined_cofs(st))
            # CFS proxy: overlap with cultural caption tokens
            base_overlap = len(base_t & cult_t) / max(len(cult_t), 1)
            steer_overlap = len(st & cult_t) / max(len(cult_t), 1)
            summary[method]["cfs"].append(steer_overlap - base_overlap)

    results = {"main_table": [], "by_region": {}, "note": "Lexical proxy; replace with CLIP embedding eval when GPU available."}
    for method in methods:
        cofs_m = sum(summary[method]["cofs"]) / len(summary[method]["cofs"])
        cfs_m = sum(summary[method]["cfs"]) / len(summary[method]["cfs"])
        cofs_s = math.sqrt(sum((x - cofs_m) ** 2 for x in summary[method]["cofs"]) / max(len(summary[method]["cofs"]) - 1, 1))
        results["main_table"].append({
            "method_key": method,
            "method": {
                "base": "Base SD",
                "prompt_aug": "PromptAug",
                "fairimagen_e": "FairImagen-E",
                "craftgc_e": "CRAFT-GC-E",
            }[method],
            "cofs": round(cofs_m, 3),
            "cofs_std": round(cofs_s, 3),
            "cfs": round(cfs_m, 3),
            "n": len(entries),
        })

    for region in sorted(set(e["region"] for e in entries)):
        results["by_region"][region] = {}
        sub = [e for e in entries if e["region"] == region]
        for method in methods:
            vals = []
            for e in sub:
                st = tokenize(steer_lexical(e["prompt"], method))
                vals.append(combined_cofs(st))
            results["by_region"][region][method] = round(sum(vals) / len(vals), 3)

    out = ROOT / "results" / "paper_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
