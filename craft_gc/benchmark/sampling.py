"""Stratified prompt sampling for reproducible multi-region evaluation."""

from __future__ import annotations

from typing import List

from craft_gc.benchmark.gcfairbench import load_benchmark
from craft_gc.detector.geo_cultural_taxonomy import REGION_KEYS


def load_stratified(per_region: int = 20) -> List[dict]:
    """Return per_region prompts from each of the five GCFairBench regions."""
    entries = load_benchmark()
    by_region: dict[str, list] = {r: [] for r in REGION_KEYS}
    for entry in entries:
        region = entry["region"]
        if region in by_region and len(by_region[region]) < per_region:
            by_region[region].append(entry)

    out: List[dict] = []
    for region in REGION_KEYS:
        out.extend(by_region[region])
    return out
