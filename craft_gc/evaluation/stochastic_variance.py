"""Stochastic fairness variance across seeds."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def fairness_variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.var(ddof=1))


def aggregate_seed_metrics(seed_records: List[Dict]) -> Dict:
    cofs_vals = [r["cofs"] for r in seed_records]
    cfs_vals = [r.get("cfs", 0.0) for r in seed_records]
    return {
        "cofs_mean": float(np.mean(cofs_vals)),
        "cofs_std": float(np.std(cofs_vals, ddof=max(0, len(cofs_vals) - 1))),
        "cofs_variance": fairness_variance(cofs_vals),
        "cfs_mean": float(np.mean(cfs_vals)),
        "cfs_variance": fairness_variance(cfs_vals),
    }
