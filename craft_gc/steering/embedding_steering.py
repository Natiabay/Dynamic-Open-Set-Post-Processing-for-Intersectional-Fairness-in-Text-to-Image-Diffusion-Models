"""Embedding-level steering (CRAFT-GC-E): projects CLIP prompt embeddings."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def steer_embedding(
    embedding: torch.Tensor,
    directions: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    beta: float = 1.0,
) -> torch.Tensor:
    """Remove bias components from a single prompt embedding [seq, dim] or [dim]."""
    out = embedding.clone()
    if out.dim() == 1:
        out = out.unsqueeze(0)

    for name, direction in directions.items():
        lam = lambdas.get(name, lambdas.get(name.replace("geo_", "geo_"), 0.0))
        if lam <= 0:
            continue
        d = direction.to(out.device).view(1, -1)
        d_norm_sq = (d * d).sum(dim=-1, keepdim=True) + 1e-8
        proj = (out * d).sum(dim=-1, keepdim=True) / d_norm_sq
        out = out - beta * lam * proj * d

    out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
    return out.squeeze(0) if embedding.dim() == 1 else out


def default_lambdas(region: Optional[str]) -> Dict[str, float]:
    weights = {"gender": 0.7, "skin_tone": 0.5}
    if region:
        weights[f"geo_{region}"] = 0.6
    return weights
