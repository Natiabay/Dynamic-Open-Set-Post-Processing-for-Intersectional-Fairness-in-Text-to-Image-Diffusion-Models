"""Bias Direction Estimator (BDE) in CLIP text space."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from craft_gc.estimator.contrastive_pairs import (
    GENDER_NEUTRAL,
    GENDER_POSITIVE,
    SKIN_NEUTRAL,
    SKIN_POSITIVE,
    build_geo_pairs,
)


class BiasDirectionEstimator:
    def __init__(self, clip_model, clip_tokenize, device: str = "cpu"):
        self.model = clip_model
        self.tokenize = clip_tokenize
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenize(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    def compute_direction(
        self,
        positive: List[str],
        negative: List[str],
        name: str,
    ) -> torch.Tensor:
        if name in self._cache:
            return self._cache[name]
        pos = self._encode(positive).mean(dim=0)
        neg = self._encode(negative).mean(dim=0)
        direction = pos - neg
        direction = direction / (direction.norm() + 1e-8)
        self._cache[name] = direction
        return direction

    def compute_all(self, region: Optional[str] = None) -> Dict[str, torch.Tensor]:
        directions = {
            "gender": self.compute_direction(GENDER_POSITIVE, GENDER_NEUTRAL, "gender"),
            "skin_tone": self.compute_direction(SKIN_POSITIVE, SKIN_NEUTRAL, "skin_tone"),
        }
        if region:
            geo_pos, geo_neg = build_geo_pairs(region)
            # Project OUT western-default bias: remove direction toward geo_neg
            directions[f"geo_{region}"] = self.compute_direction(geo_neg, geo_pos, f"geo_{region}")
        return directions
