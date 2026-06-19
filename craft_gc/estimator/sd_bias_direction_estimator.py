"""Bias directions in Stable Diffusion CLIP text-encoder space (768-d)."""

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


class SDBiasDirectionEstimator:
    """Contrastive directions using SD's CLIPTextModel (matches cross-attention input dim)."""

    def __init__(self, text_encoder, tokenizer, device: str = "cpu"):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        hidden = self.text_encoder(**tokens).last_hidden_state
        pooled = hidden.mean(dim=1)
        return pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)

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
            directions[f"geo_{region}"] = self.compute_direction(geo_neg, geo_pos, f"geo_{region}")
        return directions
