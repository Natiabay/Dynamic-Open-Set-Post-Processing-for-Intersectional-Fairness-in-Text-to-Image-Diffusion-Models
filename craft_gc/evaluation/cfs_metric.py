"""Cultural Fidelity Score using CLIP similarity."""

from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image


class CFSMetric:
    def __init__(self, clip_model, clip_preprocess, clip_tokenize, device: str = "cpu"):
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.tokenize = clip_tokenize
        self.device = device

    @torch.no_grad()
    def _encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(tensors)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenize([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def compute(
        self,
        steered_images: List[Image.Image],
        baseline_images: List[Image.Image],
        cultural_caption: str,
    ) -> Dict:
        if not steered_images or not baseline_images:
            return {"cfs": 0.0}

        text_feat = self._encode_text(cultural_caption)
        steer_feat = self._encode_image(steered_images)
        base_feat = self._encode_image(baseline_images)

        steer_sim = (steer_feat @ text_feat.T).mean().item()
        base_sim = (base_feat @ text_feat.T).mean().item()

        return {"cfs": round(float(steer_sim - base_sim), 4), "steer_sim": steer_sim, "base_sim": base_sim}
