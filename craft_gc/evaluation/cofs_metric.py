"""Counterfactual Fairness Score via CLIP zero-shot demographic probes."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

DEMO_GROUPS = [
    "a photo of a White person",
    "a photo of a Black person",
    "a photo of an East Asian person",
    "a photo of a South Asian person",
    "a photo of a Middle Eastern person",
    "a photo of a Latino person",
    "a photo of a Southeast Asian person",
]

GENDER_GROUPS = [
    "a photo of a man",
    "a photo of a woman",
]


class CoFSMetric:
    """CoFS from CLIP similarity to demographic text prototypes."""

    def __init__(self, clip_model, clip_preprocess, clip_tokenize, device: str = "cpu"):
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.tokenize = clip_tokenize
        self.device = device
        self._race_text = self._encode_text(DEMO_GROUPS)
        self._gender_text = self._encode_text(GENDER_GROUPS)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _image_features(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(tensors)
        return feats / feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def _cofs_from_probs(probs: np.ndarray) -> float:
        ideal = 1.0 / len(probs)
        return float(1.0 - np.mean(np.abs(probs - ideal)))

    @torch.no_grad()
    def compute(self, images: List[Image.Image]) -> Dict:
        if not images:
            return {"cofs_race": 0.0, "cofs_gender": 0.0, "cofs": 0.0}

        img_feat = self._image_features(images)
        race_sim = (img_feat @ self._race_text.T).softmax(dim=-1).cpu().numpy()
        gender_sim = (img_feat @ self._gender_text.T).softmax(dim=-1).cpu().numpy()

        race_mean = race_sim.mean(axis=0)
        gender_mean = gender_sim.mean(axis=0)
        cofs_race = self._cofs_from_probs(race_mean)
        cofs_gender = self._cofs_from_probs(gender_mean)

        return {
            "cofs_race": round(cofs_race, 4),
            "cofs_gender": round(cofs_gender, 4),
            "cofs": round(0.5 * (cofs_race + cofs_gender), 4),
            "race_distribution": dict(zip([f"g{i}" for i in range(len(race_mean))], race_mean.tolist())),
            "gender_distribution": dict(zip(["male", "female"], gender_mean.tolist())),
        }

    @torch.no_grad()
    def compute_embedding_bias(self, prompt: str, steered: bool = False) -> Dict:
        """Text-side proxy: measure demographic pull of prompt embedding."""
        tokens = self.tokenize([prompt]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        race_probs = (feat @ self._race_text.T).softmax(dim=-1).squeeze().cpu().numpy()
        gender_probs = (feat @ self._gender_text.T).softmax(dim=-1).squeeze().cpu().numpy()
        return {
            "cofs_race": self._cofs_from_probs(race_probs),
            "cofs_gender": self._cofs_from_probs(gender_probs),
            "cofs": self._cofs_from_probs(0.5 * (race_probs + gender_probs[:2])),
        }
