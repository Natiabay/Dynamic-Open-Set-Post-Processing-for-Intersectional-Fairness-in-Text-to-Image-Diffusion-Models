"""Cultural Attribute Detector (CAD) using CLIP text similarity."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from craft_gc.detector.geo_cultural_taxonomy import GEO_CULTURAL_TAXONOMY


class CulturalAttributeDetector:
    """Maps prompts to geo-cultural regions via CLIP zero-shot similarity."""

    def __init__(self, clip_model, clip_tokenize, device: str = "cpu", threshold: float = 0.22):
        self.model = clip_model
        self.tokenize = clip_tokenize
        self.device = device
        self.threshold = threshold
        self.region_text_features = self._encode_region_templates()

    def _build_templates(self) -> Dict[str, List[str]]:
        templates = {}
        for region, data in GEO_CULTURAL_TAXONOMY.items():
            templates[region] = (
                [f"a person from {c}" for c in data["countries"]]
                + [f"a scene {kw}" for kw in data["cultural_keywords"]]
            )
        return templates

    @torch.no_grad()
    def _encode_region_templates(self) -> Dict[str, torch.Tensor]:
        encoded = {}
        templates = self._build_templates()
        for region, texts in templates.items():
            tokens = self.tokenize(texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            encoded[region] = feats.mean(dim=0)
            encoded[region] = encoded[region] / encoded[region].norm()
        return encoded

    @torch.no_grad()
    def detect(self, prompt: str) -> Dict:
        tokens = self.tokenize([prompt]).to(self.device)
        prompt_feat = self.model.encode_text(tokens)
        prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
        prompt_feat = prompt_feat.squeeze(0)

        region_scores = {}
        for region, region_feat in self.region_text_features.items():
            region_scores[region] = float((prompt_feat @ region_feat).item())

        best_region = max(region_scores, key=region_scores.get)
        best_score = region_scores[best_region]
        detected = best_region if best_score > self.threshold else None

        detected_keywords = []
        if detected:
            pl = prompt.lower()
            for kw in GEO_CULTURAL_TAXONOMY[detected]["cultural_keywords"]:
                if any(w in pl for w in kw.lower().split() if len(w) > 3):
                    detected_keywords.append(kw)

        cultural_caption = self._cultural_caption(prompt, detected)
        return {
            "region": detected,
            "region_confidence": best_score,
            "has_cultural_context": detected is not None,
            "detected_keywords": detected_keywords,
            "all_region_scores": region_scores,
            "cultural_caption": cultural_caption,
        }

    def _cultural_caption(self, prompt: str, region: Optional[str]) -> str:
        if not region:
            return prompt
        data = GEO_CULTURAL_TAXONOMY[region]
        loc = data["cultural_keywords"][0]
        return f"{prompt}, {loc}, authentic local context"
