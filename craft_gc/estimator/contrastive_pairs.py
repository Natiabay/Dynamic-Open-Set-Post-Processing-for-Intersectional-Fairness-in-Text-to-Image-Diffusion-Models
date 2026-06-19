"""Contrastive prompt pairs for bias direction estimation."""

from typing import List, Tuple

from craft_gc.detector.geo_cultural_taxonomy import (
    GEO_CULTURAL_TAXONOMY,
    PROFESSION_PROMPTS,
)

GENDER_POSITIVE = [
    "a photo of a man", "a photo of a male doctor", "a photo of a male engineer",
    "a photo of a male lawyer", "a photo of a male CEO", "a photo of a male scientist",
    "a photo of a male teacher", "a photo of a male nurse", "a photo of a male pilot",
    "a photo of a male farmer",
]
GENDER_NEUTRAL = [
    "a photo of a person", "a photo of a doctor", "a photo of an engineer",
    "a photo of a lawyer", "a photo of a CEO", "a photo of a scientist",
    "a photo of a teacher", "a photo of a nurse", "a photo of a pilot",
    "a photo of a farmer",
]

SKIN_POSITIVE = [
    "a photo of a light-skinned person", "a photo of a fair-skinned doctor",
    "a photo of a pale-skinned professional", "a photo of a white-skinned CEO",
    "a photo of a light-skinned engineer", "a photo of a fair-skinned teacher",
]
SKIN_NEUTRAL = [
    "a photo of a person", "a photo of a doctor", "a photo of a professional",
    "a photo of a CEO", "a photo of an engineer", "a photo of a teacher",
]


def build_geo_pairs(region: str) -> Tuple[List[str], List[str]]:
    """Return (cultural_prompts, western_default_prompts) for region."""
    data = GEO_CULTURAL_TAXONOMY[region]
    cultural = []
    for role in PROFESSION_PROMPTS[:12]:
        for kw in data["cultural_keywords"][:4]:
            cultural.append(f"a photo of {role} {kw}")
    for country in data["countries"][:3]:
        cultural.append(f"a photo of a professional from {country}")

    western = []
    for role in PROFESSION_PROMPTS[:12]:
        western.append(f"a photo of {role} in America")
        western.append(f"a photo of {role} in Europe")
    return cultural, western
