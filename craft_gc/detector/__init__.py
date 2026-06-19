from .geo_cultural_taxonomy import GEO_CULTURAL_TAXONOMY, CATEGORIES, REGION_KEYS

__all__ = ["CulturalAttributeDetector", "GEO_CULTURAL_TAXONOMY", "CATEGORIES", "REGION_KEYS"]


def __getattr__(name):
    if name == "CulturalAttributeDetector":
        from .cultural_attribute_detector import CulturalAttributeDetector
        return CulturalAttributeDetector
    raise AttributeError(name)
