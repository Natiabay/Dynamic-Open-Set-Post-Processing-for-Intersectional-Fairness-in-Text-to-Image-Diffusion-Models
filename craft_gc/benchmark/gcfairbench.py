"""GCFairBench prompt generator and loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from craft_gc.detector.geo_cultural_taxonomy import (
    CATEGORIES,
    GEO_CULTURAL_TAXONOMY,
    REGION_KEYS,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def build_prompt_entry(region_key: str, category: str, template: str, idx: int) -> Dict:
    data = GEO_CULTURAL_TAXONOMY[region_key]
    loc = data["cultural_keywords"][idx % len(data["cultural_keywords"])]
    country = data["countries"][idx % len(data["countries"])]

    if category == "professions":
        prompt = f"A high-quality photo of {template} {loc}"
    elif category == "social_roles":
        prompt = f"A photo of {template} {loc}"
    elif category == "daily_activities":
        prompt = f"A photo of a person {template} {loc}"
    elif category == "cultural_practices":
        prompt = f"A photo of people {template} {loc}"
    else:
        prompt = f"A photo of a professional {template} {loc}"

    cultural_caption = f"{prompt}, authentic {country} cultural context"
    return {
        "id": f"{data['region_code']}_{category}_{idx:03d}",
        "region": region_key,
        "region_code": data["region_code"],
        "category": category,
        "prompt": prompt,
        "cultural_caption": cultural_caption,
    }


def generate_all_prompts(per_region: int = 100) -> Dict[str, List[Dict]]:
    per_cat = per_region // len(CATEGORIES)
    out = {}
    for region in REGION_KEYS:
        entries = []
        idx = 0
        for cat, templates in CATEGORIES.items():
            for t in templates[:per_cat]:
                entries.append(build_prompt_entry(region, cat, t, idx))
                idx += 1
        # Pad to exact count
        while len(entries) < per_region:
            entries.append(build_prompt_entry(region, "professions", "a doctor", idx))
            idx += 1
        out[region] = entries[:per_region]
    return out


def write_benchmark(path: Path = PROMPTS_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    all_data = generate_all_prompts(100)
    for region, entries in all_data.items():
        fname = path / f"{region}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    combined = []
    for entries in all_data.values():
        combined.extend(entries)
    with open(path / "gcfairbench_full.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(combined)} prompts to {path}")


def load_benchmark(path: Path = PROMPTS_DIR) -> List[Dict]:
    full = path / "gcfairbench_full.json"
    if not full.exists():
        write_benchmark(path)
    with open(full, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    write_benchmark()
