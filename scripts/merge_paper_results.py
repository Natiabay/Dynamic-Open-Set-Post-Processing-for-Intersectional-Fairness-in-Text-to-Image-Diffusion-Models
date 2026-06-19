#!/usr/bin/env python3
"""Merge image-level and embedding-level results into paper_results.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "base": "Base SD",
    "prompt_aug": "PromptAug",
    "fairimagen": "FairImagen",
    "craftgc": "CRAFT-GC",
    "fairimagen_e": "FairImagen-E",
    "craftgc_e": "CRAFT-GC-E",
}

REGIONS = [
    "sub_saharan_africa",
    "south_asia",
    "southeast_asia",
    "mena",
    "latin_america",
]


def load_image_level() -> dict | None:
    summary_path = ROOT / "results" / "image_eval_v2_summary.json"
    csv_path = ROOT / "results" / "image_eval_v2.csv"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        main_table = []
        for method, vals in summary.items():
            if method in ("metric_note",):
                continue
            if not isinstance(vals, dict):
                continue
            main_table.append(
                {
                    "method_key": method,
                    "method": LABELS.get(method, method),
                    "cofs": vals.get("cofs"),
                    "gender_parity": vals.get("gender_parity"),
                    "pct_female": vals.get("pct_female"),
                    "n": vals.get("n"),
                }
            )
        n_images = sum(r.get("n", 0) for r in main_table)
        n_prompts = 0
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            n_prompts = int(df["id"].nunique())
        return {
            "main_table": main_table,
            "summary": summary,
            "n_images": n_images,
            "n_prompts": n_prompts,
            "note": "Image-level CLIP zero-shot demographic classification (CoFS 0-1 scale).",
        }

    return None


def load_embedding_level() -> dict | None:
    summary_path = ROOT / "results" / "embedding_summary.json"
    csv_path = ROOT / "results" / "embedding_eval.csv"

    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        main_table = []
        methods = data.get("methods", data)
        for key, vals in methods.items():
            if key == "metric_note" or not isinstance(vals, dict):
                continue
            main_table.append(
                {
                    "method_key": key,
                    "method": LABELS.get(key, key),
                    "text_alignment_score": round(
                        vals.get(
                            "text_alignment_score_mean",
                            vals.get("cofs_mean", 0),
                        ),
                        4,
                    ),
                    "text_alignment_std": round(
                        vals.get(
                            "text_alignment_score_std",
                            vals.get("cofs_std", 0),
                        ),
                        4,
                    ),
                    "cfs": round(vals.get("cfs_mean", 0), 4),
                    "n": 500,
                }
            )
        return {
            "main_table": main_table,
            "metric_note": data.get("metric_note"),
            "note": "CLIP text-embedding alignment (not image-level CoFS).",
        }

    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        col = "text_alignment_score" if "text_alignment_score" in df.columns else "cofs"
        main_table = []
        for method in df["method"].unique():
            sub = df[df["method"] == method]
            main_table.append(
                {
                    "method_key": method,
                    "method": LABELS.get(method, method),
                    "text_alignment_score": round(float(sub[col].mean()), 4),
                    "text_alignment_std": round(float(sub[col].std()), 4),
                    "cfs": round(float(sub["cfs"].mean()), 4),
                    "n": len(sub),
                }
            )
        return {"main_table": main_table, "note": "From embedding_eval.csv"}

    return None


def main():
    image_level = load_image_level()
    embedding_level = load_embedding_level()

    if not image_level and not embedding_level:
        raise SystemExit(
            "No results found. Run run_pilot_images.py + evaluate_images.py "
            "or evaluate_embeddings.py first."
        )

    out = {
        "methods": ["base", "prompt_aug", "fairimagen", "craftgc"],
        "regions": REGIONS,
        "primary_metric": "image_level" if image_level else "embedding_level",
    }

    if image_level:
        out["image_level"] = image_level
        out["n_images"] = image_level.get("n_images", 0)
        out["n_prompts"] = image_level.get("n_prompts", 0)
    if embedding_level:
        out["embedding_level"] = embedding_level

    path = ROOT / "results" / "paper_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
