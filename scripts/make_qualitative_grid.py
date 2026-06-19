#!/usr/bin/env python3
"""Build qualitative comparison grid from manifest using relative paths."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

METHODS = ["base", "prompt_aug", "fairimagen", "craftgc"]
METHOD_LABELS = ["Base SD", "Prompt Aug", "FairImagen", "CRAFT-GC"]
THUMB = 256
LABEL_H = 36
PROMPT_H = 48


def resolve_image(root: Path, filepath: str, method: str) -> Path | None:
    candidates = []
    if filepath:
        candidates.extend([
            root / filepath,
            Path(filepath),
            root / "results" / "pilot_images" / Path(filepath).name,
            root / "results" / "pilot_images" / method / Path(filepath).name,
        ])
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def make_grid(
    manifest_path="results/pilot_images/manifest.json",
    out_path="results/qualitative_grid.png",
    n_prompts=3,
    seeds=None,
):
    if seeds is None:
        seeds = [42]

    root = ROOT
    manifest = json.loads((root / manifest_path).read_text())

    groups = defaultdict(dict)
    for entry in manifest:
        method = entry.get("method") or entry.get("mode")
        key = (entry["id"], entry["seed"])
        groups[key][method] = entry.get("filepath") or entry.get("path", "")

    target_seed = seeds[0]
    filtered = {k: v for k, v in groups.items() if str(k[1]) == str(target_seed)}

    selected_keys = []
    seen_cats = set()
    for (pid, seed), methods in filtered.items():
        cat = pid.split("_")[1] if "_" in pid else "other"
        if cat not in seen_cats and all(m in methods for m in METHODS):
            all_exist = all(
                resolve_image(root, methods[m], m) is not None for m in METHODS
            )
            if all_exist:
                selected_keys.append((pid, seed))
                seen_cats.add(cat)
            if len(selected_keys) >= n_prompts:
                break

    if not selected_keys:
        for (pid, seed), methods in filtered.items():
            if len(methods) == 4 and all(m in methods for m in METHODS):
                all_exist = all(
                    resolve_image(root, methods.get(m, ""), m) is not None for m in METHODS
                )
                if all_exist:
                    selected_keys.append((pid, seed))
            if len(selected_keys) >= n_prompts:
                break

    if not selected_keys:
        raise SystemExit("No complete prompt rows found in manifest. Run run_pilot_images.py first.")

    n_rows = len(selected_keys)
    n_cols = len(METHODS)
    total_w = n_cols * THUMB
    total_h = n_rows * (THUMB + LABEL_H) + LABEL_H + PROMPT_H * n_rows

    canvas = Image.new("RGB", (total_w, total_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    try:
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
        font_prompt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except OSError:
        font_label = ImageFont.load_default()
        font_prompt = font_label

    for j, label in enumerate(METHOD_LABELS):
        x = j * THUMB + THUMB // 2
        draw.text((x, 10), label, fill=(30, 30, 30), font=font_label, anchor="mm")

    y_offset = LABEL_H
    for pid, seed in selected_keys:
        methods = filtered.get((pid, seed), {})
        prompt_text = next((e["prompt"] for e in manifest if e["id"] == pid), pid)
        if len(prompt_text) > 70:
            prompt_text = prompt_text[:67] + "..."

        draw.text(
            (total_w // 2, y_offset + 10),
            prompt_text,
            fill=(80, 80, 80),
            font=font_prompt,
            anchor="mm",
        )
        y_offset += PROMPT_H

        for col_i, method in enumerate(METHODS):
            x = col_i * THUMB
            y = y_offset
            filepath = methods.get(method, "")
            actual_path = resolve_image(root, filepath, method)

            if actual_path:
                img = Image.open(actual_path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x + THUMB, y + THUMB], fill=(220, 220, 220))
                draw.text(
                    (x + THUMB // 2, y + THUMB // 2),
                    "missing",
                    fill=(150, 150, 150),
                    font=font_label,
                    anchor="mm",
                )

            draw.rectangle([x, y + THUMB, x + THUMB, y + THUMB + LABEL_H], fill=(240, 240, 240))
            draw.text(
                (x + THUMB // 2, y + THUMB + LABEL_H // 2),
                METHOD_LABELS[col_i],
                fill=(40, 40, 40),
                font=font_label,
                anchor="mm",
            )

        y_offset += THUMB + LABEL_H

    out = root / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300))

    paper_out = root / "craft-gc-paper" / "figures" / "qualitative_grid.png"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(paper_out, dpi=(300, 300))

    print(f"Saved qualitative grid: {out}  ({n_rows} rows × {n_cols} cols)")
    print(f"Saved: {paper_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/pilot_images/manifest.json")
    parser.add_argument("--out", default="results/qualitative_grid.png")
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    make_grid(args.manifest, args.out, args.n_prompts, args.seeds)


if __name__ == "__main__":
    main()
