#!/usr/bin/env python3
"""Build qualitative comparison grid from pilot_images manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/pilot_images/manifest.json")
    parser.add_argument("--prompt-ids", nargs="+", default=["ssa_prof_001", "sa_prof_010", "mena_prof_020"])
    parser.add_argument("--out", default="craft-gc-paper/figures/qualitative_grid.png")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}; run Colab GPU experiment first.")
        return

    manifest = json.loads(manifest_path.read_text())
    by_key = {}
    for item in manifest:
        key = (item["id"], item["mode"], item["seed"])
        by_key[key] = item

    modes = ["base", "prompt_aug", "fairimagen", "craftgc"]
    seed = 42
    cell = 256
    header_h = 28
    rows = []
    for pid in args.prompt_ids:
        row_imgs = []
        for mode in modes:
            item = by_key.get((pid, mode, seed))
            if item and Path(item["path"]).exists():
                row_imgs.append(Image.open(item["path"]).convert("RGB").resize((cell, cell)))
            else:
                placeholder = Image.new("RGB", (cell, cell), (240, 240, 240))
                d = ImageDraw.Draw(placeholder)
                d.text((20, cell // 2), f"{mode}\n(missing)", fill=(80, 80, 80))
                row_imgs.append(placeholder)
        rows.append(row_imgs)

    if not rows:
        print("No images found for grid.")
        return

    w = cell * len(modes)
    h = header_h + cell * len(rows)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    for j, mode in enumerate(modes):
        draw.text((j * cell + 8, 6), mode, fill=(0, 0, 0))
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            canvas.paste(img, (j * cell, header_h + i * cell))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    canvas.save(out.with_suffix(".pdf"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
