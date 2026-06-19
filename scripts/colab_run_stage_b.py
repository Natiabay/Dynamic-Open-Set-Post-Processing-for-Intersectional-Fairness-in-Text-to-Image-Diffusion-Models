#!/usr/bin/env python3
"""Run Stage B: stratified SD image generation + evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-region", type=int, default=20, help="Prompts per region (5 regions)")
    parser.add_argument("--limit", type=int, default=0, help="If >0, use first N prompts (legacy pilot)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--out", default="results/pilot_images")
    args = parser.parse_args()

    import torch
    from craft_gc.benchmark.gcfairbench import load_benchmark
    from craft_gc.benchmark.sampling import load_stratified
    from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Set Colab runtime to T4 GPU.")

    if args.limit > 0:
        entries = load_benchmark()[: args.limit]
    else:
        entries = load_stratified(args.per_region)

    methods = ["base", "prompt_aug", "fairimagen", "craftgc"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prompts: {len(entries)} | Methods: {len(methods)} | Seeds: {len(args.seeds)}")
    print(f"Total images: {len(entries) * len(methods) * len(args.seeds)}")

    pipe = CRAFTGCPipeline(device=args.device, load_sd=True)
    if pipe.pipe is None or pipe.sd_estimator is None:
        raise SystemExit("Stable Diffusion failed to load. Check HF_TOKEN.")

    manifest = []
    total = len(entries) * len(args.seeds) * len(methods)
    done = 0

    for entry in entries:
        prompt = entry["prompt"]
        for seed in args.seeds:
            for mode in methods:
                done += 1
                print(f"[{done}/{total}] {entry['id']} | {mode} | seed={seed}")
                result = pipe.generate(prompt, method=mode, seeds=[seed])
                img = result["images"][0]
                fname = f"{entry['id']}_{mode}_s{seed}.png"
                path = out_dir / fname
                img.save(path)
                manifest.append(
                    {
                        "id": entry["id"],
                        "prompt": prompt,
                        "region": entry["region"],
                        "category": entry.get("category"),
                        "mode": mode,
                        "seed": seed,
                        "path": str(path),
                    }
                )

    meta = out_dir / "manifest.json"
    meta.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} images to {out_dir}")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_images.py"),
            "--manifest",
            str(meta),
            "--output",
            str(ROOT / "results" / "image_eval.csv"),
            "--device",
            args.device,
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_qualitative_grid.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_results.py")], cwd=ROOT, check=True)
    print("Stage B complete.")


if __name__ == "__main__":
    main()
