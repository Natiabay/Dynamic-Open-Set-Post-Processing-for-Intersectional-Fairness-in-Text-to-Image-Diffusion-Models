#!/usr/bin/env python3
"""Generate a small SD image pilot on GPU for qualitative figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--methods", nargs="+", default=["base", "prompt_aug", "fairimagen", "craftgc"])
    parser.add_argument("--out", default="results/pilot_images")
    args = parser.parse_args()

    try:
        import torch
        from craft_gc.benchmark.gcfairbench import load_benchmark
        from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing deps. Install: pip install torch diffusers open-clip-torch transformers accelerate"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Use --device cpu (slow) or run on a GPU machine.")

    entries = load_benchmark()[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = CRAFTGCPipeline(device=args.device, load_sd=True)
    manifest = []

    for entry in entries:
        prompt = entry["prompt"]
        for seed in args.seeds:
            for mode in args.methods:
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
                        "mode": mode,
                        "seed": seed,
                        "path": str(path),
                    }
                )

    meta = out_dir / "manifest.json"
    meta.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} images to {out_dir}")


if __name__ == "__main__":
    main()
