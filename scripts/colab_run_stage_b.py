#!/usr/bin/env python3
"""Run Stage B on Colab: SD image generation + evaluation + figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. In Colab: Runtime → Change runtime type → T4 GPU, then restart."
        )
    print("GPU:", torch.cuda.get_device_name(0))


def require_deps():
    missing = []
    for pkg in ("torch", "diffusers", "transformers", "accelerate", "open_clip"):
        try:
            __import__(pkg if pkg != "open_clip" else "open_clip")
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nRun: pip install torch diffusers transformers accelerate open-clip-torch"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    require_deps()
    require_cuda()

    import torch
    from craft_gc.benchmark.gcfairbench import load_benchmark
    from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline

    methods = ["base", "prompt_aug", "fairimagen", "craftgc"]
    entries = load_benchmark()[: args.limit]
    out_dir = ROOT / "results" / "pilot_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Stable Diffusion on {args.device}...")
    pipe = CRAFTGCPipeline(device=args.device, load_sd=True)
    if pipe.pipe is None:
        raise SystemExit(
            "Stable Diffusion failed to load. Check HF_TOKEN and diffusers install."
        )

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
                        "mode": mode,
                        "seed": seed,
                        "path": str(path),
                    }
                )

    meta = out_dir / "manifest.json"
    meta.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} images to {out_dir}")

    import subprocess

    for script, extra in (
        (
            "evaluate_images.py",
            [
                "--manifest",
                str(meta),
                "--output",
                str(ROOT / "results" / "image_eval.csv"),
                "--device",
                args.device,
            ],
        ),
        ("make_qualitative_grid.py", []),
    ):
        cmd = [sys.executable, str(ROOT / "scripts" / script)] + extra
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("Stage B complete. Download results/ and craft-gc-paper/figures/")


if __name__ == "__main__":
    main()
