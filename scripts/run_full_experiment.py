"""
Full CRAFT-GC experiment runner for Google Colab (GPU).

Upload this repo to Colab or clone from GitHub, then run:
  !pip install -q torch torchvision open-clip-torch diffusers transformers accelerate pandas matplotlib seaborn
  !python scripts/run_full_experiment.py --hf-token $HF_TOKEN

Set HF_TOKEN in Colab secrets or paste at prompt (never commit tokens).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--prompt-limit", type=int, default=50, help="Image prompts (50 for pilot, 500 for full)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        run([sys.executable, "-c", "from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"])

    # Stage A: CLIP embedding (all 500)
    run([sys.executable, "scripts/evaluate_embeddings.py", "--device", args.device])
    run([sys.executable, "scripts/merge_paper_results.py"])
    run([sys.executable, "scripts/run_ablation.py"])

    if args.skip_images:
        run([sys.executable, "scripts/plot_figures.py"])
        return

    # Stage B: SD image generation
    run([
        sys.executable,
        "scripts/run_pilot_images.py",
        "--limit",
        str(args.prompt_limit),
        "--device",
        args.device,
        "--seeds",
        *[str(s) for s in args.seeds],
        "--methods",
        "base",
        "prompt_aug",
        "fairimagen",
        "craftgc",
    ])

    run([sys.executable, "scripts/evaluate_images.py", "--device", args.device])
    run([sys.executable, "scripts/make_qualitative_grid.py"])
    run([sys.executable, "scripts/plot_figures.py"])

    summary = {
        "embedding": str(ROOT / "results" / "embedding_summary.json"),
        "images": str(ROOT / "results" / "image_eval.csv"),
        "figures": str(ROOT / "craft-gc-paper" / "figures"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
