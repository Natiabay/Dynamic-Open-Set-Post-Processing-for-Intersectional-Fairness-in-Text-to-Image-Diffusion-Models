#!/usr/bin/env python3
"""Verify Base and CRAFT-GC produce different images (steering sanity check)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from craft_gc.benchmark.sampling import load_stratified
    from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return

    entry = load_stratified(1)[0]
    prompt = entry["prompt"]
    seed = 42

    pipe = CRAFTGCPipeline(device="cuda", load_sd=True)
    base = pipe.generate(prompt, method="base", seeds=[seed])["images"][0]
    craft = pipe.generate(prompt, method="craftgc", seeds=[seed])["images"][0]

    hb = hashlib.md5(base.tobytes()).hexdigest()
    hc = hashlib.md5(craft.tobytes()).hexdigest()

    out = ROOT / "results" / "steering_check"
    out.mkdir(parents=True, exist_ok=True)
    base.save(out / "base.png")
    craft.save(out / "craftgc.png")

    if hb == hc:
        print("FAIL: Base and CRAFT-GC images are identical.")
        sys.exit(1)
    print(f"PASS: steering changes output (base={hb[:8]}, craftgc={hc[:8]})")
    print(f"Saved {out}/base.png and craftgc.png")


if __name__ == "__main__":
    main()
