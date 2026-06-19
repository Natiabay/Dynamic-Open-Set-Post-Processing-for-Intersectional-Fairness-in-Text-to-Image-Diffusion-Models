"""
CRAFT-GC interactive demo for Hugging Face Spaces.

Deploy: create Space with Gradio SDK, GPU hardware, add HF_TOKEN secret.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results" / "pilot_images" / "manifest.json"
BENCH = ROOT / "craft_gc" / "benchmark" / "prompts" / "gcfairbench_full.json"

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        from craft_gc.pipeline.craft_gc_pipeline import CRAFTGCPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipe = CRAFTGCPipeline(device=device, load_sd=True)
    return _pipe


def load_gallery_prompts():
    if BENCH.exists():
        data = json.loads(BENCH.read_text())
        examples = []
        for region in ["sub_saharan_africa", "south_asia", "mena"]:
            for entry in data:
                if entry["region"] == region and entry["category"] == "professions":
                    examples.append(entry["prompt"])
                    break
        return examples[:6]
    return [
        "A high-quality photo of a doctor in Addis Ababa",
        "A high-quality photo of a teacher in Mumbai",
        "A high-quality photo of a lawyer in Cairo",
    ]


def generate_comparison(prompt, seed, progress=gr.Progress()):
    if not prompt.strip():
        return None, None, "Enter a prompt."

    progress(0.1, desc="Loading pipeline")
    pipe = get_pipeline()
    if pipe.pipe is None:
        return None, None, "Stable Diffusion not available on this Space (enable GPU)."

    progress(0.3, desc="Base SD")
    base = pipe.generate(prompt, method="base", seeds=[int(seed)])["images"][0]

    progress(0.7, desc="CRAFT-GC")
    craft = pipe.generate(prompt, method="craftgc", seeds=[int(seed)])["images"][0]

    detected = pipe.detector.detect(prompt)
    region = detected.get("region") or "none"
    score = detected.get("score", 0)
    info = f"CAD region: {region} (score={score:.2f}) | seed={seed}"
    return base, craft, info


def gallery_view(prompt_id, seed):
    if not MANIFEST.exists():
        return [None] * 4, "Upload results/pilot_images from Colab to enable gallery."

    manifest = json.loads(MANIFEST.read_text())
    modes = ["base", "prompt_aug", "fairimagen", "craftgc"]
    imgs = []
    for mode in modes:
        match = [x for x in manifest if x["id"] == prompt_id and x["mode"] == mode and x["seed"] == seed]
        if match and Path(match[0]["path"]).exists():
            from PIL import Image

            imgs.append(Image.open(match[0]["path"]))
        else:
            imgs.append(None)
    return imgs, f"{prompt_id} | seed {seed}"


with gr.Blocks(title="CRAFT-GC Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CRAFT-GC: Geo-Cultural Fairness Steering for Stable Diffusion

        Compare **Base Stable Diffusion** with **CRAFT-GC** (cross-attention steering).
        GCFairBench targets under-represented regions in text-to-image generation.
        """
    )

    with gr.Tab("Try a prompt"):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A high-quality photo of a doctor in Addis Ababa",
                scale=3,
            )
            seed = gr.Number(label="Seed", value=42, precision=0)
        btn = gr.Button("Generate comparison", variant="primary")
        info = gr.Textbox(label="Detection", interactive=False)
        with gr.Row():
            out_base = gr.Image(label="Base SD")
            out_craft = gr.Image(label="CRAFT-GC")
        gr.Examples(examples=[[p, 42] for p in load_gallery_prompts()], inputs=[prompt, seed])
        btn.click(generate_comparison, [prompt, seed], [out_base, out_craft, info])

    with gr.Tab("Benchmark gallery"):
        gr.Markdown("Pre-generated GCFairBench images (after Colab Stage B).")
        pid = gr.Textbox(label="Prompt ID", value="SSA_professions_000")
        gseed = gr.Number(label="Seed", value=42, precision=0)
        gbtn = gr.Button("Load grid")
        ginfo = gr.Textbox(label="Info", interactive=False)
        with gr.Row():
            g1 = gr.Image(label="Base")
            g2 = gr.Image(label="PromptAug")
            g3 = gr.Image(label="FairImagen")
            g4 = gr.Image(label="CRAFT-GC")
        gbtn.click(gallery_view, [pid, gseed], [[g1, g2, g3, g4], ginfo])

if __name__ == "__main__":
    demo.launch()
