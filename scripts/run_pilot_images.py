#!/usr/bin/env python3
"""Generate pilot images with Realistic Vision V5.1 and four debiasing methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, sketch, unrealistic, "
    "3d render, cgi, digital art, lowres, blurry, watermark, text, logo, "
    "deformed, ugly"
)
PHOTO_SUFFIX = ", RAW photo, realistic, photorealistic, 8k, professional photography"
NUM_STEPS = 30
GUIDANCE = 7.5


class CASAttnProcessor:
    """CRAFT-GC cross-attention steering on attention keys."""

    def __init__(self, bias_directions, lambda_weights, beta, device):
        self.bias_directions = bias_directions
        self.lambda_weights = lambda_weights
        self.beta = beta
        self.device = device
        self.step = 0
        self.total_steps = NUM_STEPS

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        residual = hidden_states
        if hasattr(attn, "group_norm") and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        batch_size, _, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(context)
        value = attn.to_v(context)

        t_norm = self.step / max(self.total_steps, 1)
        beta_t = self.beta * torch.sigmoid(torch.tensor(10.0 * (t_norm - 0.6))).item()
        self.step += 1

        if beta_t > 1e-4 and encoder_hidden_states is not None:
            for attr_name, direction in self.bias_directions.items():
                d = direction.to(key.device).float()
                lam = self.lambda_weights.get(attr_name, 1.0)
                d_norm = d / (d.norm() + 1e-8)
                proj = key.float() @ d_norm.unsqueeze(-1)
                key = key - (beta_t * lam * proj * d_norm.unsqueeze(0).unsqueeze(0)).to(key.dtype)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def load_pipeline(model_id: str, vae_id: str, device: str):
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


def with_photo(prompt: str) -> str:
    return f"{prompt}{PHOTO_SUFFIX}"


def apply_fairimagen(pipe, prompt, negative_prompt, seed):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    with torch.no_grad():
        text_embeds = pipe.text_encoder(text_inputs.input_ids)[0]

    male_inputs = pipe.tokenizer(
        ["a man", "a male person", "a boy", "he", "his"],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    neutral_inputs = pipe.tokenizer(
        ["a person", "someone", "an individual", "they", "their"],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)

    with torch.no_grad():
        male_embeds = pipe.text_encoder(male_inputs.input_ids)[0].mean(0)
        neutral_embeds = pipe.text_encoder(neutral_inputs.input_ids)[0].mean(0)

    gender_direction = male_embeds - neutral_embeds
    gender_direction = gender_direction / (gender_direction.norm(dim=-1, keepdim=True) + 1e-8)

    proj_coef = (text_embeds * gender_direction).sum(dim=-1, keepdim=True)
    text_embeds_fair = text_embeds - 0.7 * proj_coef * gender_direction

    noise_scale = 0.05 * text_embeds_fair.std()
    text_embeds_fair = text_embeds_fair + noise_scale * torch.randn_like(text_embeds_fair)

    neg_inputs = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    with torch.no_grad():
        neg_embeds = pipe.text_encoder(neg_inputs.input_ids)[0]

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt_embeds=text_embeds_fair,
        negative_prompt_embeds=neg_embeds,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    ).images[0]
    return image


def compute_bias_directions(pipe, device):
    def encode(texts):
        tokens = pipe.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            embeds = pipe.text_encoder(tokens.input_ids)[0]
        return embeds.mean(dim=1)

    male_emb = encode(["a man", "a male doctor", "a male engineer", "he", "his", "him"])
    neutral_emb = encode(["a person", "a doctor", "an engineer", "they", "their", "them"])
    gender_dir = male_emb.mean(0) - neutral_emb.mean(0)
    gender_dir = gender_dir / (gender_dir.norm() + 1e-8)

    light_emb = encode(["a white person", "a pale-skinned doctor", "a fair-skinned engineer"])
    dark_emb = encode(["a dark-skinned person", "an African doctor", "a person of color"])
    skin_dir = light_emb.mean(0) - dark_emb.mean(0)
    skin_dir = skin_dir / (skin_dir.norm() + 1e-8)

    return {"gender": gender_dir, "skin_tone": skin_dir}


def apply_craftgc(pipe, prompt, negative_prompt, seed, beta=0.8):
    device = pipe.device
    bias_directions = compute_bias_directions(pipe, device)
    lambda_weights = {"gender": 0.7, "skin_tone": 0.5}

    original_processors = {}
    cas = CASAttnProcessor(
        bias_directions=bias_directions,
        lambda_weights=lambda_weights,
        beta=beta,
        device=device,
    )

    new_processors = {}
    for name, proc in pipe.unet.attn_processors.items():
        original_processors[name] = proc
        if "attn2" in name:
            new_processors[name] = cas
        else:
            new_processors[name] = proc

    pipe.unet.set_attn_processor(new_processors)
    try:
        cas.step = 0
        cas.total_steps = NUM_STEPS
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]
    finally:
        pipe.unet.set_attn_processor(original_processors)
    return image


def generate_image(pipe, prompt, method, seed):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if method == "base":
        return pipe(
            prompt=with_photo(prompt),
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]

    if method == "prompt_aug":
        aug_prompt = (
            f"A diverse group of people, {prompt.lower()}, photorealistic, professional photo"
        )
        return pipe(
            prompt=aug_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]

    if method == "fairimagen":
        return apply_fairimagen(pipe, with_photo(prompt), NEGATIVE_PROMPT, seed)

    if method == "craftgc":
        return apply_craftgc(pipe, with_photo(prompt), NEGATIVE_PROMPT, seed)

    raise ValueError(f"Unknown method: {method}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of prompts (max 100)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["base", "prompt_aug", "fairimagen", "craftgc"],
    )
    parser.add_argument("--out", default="results/pilot_images")
    parser.add_argument("--model_id", default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--vae_id", default="stabilityai/sd-vae-ft-mse")
    args = parser.parse_args()

    if args.limit > 100:
        raise SystemExit("--limit max is 100")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required. Set Colab runtime to GPU.")

    from craft_gc.benchmark.gcfairbench import load_benchmark, write_benchmark

    write_benchmark()
    entries = load_benchmark()[: args.limit]
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} + VAE {args.vae_id}...")
    pipe = load_pipeline(args.model_id, args.vae_id, args.device)

    manifest = []
    total = len(entries) * len(args.seeds) * len(args.methods)

    with tqdm(total=total, desc="Generating images") as pbar:
        for entry in entries:
            prompt = entry["prompt"]
            for seed in args.seeds:
                for method in args.methods:
                    method_dir = out_root / method
                    method_dir.mkdir(parents=True, exist_ok=True)

                    img = generate_image(pipe, prompt, method, seed)
                    fname = f"{entry['id']}_s{seed}.png"
                    save_path = method_dir / fname
                    img.save(save_path)

                    rel_path = str(Path(args.out) / method / fname)
                    manifest.append(
                        {
                            "id": entry["id"],
                            "prompt": prompt,
                            "region": entry["region"],
                            "method": method,
                            "seed": seed,
                            "filepath": rel_path,
                        }
                    )
                    pbar.update(1)

    meta = out_root / "manifest.json"
    meta.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} images under {out_root}")
    print(f"Manifest: {meta}")


if __name__ == "__main__":
    main()
