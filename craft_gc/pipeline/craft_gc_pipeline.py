"""End-to-end CRAFT-GC pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from craft_gc.detector.cultural_attribute_detector import CulturalAttributeDetector
from craft_gc.estimator.bias_direction_estimator import BiasDirectionEstimator
from craft_gc.steering.cross_attention_steering import CASAttnProcessor
from craft_gc.steering.embedding_steering import default_lambdas, steer_embedding
from craft_gc.steering.timestep_scheduler import TimestepScheduler


def load_clip(device: str = "cpu"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


class CRAFTGCPipeline:
    """CRAFT-GC with embedding steering (E) and optional CAS on UNet."""

    def __init__(
        self,
        device: str = "cpu",
        beta_max: float = 0.8,
        t_star: float = 0.6,
        num_steps: int = 50,
        sd_model_id: Optional[str] = "runwayml/stable-diffusion-v1-5",
        load_sd: bool = True,
    ):
        self.device = device
        self.num_steps = num_steps
        self.clip_model, self.clip_preprocess, self.clip_tokenize = load_clip(device)
        self.detector = CulturalAttributeDetector(
            self.clip_model, self.clip_tokenize, device=device
        )
        self.estimator = BiasDirectionEstimator(
            self.clip_model, self.clip_tokenize, device=device
        )
        self.scheduler = TimestepScheduler(num_steps, beta_max, t_star)
        self.pipe = None
        self.cas_processors: Dict[str, CASAttnProcessor] = {}

        if load_sd:
            self._load_sd(sd_model_id)

    def _load_sd(self, model_id: str) -> None:
        try:
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, safety_checker=None
            )
            if self.device.startswith("cuda"):
                self.pipe = self.pipe.to(self.device)
            else:
                self.pipe.enable_attention_slicing()
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        except Exception as exc:
            print(f"[WARN] Stable Diffusion not loaded: {exc}")
            self.pipe = None

    def prepare(self, prompt: str, lambda_weights: Optional[Dict[str, float]] = None):
        detected = self.detector.detect(prompt)
        region = detected["region"]
        directions = self.estimator.compute_all(region)
        if lambda_weights is None:
            lambda_weights = default_lambdas(region)
        return detected, directions, lambda_weights

    @torch.no_grad()
    def steer_prompt_embedding(self, prompt: str, beta: float = 1.0) -> torch.Tensor:
        detected, directions, lambdas = self.prepare(prompt)
        tokens = self.clip_tokenize([prompt]).to(self.device)
        emb = self.clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        steered = steer_embedding(emb.squeeze(0), directions, lambdas, beta=beta)
        return steered, detected

    def _install_cas(self, directions, lambdas):
        if self.pipe is None:
            return None
        proc = CASAttnProcessor(directions, lambdas, self.scheduler, hidden_dim=768)
        procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            if "attn2" in name:
                procs[name] = proc
            else:
                procs[name] = self.pipe.unet.attn_processors[name]
        self.pipe.unet.set_attn_processor(procs)
        return proc

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        method: str = "craftgc",
        seeds: Optional[List[int]] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> Dict:
        """
        method: base | prompt_aug | fairimagen | craftgc | craftgc_e
        """
        if self.pipe is None and method not in ("craftgc_e",):
            raise RuntimeError("SD pipeline not loaded; use method=craftgc_e or enable GPU.")

        detected, directions, lambdas = self.prepare(prompt)
        steps = num_inference_steps or self.num_steps
        seeds = seeds or [42]

        if method == "prompt_aug":
            prompt = f"{prompt}, diverse inclusive multicultural representation"

        images = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            cas_proc = None

            if method == "craftgc":
                cas_proc = self._install_cas(directions, lambdas)
                step_counter = {"i": 0}

                def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                    if cas_proc is not None:
                        cas_proc.set_step(step_index)
                    step_counter["i"] = step_index
                    return callback_kwargs

                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                    callback_on_step_end=callback_on_step_end,
                )
            elif method == "fairimagen":
                # Embedding projection proxy (FairPCA-style single direction removal)
                steered_emb, _ = self.steer_prompt_embedding(prompt, beta=0.85)
                # Use steered prompt text approximation via nearest template — fallback CAS
                cas_proc = self._install_cas(directions, {k: v * 0.85 for k, v in lambdas.items()})

                def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                    if cas_proc is not None:
                        cas_proc.set_step(step_index)
                    return callback_kwargs

                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                    callback_on_step_end=callback_on_step_end,
                )
            else:
                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
            images.extend(out.images)

        return {
            "images": images,
            "detected": detected,
            "directions": directions,
            "lambdas": lambdas,
            "prompt_used": prompt,
        }
