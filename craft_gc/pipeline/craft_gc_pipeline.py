"""End-to-end CRAFT-GC pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from craft_gc.detector.cultural_attribute_detector import CulturalAttributeDetector
from craft_gc.estimator.bias_direction_estimator import BiasDirectionEstimator
from craft_gc.estimator.sd_bias_direction_estimator import SDBiasDirectionEstimator
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
    """CRAFT-GC with OpenCLIP detection and SD-native cross-attention steering."""

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
        self.sd_estimator: Optional[SDBiasDirectionEstimator] = None
        self.scheduler = TimestepScheduler(num_steps, beta_max, t_star)
        self.pipe = None

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
            self.sd_estimator = SDBiasDirectionEstimator(
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                device=self.device,
            )
        except Exception as exc:
            print(f"[WARN] Stable Diffusion not loaded: {exc}")
            self.pipe = None
            self.sd_estimator = None

    def prepare(
        self, prompt: str, lambda_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[dict, Dict[str, torch.Tensor], Dict[str, float]]:
        detected = self.detector.detect(prompt)
        region = detected["region"]

        if self.sd_estimator is not None:
            directions = self.sd_estimator.compute_all(region)
        else:
            directions = self.estimator.compute_all(region)

        if lambda_weights is None:
            lambda_weights = default_lambdas(region)
        return detected, directions, lambda_weights

    def _install_cas(self, directions, lambdas) -> Optional[CASAttnProcessor]:
        if self.pipe is None:
            return None
        text_dim = 768
        proc = CASAttnProcessor(directions, lambdas, self.scheduler, text_dim=text_dim)
        procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            if "attn2" in name:
                procs[name] = proc
            else:
                procs[name] = self.pipe.unet.attn_processors[name]
        self.pipe.unet.set_attn_processor(procs)
        return proc

    def _reset_attn(self) -> None:
        if self.pipe is not None:
            self.pipe.unet.set_default_attn_processor()

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
            raise RuntimeError("SD pipeline not loaded.")

        detected, directions, lambdas = self.prepare(prompt)
        steps = num_inference_steps or self.num_steps
        seeds = seeds or [42]

        if method == "prompt_aug":
            prompt = f"{prompt}, diverse inclusive multicultural representation"

        images = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            cas_proc = None

            try:
                if method in ("craftgc", "fairimagen"):
                    scale = 1.0 if method == "craftgc" else 0.85
                    scaled = {k: v * scale for k, v in lambdas.items()}
                    cas_proc = self._install_cas(directions, scaled)

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
                    self._reset_attn()
                    out = self.pipe(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        height=height,
                        width=width,
                    )
            finally:
                self._reset_attn()

            images.extend(out.images)

        return {
            "images": images,
            "detected": detected,
            "directions": directions,
            "lambdas": lambdas,
            "prompt_used": prompt,
        }
