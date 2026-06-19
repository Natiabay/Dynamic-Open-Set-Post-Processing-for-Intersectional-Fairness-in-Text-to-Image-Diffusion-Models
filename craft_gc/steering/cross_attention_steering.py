"""Cross-attention steering on SD text encoder states (768-d)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from craft_gc.steering.timestep_scheduler import TimestepScheduler


class CASAttnProcessor:
    """Remove bias components from cross-attention encoder states before K/V projection."""

    def __init__(
        self,
        bias_directions: Dict[str, torch.Tensor],
        lambda_weights: Dict[str, float],
        scheduler: TimestepScheduler,
        text_dim: int = 768,
    ):
        self.bias_directions = {k: v.float() for k, v in bias_directions.items()}
        self.lambda_weights = lambda_weights
        self.scheduler = scheduler
        self.text_dim = text_dim
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def _steer_encoder(self, enc: torch.Tensor) -> torch.Tensor:
        beta = self.scheduler.get_beta(self.current_step)
        if beta < 1e-6:
            return enc

        out = enc.clone()
        for name, direction in self.bias_directions.items():
            lam = self.lambda_weights.get(name, 0.0)
            if lam <= 0:
                continue
            d = direction.to(out.device)
            if d.shape[-1] != out.shape[-1]:
                continue
            d = d.view(1, 1, -1)
            d_norm_sq = (d * d).sum(dim=-1, keepdim=True) + 1e-8
            proj = (out * d).sum(dim=-1, keepdim=True) / d_norm_sq
            out = out - beta * lam * proj * d
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = self._steer_encoder(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        heads = getattr(attn, "heads", getattr(attn, "num_heads", 8))
        head_dim = inner_dim // heads

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        if attention_mask is not None and hasattr(attn, "prepare_attention_mask"):
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        scale = getattr(attn, "resample_scale_with_output_factor", 1.0)
        if scale and scale != 1.0:
            hidden_states = hidden_states / scale

        return hidden_states
