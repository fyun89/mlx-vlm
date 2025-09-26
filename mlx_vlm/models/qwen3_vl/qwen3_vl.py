# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Any, Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3VLConfig
from .vision import Qwen3Vision
from ..qwen2_5_vl import pack_vision_embeds  # packing is generic

# text decoder: reuse Qwen3 / Qwen3-MoE from mlx-lm
from mlx_lm.models import qwen3 as qwen3_dense
from mlx_lm.models import qwen3_moe as qwen3_moe

class Qwen3VL(nn.Module):
    """
    Minimal multimodal model:
      vision: Qwen3Vision → projector (linear) to text hidden
      text:   Qwen3 (dense or MoE) decoder
    """
    def __init__(self, cfg: Qwen3VLConfig):
        super().__init__()
        self.cfg = cfg
        # ---- vision ----
        self.vision = Qwen3Vision(cfg.vision_config)
        vdim = int(cfg.vision_config.get("hidden_size", 1024))
        tdim = int(cfg.text_config.get("hidden_size", 4096))
        self.projector = nn.Linear(vdim, tdim, bias=True)  # if HF exposes explicit projector, your converter will map it here
        # ---- decoder ----
        if "moe" in cfg.model_type:
            targs = qwen3_moe.ModelArgs.from_dict(cfg.text_config)
            self.decoder = qwen3_moe.Model(targs)
        else:
            targs = qwen3_dense.ModelArgs.from_dict(cfg.text_config)
            self.decoder = qwen3_dense.Model(targs)
        # token ids
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id
        self.vision_token_id = cfg.vision_token_id

    # ----- helpers -----
    def encode_images(self, images: List[mx.array]) -> mx.array:
        feats = self.vision(images)          # [B, T_img, vdim]
        feats = self.projector(feats)        # [B, T_img, tdim]
        return feats

    # ----- forward -----
    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        images: Optional[List[mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        cache: Any = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if images is not None and len(images) > 0:
            img_embeds = self.encode_images(images)
            x, mask = pack_vision_embeds(
                input_ids,
                img_embeds,
                attention_mask,
                start_id=self.vision_start_id,
                end_id=self.vision_end_id,
                place_id=self.vision_token_id,
            )
            return self.decoder(None, input_embeddings=x, mask=mask, cache=cache)
        else:
            # pure text
            if input_embeddings is not None:
                return self.decoder(None, input_embeddings=input_embeddings, mask=attention_mask, cache=cache)
            return self.decoder(input_ids, mask=attention_mask, cache=cache)

    # expose layers (nice for generation utils)
    @property
    def layers(self):
        return self.decoder.model.layers