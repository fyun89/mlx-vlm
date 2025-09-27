# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3VLConfig
from .vision import Qwen3Vision

# text decoder: reuse Qwen3 / Qwen3-MoE from mlx-lm
from mlx_lm.models import qwen3 as qwen3_dense
from mlx_lm.models import qwen3_moe as qwen3_moe

def _embed_tokens(lang_model: nn.Module, input_ids: mx.array) -> mx.array:
    """
    Returns token embeddings for input_ids using whatever the Qwen3 text backbone exposes.
    Tries common attribute names to be robust across mlx-lm versions.
    """
    m = getattr(lang_model, "model", lang_model)  # unwrap if needed
    if hasattr(m, "embed_tokens"):
        return m.embed_tokens(input_ids)
    if hasattr(m, "tok_embeddings"):
        return m.tok_embeddings(input_ids)
    # Fall back: many decoders expose an 'embed' method; try it last.
    if hasattr(m, "embed"):
        return m.embed(input_ids)
    raise AttributeError("Cannot find embedding layer on Qwen3 language model.")


def _pack_vision_embeds(
    lang_model: nn.Module,
    input_ids: mx.array,
    img_embeds: mx.array,
    attention_mask: Optional[mx.array],
    *,
    start_id: int,
    end_id: int,
    place_id: int,
) -> Tuple[mx.array, Optional[mx.array]]:
    """
    Replace a single <vision> span:
        ... <start_id> [place_id x N] <end_id> ...
    with the actual `img_embeds` sequence. Assumes batch=1 for simplicity.

    Returns:
        input_embeddings (mx.array [1, T', D]), attention_mask ([1, T'] or None)
    """
    if input_ids is None:
        raise ValueError("input_ids are required when packing vision embeddings.")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("This minimal helper currently supports batch=1.")

    ids = input_ids[0]
    # locate the first span
    start_pos = None
    end_pos = None
    for i, tok in enumerate(ids.tolist()):
        if start_pos is None and tok == start_id:
            start_pos = i
        elif start_pos is not None and tok == end_id:
            end_pos = i
            break
    if start_pos is None or end_pos is None or end_pos <= start_pos:
        raise ValueError("Did not find a valid <vision> span in input_ids.")

    # embed tokens
    tok_embeds = _embed_tokens(lang_model, input_ids)  # [1, T, D]
    before = tok_embeds[:, :start_pos, :]
    after  = tok_embeds[:, end_pos + 1 :, :]

    # concatenate: before + img_embeds + after
    # ensure img_embeds is [1, T_img, D]
    if img_embeds.ndim == 2:
        img_embeds = img_embeds[None, ...]
    packed = mx.concatenate([before, img_embeds, after], axis=1)

    if attention_mask is not None:
        am_before = attention_mask[:, :start_pos]
        am_after  = attention_mask[:, end_pos + 1 :]
        am_img    = mx.ones((attention_mask.shape[0], img_embeds.shape[1]), dtype=attention_mask.dtype)
        new_mask  = mx.concatenate([am_before, am_img, am_after], axis=1)
    else:
        new_mask = None

    return packed, new_mask

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
        # tcfg = dict(cfg.text_config)
        # required = {
        #        "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
        #        "num_attention_heads", "num_experts", "num_experts_per_tok",
        #        "decoder_sparse_step", "mlp_only_layers", "moe_intermediate_size",
        #        "rms_norm_eps", "vocab_size", "num_key_value_heads", "head_dim",
        #        "rope_theta", "max_position_embeddings", "norm_topk_prob",
        #        # common extras that often matter:
        #        "rope_scaling", "rope_traditional",
        # }

        # raw = getattr(cfg, "raw_config", None) or {}
        # for k in required:
        #     if k not in tcfg and k in raw:
        #         tcfg[k] = raw[k]
        # # Also pick from raw["text_config"] if present (defensive)
        # raw_text = (raw.get("text_config") or {}) if isinstance(raw, dict) else {}
        # for k in required:
        #     if k not in tcfg and k in raw_text:
        #         tcfg[k] = raw_text[k]
        # tcfg.setdefault("tie_word_embeddings", False)
        # tcfg.setdefault("mlp_only_layers", [])
        # if "moe" in cfg.model_type:
        #     targs = qwen3_moe.ModelArgs.from_dict(tcfg)
        #     self.decoder = qwen3_moe.Model(targs)
        # else:
        #     targs = qwen3_dense.ModelArgs.from_dict(tcfg)
        #     self.decoder = qwen3_dense.Model(targs)
        #   # token ids
        
        # # ----- expose submodules under language_model.model.* so keys match HF -----
        # lm_core = self.decoder.model  # inner core that has .layers, etc.

        # # 1) Visual tower
        # # keep your own attributes for convenience...
        # self.visual = self.vision
        # # ...but also attach to the decoder core so names become language_model.model.visual.*
        # lm_core.visual = self.visual

        # # 2) Projector (visual -> text)
        # # HF usually places this under mm_projector.* ; map ours there
        # lm_core.mm_projector = self.projector

        # # 3) Final norm and LM head (HF expects these on the core)
        # lm_core.norm = nn.RMSNorm(tdim, eps=float(self.cfg.text_config.get("rms_norm_eps", 1e-6)))
        # lm_core.lm_head = nn.Linear(
        #     tdim, int(self.cfg.text_config.get("vocab_size", 32000)), bias=False
        # )

        # self.vision_start_id = cfg.vision_start_token_id
        # self.vision_end_id = cfg.vision_end_token_id
        # self.vision_token_id = cfg.vision_token_id
        
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
            x, mask = _pack_vision_embeds(
                self.decoder,
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
    
    def load_weights(self, items, strict: bool = True):
        dec = getattr(self.decoder, "model", self.decoder)
        has_tok_emb = hasattr(dec, "tok_embeddings")
        has_embed_tokens = hasattr(dec, "embed_tokens")

        def remap(name: str) -> str:
            if name.startswith("model.language_model."):
                return "decoder.model." + name[len("model.language_model."):]
            if name.startswith("language_model.model."):
                return "decoder.model." + name[len("language_model.model."):]
            if name.startswith("language_model."):
                return "decoder." + name[len("language_model."):]
            if name.startswith("model.visual."):
                return "vision." + name[len("model.visual."):]
            if name.startswith("visual."):
                return "vision." + name[len("visual."):]
            if name.startswith("model.mm_projector."):
                return "projector." + name[len("model.mm_projector."):]
            if name.startswith("mm_projector."):
                return "projector." + name[len("mm_projector."):]
            return name

        def maybe_canonicalize(rk: str) -> str:
            if rk == "decoder.model.embed_tokens.weight" and has_tok_emb and not has_embed_tokens:
                return "decoder.model.tok_embeddings.weight"
            if rk.startswith("decoder.norm.") and hasattr(dec, "norm"):
                return rk.replace("decoder.norm.", "decoder.model.norm.")
            return rk

        # --- first pass: remap & find max indices to size lists correctly ---
        max_blk = -1
        max_ds  = -1
        remapped = []
        for k, v in items:
            rk = maybe_canonicalize(remap(k))
            remapped.append((rk, v))
            # track largest indices we will touch
            if rk.startswith("vision.blocks."):
                parts = rk.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    max_blk = max(max_blk, int(parts[2]))
            elif rk.startswith("vision.deepstack_merger_list."):
                parts = rk.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    max_ds = max(max_ds, int(parts[2]))

        if max_blk >= 0:
            self.vision.ensure_block_count(max_blk + 1)
        if max_ds >= 0:
            self.vision.ensure_deepstack_count(max_ds + 1)

        # --- second pass: filter out obvious non-owned params; let MLX ignore the rest (strict=False) ---
        filtered = []
        for rk, v in remapped:
            # toss known non-param blobs if any (rare)
            if rk.endswith(".wpe") or rk.endswith(".rope.freqs"):
                continue
            # keep all remapped params; MLX will skip unknowns when strict=False
            filtered.append((rk, v))

        return super().load_weights(filtered, strict=False)