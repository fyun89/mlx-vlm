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
from mlx_vlm.utils import skip_multimodal_module

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
        self.tdim = int(cfg.text_config.get("hidden_size", 4096))
        self.projector = nn.Linear(vdim, self.tdim, bias=True)
        print(f"WHETHER MOE ----> {cfg.model_type}")

        # ---- decoder ----
        if "moe" in cfg.model_type:
            targs = qwen3_moe.ModelArgs.from_dict(cfg.text_config)
            self.decoder = qwen3_moe.Model(targs)
        else:
            targs = qwen3_dense.ModelArgs.from_dict(cfg.text_config)
            self.decoder = qwen3_dense.Model(targs)

        # ----- TIE LM HEAD TO EMBEDDINGS -----
        dec  = self.decoder
        core = getattr(dec, "model", dec)
        vocab  = int(cfg.text_config.get("vocab_size"))
        hidden = int(cfg.text_config.get("hidden_size"))

        # locate embeddings
        emb = getattr(core, "embed_tokens", None) or getattr(core, "tok_embeddings", None)
        if emb is None:
            raise RuntimeError("Cannot locate token embedding module on Qwen3 decoder.")

        # ensure a head exists and is correctly shaped
        need_head = (not hasattr(dec, "lm_head"))
        if not need_head:
            try:
                need_head = tuple(dec.lm_head.weight.shape) != (vocab, hidden)
            except Exception:
                need_head = True
        if need_head:
            dec.lm_head = nn.Linear(hidden, vocab, bias=False)

        # TIE: use the SAME Parameter object
        dec.lm_head.weight = emb.weight

        # mirror to core so both access paths work
        if hasattr(dec, "model"):
            dec.model.lm_head = dec.lm_head

        # sanity prints AFTER the tie
        print("tied?", dec.lm_head.weight is emb.weight)
        print("embed class:", type(emb).__name__)
        print("head  class:", type(dec.lm_head).__name__)

        # ---- token ids ----
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id
        self.vision_token_id = cfg.vision_token_id

        # ---- optional decoder-only quantization (do NOT quantize tied tensors) ----
        quant = {}
        raw_cfg = getattr(cfg, "raw_config", None) or {}
        if isinstance(raw_cfg, dict):
            quant = raw_cfg.get("quantization") or {}
        if not quant and isinstance(cfg.text_config, dict):
            quant = cfg.text_config.get("quantization") or {}

        if quant:
            q_bits = int(quant.get("bits", 4))
            q_group_size = int(quant.get("group_size", 64))

            def _pred(path, module, *_):
                # Skip the tied head and embeddings
                if ("lm_head" in path or
                    ".embed_tokens" in path or
                    ".tok_embeddings" in path):
                    return False
                if not hasattr(module, "to_quantized"):
                    return False
                if skip_multimodal_module(path):
                    return False
                if not (path.startswith("decoder.") or ".decoder." in path):
                    return False
                try:
                    return (module.weight.shape[1] % q_group_size == 0) and {
                        "group_size": q_group_size,
                        "bits": q_bits,
                    }
                except Exception:
                    return False

            nn.quantize(self, class_predicate=_pred)

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
            if input_embeddings is not None:
                return self.decoder(None, input_embeddings=input_embeddings, mask=attention_mask, cache=cache)
            return self.decoder(input_ids, mask=attention_mask, cache=cache)

    @property
    def layers(self):
        return self.decoder.model.layers
    
    def ensure_block_count(self, n: int):
        """Grow vision.blocks to length n."""
        cur = len(self.blocks)
        if n <= cur:
            return
        dim   = self.norm.normalized_shape[0]          # same hidden dim
        heads = getattr(self, "_num_heads", None) or 16
        mlp_r = getattr(self, "_mlp_ratio", None) or 4.0
        qkv_b = True
        # remember some init values on first call
        self._num_heads  = heads
        self._mlp_ratio  = mlp_r
        for _ in range(cur, n):
            self.blocks.append(Block(dim, heads, mlp_r, qkv_b))

    def ensure_deepstack_count(self, n: int):
        """Grow vision.deepstack_merger_list to length n."""
        cur = len(self.deepstack_merger_list)
        if n <= cur:
            return
        dim = self.norm.normalized_shape[0]
        for _ in range(cur, n):
            self.deepstack_merger_list.append(DeepStackMerger(dim))

    # ---------------- weight loading (key remap + head-shape guard) ----------------
    def load_weights(self, items, strict: bool = True):
        dec = getattr(self.decoder, "model", self.decoder)
        has_tok_emb = hasattr(dec, "tok_embeddings")
        has_embed_tokens = hasattr(dec, "embed_tokens")

        # --- remap as you already do ---
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
            if name.startswith("lm_head."):
                return "SKIP.LM_HEAD"
            return name

        def maybe_canonicalize(rk: str) -> str:
            if rk == "decoder.model.embed_tokens.weight" and has_tok_emb and not has_embed_tokens:
                return "decoder.model.tok_embeddings.weight"
            if rk.startswith("decoder.norm.") and hasattr(dec, "norm"):
                return rk.replace("decoder.norm.", "decoder.model.norm.")
            return rk

        # --- pass 1: remap and find maximum vision indices we must support
        remapped = []
        max_blk  = -1
        max_ds   = -1
        for k, v in items:
            rk = maybe_canonicalize(remap(k))
            if rk == "SKIP.LM_HEAD":
                continue
            remapped.append((rk, v))
            if rk.startswith("vision.blocks."):
                parts = rk.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    max_blk = max(max_blk, int(parts[2]))
            elif rk.startswith("vision.deepstack_merger_list."):
                parts = rk.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    max_ds = max(max_ds, int(parts[2]))

        # --- grow vision lists *before* any update so indices exist
        if max_blk >= 0:
            self.vision.ensure_block_count(max_blk + 1)
        if max_ds >= 0:
            self.vision.ensure_deepstack_count(max_ds + 1)

        # --- pass 2: shape-guard for quantized tensors on attention proj (your recent filter)
        H  = int(self.cfg.text_config.get("hidden_size", 4096))
        KV = int(self.cfg.text_config.get("num_key_value_heads", 4)) * int(self.cfg.text_config.get("head_dim", 128))
        GS = 64

        def is_qw(n): return n.endswith(".qweight")
        def is_sc(n): return n.endswith(".scales")
        def is_ze(n): return n.endswith(".zeros")
        def expect_ok(name: str, arr) -> bool:
            shape = tuple(getattr(arr, "shape", ()))
            if is_qw(name):
                if ".self_attn.q_proj." in name: return shape == (H,  H // 8)
                if ".self_attn.k_proj." in name: return shape == (KV, H // 8)
                if ".self_attn.v_proj." in name: return shape == (KV, H // 8)
            if is_sc(name) or is_ze(name):
                if ".self_attn.q_proj." in name: return shape == (H,  H // GS)
                if ".self_attn.k_proj." in name: return shape == (KV, H // GS)
                if ".self_attn.v_proj." in name: return shape == (KV, H // GS)
            return True

        filtered = []
        dropped  = 0
        for rk, v in remapped:
            if rk.endswith(".wpe") or rk.endswith(".rope.freqs"):
                continue
            if (".self_attn." in rk) and (is_qw(rk) or is_sc(rk) or is_ze(rk)):
                if not expect_ok(rk, v):
                    dropped += 1
                    continue
            if rk.startswith("decoder.lm_head."):
                continue
            filtered.append((rk, v))

        if dropped:
            print(f"[q3vl] dropped {dropped} mismatched quant tensors (bad shapes)")

        # --- optional: quantize only where qweight exists to avoid double-quant
        qprefix = {rk[:-9] for rk, _ in filtered if rk.endswith(".qweight")}
        def _quant_pred(path, module, *_):
            if ("lm_head" in path) or (".embed_tokens" in path) or (".tok_embeddings" in path):
                return False
            return isinstance(module, nn.Linear) and (path in qprefix)

        if qprefix:
            nn.quantize(self, class_predicate=_quant_pred)

        return super().load_weights(filtered, strict=False)