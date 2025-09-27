from __future__ import annotations
from typing import Dict, Iterable, Tuple

# Map HF (name, tensor) → MLX (name, tensor) pairs
def map_hf_to_mlx_keys(hf_items: Iterable[Tuple[str, object]]) -> Dict[str, object]:
    """
    Minimal, explicit key mapping for Qwen3-VL:
      - keep BOTH vision + text
      - vision: model.visual.* → vision.*
      - text:   model.language_model.* → decoder.model.*
                model.embed_tokens.*   → decoder.model.embed_tokens.*
                lm_head.weight         → decoder.lm_head.weight
      - projector: model.mm_projector* / model.projector.* → projector.*
      - deepstack/merger are kept under vision.*
    """
    out: Dict[str, object] = {}

    def put(k: str, v):
        out[k] = v

    for k, v in hf_items:
        # ---- vision ----
        if k.startswith("model.visual."):
            put("vision." + k[len("model.visual."):], v)
            continue

        # Some repos use multi_modal_projector / mm_projector / projector
        if k.startswith("model.mm_projector") or k.startswith("model.multi_modal_projector"):
            suffix = k.split(".", 1)[1]  # after "model."
            put("projector." + suffix.split(".", 1)[1], v)
            continue
        if k.startswith("model.projector."):
            put("projector." + k[len("model.projector."):], v)
            continue

        # ---- text ----
        if k.startswith("model.language_model."):
            put("decoder.model." + k[len("model.language_model."):], v)
            continue
        if k.startswith("model.embed_tokens."):
            put("decoder.model." + k[len("model."):], v)
            continue
        if k == "lm_head.weight":
            put("decoder.lm_head.weight", v)
            continue

        # fallback: keep other model.* params in decoder.*
        if k.startswith("model."):
            put("decoder." + k[len("model."):], v)
            continue

        # else: ignore optimizer states etc.

    return out