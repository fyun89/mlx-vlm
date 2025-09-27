from .config import Qwen3VLConfig, TextConfig, VisionConfig
from .qwen3_vl import Qwen3VL
from .processor import load_processor
from .vision import Qwen3Vision
from mlx_lm.models import qwen3_moe
import mlx.nn as nn

ModelConfig = Qwen3VLConfig
Model = Qwen3VL
TextConfig = TextConfig
VisionConfig = VisionConfig
VisionModel = Qwen3Vision
LanguageModelArgs = qwen3_moe.ModelArgs

class LanguageModel(qwen3_moe.Model):
    def __init__(self, text_cfg: TextConfig | dict):
        print("[q3vl] LanguageModel wrapper active")
        # merge fields from text_cfg and its raw top-level config (if present)
        tcfg = dict(text_cfg)
        raw = getattr(text_cfg, "raw", None) or {}

        required = {
            "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "num_experts", "num_experts_per_tok",
            "decoder_sparse_step", "mlp_only_layers", "moe_intermediate_size",
            "rms_norm_eps", "vocab_size", "num_key_value_heads", "head_dim",
            "rope_theta", "max_position_embeddings", "norm_topk_prob",
        }
        for k in required:
            if k not in tcfg and k in raw:
                tcfg[k] = raw[k]
        raw_text = (raw.get("text_config") or {}) if isinstance(raw, dict) else {}
        for k in required:
            if k not in tcfg and k in raw_text:
                tcfg[k] = raw_text[k]

        # safe defaults often missing
        tcfg.setdefault("tie_word_embeddings", False)
        tcfg.setdefault("mlp_only_layers", [])

        args = qwen3_moe.ModelArgs.from_dict(tcfg)
        super().__init__(args)

    def load_weights(self, items, strict=True):
        print("[q3vl] LanguageModel.load_weights override active:", len(items))
        """
        items: list[(name, array)] from HF. We:
          - strip 'model.language_model.' -> 'model.'
          - drop clearly multimodal/vision keys
          - drop heads/norm that don't exist on the bare text model in sanitize phase
          - keep only keys that match this module's expected parameter names
        """
        # All expected param names for this text backbone:
        expected = set(k for k, _ in self.named_parameters())

        def remap(name: str) -> str:
            # HF text path contains 'model.language_model.' while mlx expects 'model.'
            if name.startswith("model.language_model."):
                return "model." + name[len("model.language_model."):]
            return name

        filtered = []
        for k, v in items:
            rk = remap(k)

            # Drop multimodal / projector / mergers / vision bits at sanitize time
            if (
                ".visual." in rk
                or ".mm_projector." in rk
                or ".projector." in rk
                or ".deepstack_merger_list." in rk
                or ".merger." in rk
                or ".image_" in rk
                or ".vision_" in rk
            ):
                continue

            # The sanitize LanguageModel instance typically doesn't expose lm_head/norm
            # If your qwen3_moe.Model does include them, they will survive via 'expected'
            if rk.startswith("lm_head.") or ".norm." in rk:
                # Defer these to the full multimodal model load
                continue

            # Keep only keys that the text backbone actually has
            if rk in expected:
                filtered.append((rk, v))

        # Now delegate to base loader
        return super().load_weights(filtered, strict=strict)

def load(model_path_or_repo_id: str, cfg: dict | None = None, **kwargs):
    """
    Factory used by the top-level mlx_vlm.load() dispatcher.
    - cfg is an already-fetched HF config dict (preferred).
    - kwargs can pass through generation params if your loader expects them.
    Returns (model, processor).
    """
    if cfg is None:
        # defer to top-level utils that already fetch configs in your repo
        raise ValueError("Qwen3-VL loader requires `cfg` (HF config.json as dict)")

    qcfg = Qwen3VLConfig.from_hf_config(cfg)
    model = Qwen3VL(qcfg)
    processor = load_processor(model_path_or_repo_id)
    return model, processor
