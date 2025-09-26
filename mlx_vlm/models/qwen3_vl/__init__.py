from .config import Qwen3VLConfig
from .qwen3_vl import Qwen3VL
from .processor import load_processor

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