from __future__ import annotations
from typing import Any
from transformers import AutoProcessor  # HF processor handles resize/normalize & chat template

def load_processor(model_path_or_repo_id: str) -> Any:
    """
    Return the HF AutoProcessor for Qwen3-VL.
    Qwen3-VL expects images resized to multiples of 32 and patch_size 16; the
    official processor encodes this. If your pipeline wraps the processor, keep
    the API compatible with other mlx-vlm models (apply_chat_template, etc.).
    """
    proc = AutoProcessor.from_pretrained(model_path_or_repo_id, trust_remote_code=True)
    return proc