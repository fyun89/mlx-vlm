from huggingface_hub import snapshot_download
import json, os
# from mlx_vlm.utils import detect_arch_from_cfg
from mlx_vlm.models.qwen3_vl import load as load_q3

repo = "Qwen/Qwen3-VL-235B-A22B-Instruct"
path = snapshot_download(repo, allow_patterns=["config.json"])
cfg = json.load(open(os.path.join(path, "config.json")))
# print("arch:", detect_arch_from_cfg(cfg))
m, p = load_q3(repo, cfg=cfg)
print("ok, built:", type(m))