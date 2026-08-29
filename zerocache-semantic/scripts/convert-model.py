"""One-time: fetch all-MiniLM-L6-v2 and write an f16 safetensors + tokenizer.json
+ config.json into ../model/. Committed to the repo so the crate builds offline.

    uv run --with safetensors --with numpy python convert-model.py
"""

import pathlib
import urllib.request

import numpy as np
from safetensors.numpy import load_file, save_file

BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
OUT = pathlib.Path(__file__).resolve().parent.parent / "model"
OUT.mkdir(exist_ok=True)

for f in ("model.safetensors", "tokenizer.json", "config.json"):
    print("fetch", f)
    urllib.request.urlretrieve(f"{BASE}/{f}", OUT / f)

tensors = {k: v.astype(np.float16) for k, v in load_file(OUT / "model.safetensors").items()}
save_file(tensors, str(OUT / "all-MiniLM-L6-v2.f16.safetensors"))
(OUT / "model.safetensors").unlink()

print("wrote", sorted(p.name for p in OUT.iterdir()))
