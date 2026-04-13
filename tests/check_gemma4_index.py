#!/usr/bin/env python3
"""
Quick diagnostic for Gemma4 safetensors index consistency.

Usage:
  python tests/check_gemma4_index.py --model-path /path/to/merged-model

Paste the full output back to me, and I can compare it against official Gemma4 layout.
"""

import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Merged model directory")
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    data = json.loads(idx_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    keys = set(weight_map.keys())

    print(f"MODEL_DIR: {model_dir}")
    print(f"INDEX_FILE: {idx_path}")
    print(f"TOTAL_TENSORS: {len(keys)}")

    # Detect text layer ids from keys
    pat = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
    layers = sorted({int(m.group(1)) for k in keys for m in [pat.match(k)] if m})

    print(f"TEXT_LAYER_COUNT: {len(layers)}")
    print(f"TEXT_LAYER_MIN: {layers[0] if layers else -1}")
    print(f"TEXT_LAYER_MAX: {layers[-1] if layers else -1}")

    def missing_layers(name_template: str):
        miss = []
        for i in layers:
            k = name_template.format(i=i)
            if k not in keys:
                miss.append(i)
        return miss

    checks = {
        "q_proj": "model.language_model.layers.{i}.self_attn.q_proj.weight",
        "k_proj": "model.language_model.layers.{i}.self_attn.k_proj.weight",
        "v_proj": "model.language_model.layers.{i}.self_attn.v_proj.weight",
        "o_proj": "model.language_model.layers.{i}.self_attn.o_proj.weight",
        "q_norm": "model.language_model.layers.{i}.self_attn.q_norm.weight",
        "k_norm": "model.language_model.layers.{i}.self_attn.k_norm.weight",
        "layer_scalar": "model.language_model.layers.{i}.layer_scalar",
        "mlp_up": "model.language_model.layers.{i}.mlp.up_proj.weight",
        "mlp_gate": "model.language_model.layers.{i}.mlp.gate_proj.weight",
        "mlp_down": "model.language_model.layers.{i}.mlp.down_proj.weight",
    }

    print("=== TEXT_LAYER_PRESENCE ===")
    for name, tpl in checks.items():
        miss = missing_layers(tpl)
        print(f"{name}_MISSING_COUNT: {len(miss)}")
        print(f"{name}_MISSING_LAYERS: {miss}")

    # Naming style summary
    vision_linear = sum(
        1
        for k in keys
        if "model.vision_tower.encoder.layers." in k and ".linear.weight" in k
    )
    text_linear = sum(
        1
        for k in keys
        if "model.language_model.layers." in k and ".linear.weight" in k
    )
    print("=== NAMING_STYLE ===")
    print(f"VISION_LINEAR_WEIGHT_KEYS: {vision_linear}")
    print(f"TEXT_LINEAR_WEIGHT_KEYS: {text_linear}")

    print("=== EMBED_VISION ===")
    for k in [
        "model.embed_vision.embedding_projection.weight",
        "model.embed_vision.mm_soft_emb_norm.weight",
    ]:
        print(f"{k}: {'EXISTS' if k in keys else 'MISSING'}")

    # Useful heads/tails to spot broken sharding merges
    sorted_keys = sorted(keys)
    print("=== SAMPLE_KEYS_HEAD_20 ===")
    for k in sorted_keys[:20]:
        print(k)
    print("=== SAMPLE_KEYS_TAIL_20 ===")
    for k in sorted_keys[-20:]:
        print(k)


if __name__ == "__main__":
    main()
