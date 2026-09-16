#!/usr/bin/env python3
"""Export the jpu weights for the direct WebGPU runtime.

Weights are stored as two little-endian fp16 values per u32. The shader expands
those values to fp32 while computing, so the browser does not need ONNX Runtime
or a large serialized graph. The layout is explicit and versioned in the JSON
metadata file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jp_lexer.features import CAT_FEATURE_NAMES
from jp_lexer.model import JpLexer, ModelConfig

ORDER = [
    "codepoint_embedding.weight", "bigram_embedding.weight",
    "input_projection.weight", "input_projection.bias",
]
for i in range(5):
    ORDER += [
        f"convolution.{i}.depthwise.weight", f"convolution.{i}.depthwise.bias",
        f"convolution.{i}.pointwise.weight", f"convolution.{i}.pointwise.bias",
        f"convolution.{i}.norm.weight", f"convolution.{i}.norm.bias",
    ]
ORDER += [
    "scan.forward_project.weight", "scan.forward_project.bias",
    "scan.backward_project.weight", "scan.backward_project.bias",
    "scan.forward_decay", "scan.backward_decay",
    "boundary_head.weight", "boundary_head.bias",
    "atom_head.weight", "atom_head.bias",
    "function_head.weight", "function_head.bias",
    "inflection_head.weight", "inflection_head.bias",
    "role_head.weight", "role_head.bias",
]


def gzip_copy(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, target.open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=9, mtime=0) as output:
            output.write(source.read())
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("50k", "150k", "250k"),
                        help="public model name; inferred from the checkpoint config when omitted")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["config"])
    model = JpLexer(config)
    model.load_state_dict(payload["state_dict"])
    state = model.state_dict()
    model_sizes = {tuple(ModelConfig.for_size(size).__dict__.values()): size
                   for size in ("50k", "150k", "250k")}
    model_size = args.model_size or model_sizes.get(tuple(config.__dict__.values()))
    if model_size is None:
        raise SystemExit("cannot infer model size from checkpoint; pass --model-size")
    flat = []
    tensors = {}
    offset = 0
    for name in ORDER:
        values = state[name].detach().cpu().numpy().astype("<f4", copy=False).reshape(-1)
        tensors[name] = {"offset": offset, "length": int(values.size), "shape": list(state[name].shape)}
        flat.append(values)
        offset += values.size
    values = np.concatenate(flat).astype("<f2")
    if values.size % 2:
        values = np.concatenate([values, np.zeros(1, dtype="<f2")])
    half_bits = values.view("<u2")
    packed = (half_bits[0::2].astype("<u4") | (half_bits[1::2].astype("<u4") << 16))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if model_size is None:
        raise SystemExit("cannot infer model size from checkpoint; pass --model-size")
    expected = ModelConfig.for_size(model_size)
    if config != expected:
        raise SystemExit(f"checkpoint config does not match {model_size}; refuse to publish a mislabeled asset")
    binary = args.output_dir / f"jpu-{model_size}-custom.bin"
    binary.write_bytes(packed.tobytes())
    gzip_copy(binary)
    meta = {
        "schema": "jpu.custom-webgpu.v1",
        "model_size": model_size,
        "max_length": 128,
        "checkpoint": str(args.checkpoint),
        "weight_format": "two-little-endian-fp16-per-u32",
        "parameter_count": int(offset),
        "packed_words": int(packed.size),
        "hidden": model.config.hidden,
        "codepoint_buckets": model.config.codepoint_buckets,
        "bigram_buckets": model.config.bigram_buckets,
        "categorical_dim": len(CAT_FEATURE_NAMES),
        "dilations": list(model.config.dilations),
        "atom_classes": 12, "function_classes": 12,
        "inflection_classes": 10, "role_classes": 9, "boundary_classes": 5,
        "tensors": tensors,
    }
    meta_path = args.output_dir / f"jpu-{model_size}-custom.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    print(json.dumps({"binary": str(binary), "bytes": binary.stat().st_size,
                      "gzip_bytes": binary.with_suffix(binary.suffix + ".gz").stat().st_size,
                      "sha256": digest, "parameter_count": offset}, indent=2))


if __name__ == "__main__":
    main()
