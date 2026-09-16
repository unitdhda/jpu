#!/usr/bin/env python3
"""Preflight and numerical smoke tests for the browser ONNX exports.

This validates the graph contract and executes both artifacts with ONNX Runtime
CPU. Actual WebGPU execution is covered by ``/webgpu-smoke.html`` in a browser
with a WebGPU adapter; CPU execution is intentionally only a portable preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch

from jp_lexer.features import CAT_FEATURE_NAMES, collate_records
from jp_lexer.model import JpLexer, ModelConfig

INPUTS = {
    "codepoints": ("tensor(int32)", ["batch", 128]),
    "bigrams": ("tensor(int32)", ["batch", 128]),
    "categorical": ("tensor(float)", ["batch", 128, len(CAT_FEATURE_NAMES)]),
    "mask": ("tensor(bool)", ["batch", 128]),
}
OUTPUTS = ("boundaries", "atom", "function", "inflection", "role")


def feed_for(record: Dict, max_length: int = 128) -> Dict[str, np.ndarray]:
    batch = collate_records([record], 8192, 8192)
    length = len(record["text"])
    if length > max_length:
        raise ValueError("test record exceeds fixed ONNX length")
    feed = {
        "codepoints": np.zeros((1, max_length), dtype=np.int32),
        "bigrams": np.zeros((1, max_length), dtype=np.int32),
        "categorical": np.zeros((1, max_length, len(CAT_FEATURE_NAMES)), dtype=np.float32),
        "mask": np.zeros((1, max_length), dtype=np.bool_),
    }
    for name in ("codepoints", "bigrams", "categorical"):
        feed[name][0, :length] = batch.inputs[name][0].numpy()
    feed["mask"][0, :length] = True
    return feed


def check_graph(path: Path) -> ort.InferenceSession:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual_inputs = {item.name: (item.type, item.shape) for item in session.get_inputs()}
    for name, (dtype, shape) in INPUTS.items():
        if actual_inputs.get(name) != (dtype, shape):
            raise AssertionError(f"{path.name}: {name} contract is {actual_inputs.get(name)!r}, expected {(dtype, shape)!r}")
    actual_outputs = tuple(item.name for item in session.get_outputs())
    if actual_outputs != OUTPUTS:
        raise AssertionError(f"{path.name}: outputs {actual_outputs!r}, expected {OUTPUTS!r}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/webgpu"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/phase3-small-150k-85-10-5.pt"))
    parser.add_argument("--record", type=Path, default=Path("artifacts/representative/canonical.jsonl"))
    args = parser.parse_args()

    paths = {
        "fp32": args.model_dir / "jpu-150k-fp32.onnx",
        "int8": args.model_dir / "jpu-150k-int8.onnx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing model(s): " + ", ".join(missing) + "\nrun scripts/export_webgpu.py first")
    with args.record.open(encoding="utf-8") as source:
        record = json.loads(next(source))
    feed = feed_for(record)

    sessions = {name: check_graph(path) for name, path in paths.items()}
    values = {name: sessions[name].run(None, feed) for name in paths}
    fp32, int8 = values["fp32"], values["int8"]
    quant_errors = [np.abs(a - b) for a, b in zip(fp32, int8)]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = JpLexer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch_feed = {
        "codepoints": torch.from_numpy(feed["codepoints"]),
        "bigrams": torch.from_numpy(feed["bigrams"]),
        "categorical": torch.from_numpy(feed["categorical"]),
        "mask": torch.from_numpy(feed["mask"]),
    }
    with torch.inference_mode():
        reference = model(torch_feed)
    torch_errors = [
        np.abs(values["fp32"][index] - reference[name].numpy())
        for index, name in enumerate(OUTPUTS)
    ]

    report = {
        "status": "PASS",
        "record_codepoints": len(record["text"]),
        "graphs": {name: {"bytes": path.stat().st_size, "provider": "CPU preflight"}
                   for name, path in paths.items()},
        "pytorch_fp32_max_abs_error": float(max(np.max(x) for x in torch_errors)),
        "int8_vs_fp32_max_abs_error": float(max(np.max(x) for x in quant_errors)),
        "int8_vs_fp32_mean_abs_error": float(np.mean(np.concatenate([x.reshape(-1) for x in quant_errors]))),
        "output_shapes": {name: list(array.shape) for name, array in zip(OUTPUTS, fp32)},
    }
    if not np.isfinite(np.concatenate([x.reshape(-1) for x in int8])).all():
        raise AssertionError("int8 output contains non-finite values")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
