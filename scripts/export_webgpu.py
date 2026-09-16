#!/usr/bin/env python3
"""Export a fixed-length jpu checkpoint for ONNX Runtime WebGPU.

The model's affine scan is exported at a fixed max length so the traced loop is
portable to browsers. Inputs are padded to this length by the JS runtime.
This script emits FP32 and optional ONNX Runtime static-int8 artifacts plus
transport gzip files and a checksum/size report.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict

import numpy as np
import torch
from torch import nn

from jp_lexer.model import JpLexer, ModelConfig
from jp_lexer.features import CAT_FEATURE_NAMES

INPUT_NAMES = ("codepoints", "bigrams", "categorical", "mask")
OUTPUT_NAMES = ("boundaries", "atom", "function", "inflection", "role")


class ExportWrapper(nn.Module):
    def __init__(self, model: JpLexer):
        super().__init__(); self.model = model

    def forward(self, codepoints, bigrams, categorical, mask):
        outputs = self.model.forward_export({"codepoints": codepoints, "bigrams": bigrams,
                                             "categorical": categorical, "mask": mask})
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def gzip_copy(path: Path) -> Path:
    target=path.with_suffix(path.suffix + ".gz")
    # A fixed mtime keeps transport artifacts reproducible across exports.
    with path.open("rb") as source, target.open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=9, mtime=0) as out:
            shutil.copyfileobj(source, out)
    return target


def export_onnx(model: JpLexer, output: Path, max_length: int) -> None:
    wrapper=ExportWrapper(model).eval()
    # WebGPU has much better coverage for int32 than int64. Embedding accepts
    # int32 indices, so keep the browser contract in the provider-friendly
    # type instead of relying on a CPU cast/fallback.
    cp=torch.zeros((1, max_length), dtype=torch.int32)
    bg=torch.zeros((1, max_length), dtype=torch.int32)
    categorical=torch.zeros((1, max_length, len(CAT_FEATURE_NAMES)), dtype=torch.float32)
    mask=torch.ones((1, max_length), dtype=torch.bool)
    # The length is intentionally static: the Python diagonal scan becomes a
    # finite ONNX graph and JS pads shorter strings to the same contract.
    torch.onnx.export(wrapper, (cp, bg, categorical, mask), str(output),
                      input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                      dynamic_axes={name: {0: "batch"} for name in (*INPUT_NAMES, *OUTPUT_NAMES)},
                      opset_version=18, dynamo=False, do_constant_folding=True)


def _feed(record: Dict, max_length: int) -> Dict[str, np.ndarray]:
    from jp_lexer.features import collate_records
    import numpy as np
    if len(record["text"]) > max_length:
        raise ValueError("calibration example exceeds --max-length")
    batch=collate_records([record], 8192, 8192)
    feed={"codepoints":np.zeros((1,max_length),dtype=np.int32),
          "bigrams":np.zeros((1,max_length),dtype=np.int32),
          "categorical":np.zeros((1,max_length,len(CAT_FEATURE_NAMES)),dtype=np.float32),
          "mask":np.zeros((1,max_length),dtype=np.bool_)}
    length=len(record["text"])
    for key in feed: feed[key][0,:length]=batch.inputs[key][0].numpy()
    feed["mask"][0,:length]=True
    return feed


def quantize_onnx(source: Path, target: Path, calibration_data: Path,
                  max_length: int, limit: int) -> Dict[str, float]:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    rows=[]
    with calibration_data.open(encoding="utf-8") as source_file:
        for line in source_file:
            record = json.loads(line)
            # Length-bucket exports must calibrate only on examples that fit
            # the exported graph. Keep scanning rather than failing on the
            # first long record.
            if len(record["text"]) <= max_length:
                rows.append(record)
            if len(rows) >= limit: break
    if not rows: raise ValueError("calibration data has no examples for this max length")

    class Reader(CalibrationDataReader):
        def __init__(self): self.rows=iter(rows)
        def get_next(self):
            try: return _feed(next(self.rows), max_length)
            except StopIteration: return None

    # Static QDQ calibration is used instead of dynamic ConvInteger. QDQ keeps
    # convolution operators in a provider-compatible form and calibrates the
    # long-range scan activations on representative text.
    quantize_static(str(source), str(target), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "MatMul"],
                    extra_options={"ForceQuantizeNoInputCheck": True})
    return {"calibration_examples": len(rows)}


def compare_onnx(source: Path, quantized: Path, feed: Dict[str, np.ndarray]) -> Dict[str, float]:
    import onnxruntime as ort
    reference=ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feed)
    candidate=ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"]).run(None, feed)
    errors=[np.abs(left-right) for left,right in zip(reference,candidate)]
    return {"max_abs_logit_error":float(max(np.max(x) for x in errors)),
            "mean_abs_logit_error":float(np.mean(np.concatenate([x.reshape(-1) for x in errors])))}


def artifact(path: Path) -> Dict[str, object]:
    compressed=path.with_suffix(path.suffix + ".gz")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
            "gzip_path": str(compressed), "gzip_bytes": compressed.stat().st_size,
            "gzip_sha256": sha256(compressed)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--suffix", default="", help="filename suffix for length-bucket exports")
    parser.add_argument("--calibration-data", type=Path,
                        help="canonical JSONL used for static int8 activation calibration")
    parser.add_argument("--calibration-examples", type=int, default=256)
    parser.add_argument("--no-quantize", action="store_true")
    args=parser.parse_args()
    if args.max_length < 1: raise SystemExit("--max-length must be positive")
    payload=torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model=JpLexer(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"]); model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32=args.output_dir / f"jpu-150k-fp32{args.suffix}.onnx"
    export_onnx(model, fp32, args.max_length)
    files=[fp32]
    gzip_copy(fp32)
    int8=None; quantization_report=None
    if not args.no_quantize:
        if args.calibration_data is None:
            raise SystemExit("--calibration-data is required for static int8 export")
        int8=args.output_dir / f"jpu-150k-int8{args.suffix}.onnx"
        quantization_report=quantize_onnx(fp32, int8, args.calibration_data,
                                          args.max_length, args.calibration_examples)
        files.append(int8); gzip_copy(int8)
        # Verify the quantized graph on the first calibration example before
        # advertising it to the browser runtime.
        with args.calibration_data.open(encoding="utf-8") as source_file:
            first=json.loads(next(source_file))
        quantization_report.update(compare_onnx(fp32, int8, _feed(first, args.max_length)))
    report={"schema":"jpu.webgpu-export.v1", "checkpoint":str(args.checkpoint),
            "max_length":args.max_length, "input_names":INPUT_NAMES,
            "output_names":OUTPUT_NAMES, "parameter_report":model.parameter_report(),
            "artifacts":[artifact(path) for path in files],
            "quantization":{"method":"onnxruntime.quantization.quantize_static",
                             "format":"QDQ", "weight_type":"QInt8",
                             "activation_type":"QUInt8", "per_channel":True,
                             "op_types":["Conv", "MatMul"], "conv_kernels":"QDQ int8",
                             **(quantization_report or {})} if int8 else None,
            "note":"gzip files are distribution artifacts; serve ONNX with HTTP compression or fetch/decompress gzip explicitly."}
    report_path=args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=list)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__": main()
