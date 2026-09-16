#!/usr/bin/env python3
"""Benchmark training-step configurations on Apple MPS before a long run.

The benchmark times only an already-collated, already-device-resident
forward/backward/optimizer step. It does not hide data loading or MPS startup;
those are reported separately. Use this to select a conservative batch size
for the production training job.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from jp_lexer.features import collate_records
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer


def load_records(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def sync() -> None:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def step(model: JpLexer, optimizer: torch.optim.Optimizer, inputs: dict,
         targets: dict) -> float:
    optimizer.zero_grad(set_to_none=True)
    losses = multitask_loss(model(inputs), targets)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(losses["total"].detach().cpu())


def benchmark_one(rows: list[dict], size: str, batch_size: int,
                  warmup: int, iterations: int, device: torch.device) -> dict:
    model = JpLexer.from_size(size).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    group = [rows[i % len(rows)] for i in range(batch_size)]
    batch = collate_records(group, model.config.codepoint_buckets, model.config.bigram_buckets)
    inputs = {key: value.to(device) for key, value in batch.inputs.items()}
    targets = {key: value.to(device) for key, value in batch.targets.items()}
    sync()
    warm_start = time.perf_counter()
    for _ in range(warmup):
        step(model, optimizer, inputs, targets)
    sync()
    warmup_ms = (time.perf_counter() - warm_start) * 1000
    times = []
    losses = []
    for _ in range(iterations):
        start = time.perf_counter()
        losses.append(step(model, optimizer, inputs, targets))
        sync()
        times.append((time.perf_counter() - start) * 1000)
    allocated = int(torch.mps.current_allocated_memory()) if hasattr(torch, "mps") else None
    result = {
        "size": size, "batch_size": batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sequence_codepoints": int(inputs["mask"].shape[1]),
        "warmup_ms": warmup_ms,
        "step_ms": {"mean": statistics.mean(times), "median": statistics.median(times),
                    "min": min(times), "max": max(times)},
        "sentences_per_second": batch_size / (statistics.mean(times) / 1000),
        "codepoints_per_second": batch_size * int(inputs["mask"].shape[1]) / (statistics.mean(times) / 1000),
        "last_loss": losses[-1], "mps_allocated_bytes": allocated,
    }
    del model, optimizer, inputs, targets, batch
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument("--sizes", nargs="+", default=["50k", "150k", "250k"],
                        choices=["50k", "100k", "150k", "250k"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    rows = load_records(args.data, args.records)
    if not rows:
        raise SystemExit("no canonical records")
    device = torch.device("mps")
    results = []
    for size in args.sizes:
        for batch_size in args.batch_sizes:
            print(f"benchmarking size={size} batch={batch_size}", flush=True)
            results.append(benchmark_one(rows, size, batch_size, args.warmup, args.iterations, device))
    report = {"device": str(device), "torch": torch.__version__, "records": len(rows),
              "warmup_steps": args.warmup, "measured_steps": args.iterations,
              "results": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
