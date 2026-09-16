#!/usr/bin/env python3
"""Memory-bounded production training over canonical JSONL.

The 2M-sentence corpus is too large to materialize on this 8 GB host. This
trainer keeps the document-level 85/10/5 split deterministic and streams one
split from disk with a bounded shuffle buffer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterator, Mapping

import torch

from jp_lexer.features import collate_records
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer


def document_split(record: Mapping[str, str]) -> str:
    key = str(record["source"]) + "\0" + str(record["document_id"])
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") % 100
    return "train" if bucket < 85 else "dev" if bucket < 95 else "test"


def iter_records(path: Path, split: str | None = None) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if split is None or document_split(record) == split:
                yield record


def shuffled_records(path: Path, split: str, buffer_size: int, seed: int) -> Iterator[dict]:
    rng = random.Random(seed)
    buffer: list[dict] = []
    for record in iter_records(path, split):
        if len(buffer) < buffer_size:
            buffer.append(record)
        else:
            index = rng.randrange(len(buffer)); buffer[index], record = record, buffer[index]
            yield record
    while buffer:
        index = rng.randrange(len(buffer)); yield buffer.pop(index)


def count_splits(path: Path) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "test": 0}
    for record in iter_records(path): counts[document_split(record)] += 1
    return counts


def move(values: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def train_one(path: Path, size: str, batch_size: int, epochs: int, seed: int,
              device: torch.device, shuffle_buffer: int, eval_limit: int) -> dict:
    torch.manual_seed(seed)
    model = JpLexer.from_size(size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train(); batch_rows: list[dict] = []; batches = 0; loss_sum = 0.0; measured = 0
        start = time.perf_counter()
        for record in shuffled_records(path, "train", shuffle_buffer, seed + epoch):
            batch_rows.append(record)
            if len(batch_rows) < batch_size: continue
            batch = collate_records(batch_rows, model.config.codepoint_buckets, model.config.bigram_buckets)
            inputs, targets = move(batch.inputs, device), move(batch.targets, device)
            optimizer.zero_grad(set_to_none=True)
            losses = multitask_loss(model(inputs), targets); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            batches += 1; batch_rows = []
            if batches % 100 == 0:
                loss_sum += float(losses["total"].detach().cpu()); measured += 1
        if batch_rows:
            batch = collate_records(batch_rows, model.config.codepoint_buckets, model.config.bigram_buckets)
            inputs, targets = move(batch.inputs, device), move(batch.targets, device)
            optimizer.zero_grad(set_to_none=True); losses = multitask_loss(model(inputs), targets)
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); batches += 1
            loss_sum += float(losses["total"].detach().cpu()); measured += 1
        if device.type == "mps": torch.mps.synchronize()
        history.append({"epoch": epoch + 1, "train_loss_sampled": loss_sum / max(1, measured),
                        "batches": batches, "seconds": time.perf_counter() - start})
        print(json.dumps({"size": size, **history[-1]}), flush=True)

    result = {"size": size, "batch_size": batch_size, "epochs": epochs, "seed": seed,
              "parameter_report": model.parameter_report(), "history": history,
              "split_counts": count_splits(path)}
    result["checkpoint"] = {"config": model.config.__dict__,
                             "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--sizes", nargs="+", default=["50k", "150k", "250k"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 64, 64])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.sizes) != len(args.batch_sizes): raise SystemExit("sizes and batch-sizes must have equal length")
    if args.device == "mps" and not torch.backends.mps.is_available(): raise SystemExit("MPS is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = count_splits(args.data); print(json.dumps({"data": str(args.data), "split_counts": counts}), flush=True)
    device = torch.device(args.device)
    for size, batch_size in zip(args.sizes, args.batch_sizes):
        result = train_one(args.data, size, batch_size, args.epochs, args.seed, device, args.shuffle_buffer, 0)
        checkpoint_path = args.output_dir / f"jpu-{size}-85-10-5.pt"
        torch.save(result.pop("checkpoint"), checkpoint_path)
        result["checkpoint_path"] = str(checkpoint_path)
        (args.output_dir / f"jpu-{size}-85-10-5.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if device.type == "mps": torch.mps.empty_cache()


if __name__ == "__main__": main()
