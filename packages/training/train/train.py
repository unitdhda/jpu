#!/usr/bin/env python3
"""Small training and deterministic overfit harness for the Phase 2/3 bridge.

Splits are assigned by source/document, never by sentence. The full metrics and
baseline suite remain Phase 3 work; this script records loss and split counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping, Sequence

import torch

from jp_lexer.features import collate_records
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer


def load_records(path: Path, limit: int | None = None) -> List[dict]:
    records=[]
    with path.open(encoding="utf-8") as source:
        for line in source:
            records.append(json.loads(line))
            if limit and len(records) >= limit: break
    return records


def document_split(record: Mapping[str, str]) -> str:
    """Stable 85/10/5 split bucket for one source document."""
    key = str(record["source"]) + "\0" + str(record["document_id"])
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") % 100
    return "train" if bucket < 85 else "dev" if bucket < 95 else "test"


def split_records(records: Sequence[dict]) -> Dict[str, List[dict]]:
    result={"train": [], "dev": [], "test": []}
    seen={}
    for record in records:
        key=(record["source"], record["document_id"])
        split=document_split(record)
        if key in seen and seen[key] != split:
            raise ValueError("document assigned to multiple splits: %r" % (key,))
        seen[key]=split
        result[split].append(record)
    return result


def _move(values: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def _batches(records: Sequence[dict], batch_size: int, buckets: int,
             shuffle: bool, seed: int) -> List[List[dict]]:
    order=list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [[records[i] for i in order[start:start + batch_size]]
            for start in range(0, len(order), batch_size)]


def _evaluate(model: JpLexer, records: Sequence[dict], batch_size: int,
              device: torch.device) -> float:
    model.eval(); total=0.0; count=0
    with torch.inference_mode():
        for rows in _batches(records, batch_size, model.config.codepoint_buckets, False, 0):
            batch=collate_records(rows, model.config.codepoint_buckets, model.config.bigram_buckets)
            losses=multitask_loss(model(_move(batch.inputs, device)), _move(batch.targets, device))
            total += float(losses["total"])
            count += 1
    return total / max(1, count)


def run_training(records: Sequence[dict], size: str = "150k", epochs: int = 5,
                 batch_size: int = 64, seed: int = 7, device_name: str = "mps") -> dict:
    """Train on the document-level train split and report dev/test loss."""
    splits=split_records(records)
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    device=torch.device(device_name)
    torch.manual_seed(seed)
    model=JpLexer.from_size(size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history=[]
    for epoch in range(epochs):
        model.train(); total=0.0; batches=0
        for rows in _batches(splits["train"], batch_size, model.config.codepoint_buckets, True, seed + epoch):
            batch=collate_records(rows, model.config.codepoint_buckets, model.config.bigram_buckets)
            losses=multitask_loss(model(_move(batch.inputs, device)), _move(batch.targets, device))
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(losses["total"].detach()); batches += 1
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, batches),
                        "dev_loss": _evaluate(model, splits["dev"], batch_size, device)})
    result={"mode":"train", "size":size, "device":device_name, "epochs":epochs,
            "batch_size":batch_size, "seed":seed,
            "split_records":{key:len(value) for key,value in splits.items()},
            "split_documents":{key:len({(r["source"],r["document_id"]) for r in value})
                               for key,value in splits.items()},
            "history":history, "test_loss":_evaluate(model, splits["test"], batch_size, device),
            "parameter_report":model.parameter_report()}
    # Save CPU tensors so the artifact is portable beyond this MPS host.
    result["checkpoint"]={"config":model.config.__dict__, "state_dict":{
        key:value.detach().cpu() for key,value in model.state_dict().items()}}
    return result


def run_overfit(records: List[dict], size: str = "150k", steps: int = 200,
                batch_size: int = 16, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=JpLexer.from_size(size)
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train(); history=[]
    for step in range(steps):
        batch_records=[records[i % len(records)] for i in range(step * batch_size, (step + 1) * batch_size)]
        batch=collate_records(batch_records, model.config.codepoint_buckets, model.config.bigram_buckets)
        losses=multitask_loss(model(batch.inputs), batch.targets)
        optimizer.zero_grad(set_to_none=True); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step":step + 1, "loss":float(losses["total"].detach())})
    state={key:value.detach().clone() for key,value in model.state_dict().items()}
    restored=JpLexer(model.config); restored.load_state_dict(state)
    return {"mode":"overfit", "size":size, "examples":len(records), "steps":steps,
            "initial_loss":history[0]["loss"], "final_loss":history[-1]["loss"],
            "history":history, "parameter_report":model.parameter_report()}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--mode", choices=("train", "overfit"), default="overfit")
    parser.add_argument("--examples", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", choices=("50k", "100k", "150k", "250k"), default="150k")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    records=load_records(args.data, args.examples)
    if not records: raise SystemExit("no records")
    if args.mode == "train":
        result=run_training(records, args.size, args.epochs, args.batch_size, device_name=args.device)
        checkpoint_path=args.output.with_suffix(".pt") if args.output else None
        if checkpoint_path:
            torch.save(result.pop("checkpoint"), checkpoint_path)
            result["checkpoint_path"]=str(checkpoint_path)
    else:
        result=run_overfit(records, args.size, args.steps, args.batch_size)
    rendered=json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output: args.output.write_text(rendered+"\n", encoding="utf-8")


if __name__ == "__main__": main()
