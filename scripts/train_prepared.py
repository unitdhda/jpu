#!/usr/bin/env python3
"""Train a model from fixed-length memory-mapped arrays."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer

FIELDS = ("codepoints", "bigrams", "categorical", "mask", "boundaries", "atom", "function", "function_mask", "inflection", "inflection_mask", "role")

def batch_refs(root, split, batch_size, seed):
    meta = json.loads((root / "metadata.json").read_text())
    refs = []
    for bucket in (32, 64, 128):
        count = meta["counts"].get(f"{split}-{bucket}", 0)
        refs.extend((bucket, start) for start in range(0, count - batch_size + 1, batch_size))
    # Keep each epoch mostly shape-homogeneous. MPSGraph can otherwise
    # repeatedly specialize kernels while 32/64/128-codepoint batches are
    # randomly interleaved.
    groups = []
    rng = random.Random(seed)
    for bucket in (32, 64, 128):
        group = [ref for ref in refs if ref[0] == bucket]
        rng.shuffle(group); groups.append(group)
    rng.shuffle(groups)
    return [ref for group in groups for ref in group]

def load_batch(stores, bucket, start, batch_size):
    arrays = {field: array[start:start + batch_size] for field, array in stores[bucket].items()}
    inputs = {"codepoints": torch.from_numpy(arrays["codepoints"].astype(np.int64, copy=True)), "bigrams": torch.from_numpy(arrays["bigrams"].astype(np.int64, copy=True)), "categorical": torch.from_numpy(arrays["categorical"].astype(np.float32, copy=True)), "mask": torch.from_numpy(arrays["mask"].astype(bool, copy=True))}
    targets = {"boundaries": torch.from_numpy(arrays["boundaries"].astype(np.float32, copy=True)), "atom": torch.from_numpy(arrays["atom"].astype(np.int64, copy=True)), "function": torch.from_numpy(arrays["function"].astype(np.int64, copy=True)), "function_mask": torch.from_numpy(arrays["function_mask"].astype(bool, copy=True)), "inflection": torch.from_numpy(arrays["inflection"].astype(np.float32, copy=True)), "inflection_mask": torch.from_numpy(arrays["inflection_mask"].astype(bool, copy=True)), "role": torch.from_numpy(arrays["role"].astype(np.int64, copy=True)), "mask": inputs["mask"]}
    return inputs, targets

def move(values, device): return {key: value.to(device) for key, value in values.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data", type=Path); ap.add_argument("--model", choices=("50k", "100k", "150k", "250k"), required=True); ap.add_argument("--batch-size", type=int, required=True); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="mps"); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available(): raise SystemExit("MPS is not available")
    device = torch.device(args.device); root = args.data / args.model
    stores = {bucket: {field: np.load(root / f"train-{bucket}-{field}.npy", mmap_mode="r") for field in FIELDS} for bucket in (32, 64, 128) if (root / f"train-{bucket}-mask.npy").exists()}
    model = JpLexer.from_size(args.model).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); history = []
    for epoch in range(args.epochs):
        model.train(); refs = batch_refs(root, "train", args.batch_size, args.seed + epoch); start = time.perf_counter(); total = 0.0
        for step, (bucket, row) in enumerate(refs, 1):
            inputs, targets = load_batch(stores, bucket, row, args.batch_size); inputs, targets = move(inputs, device), move(targets, device)
            optimizer.zero_grad(set_to_none=True); losses = multitask_loss(model(inputs), targets); losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step % 100 == 0: total += float(losses["total"].detach().cpu())
            if step % 1000 == 0: print(json.dumps({"model": args.model, "epoch": epoch + 1, "step": step, "steps": len(refs)}), flush=True)
        if device.type == "mps": torch.mps.synchronize()
        row = {"epoch": epoch + 1, "steps": len(refs), "sampled_loss": total / max(1, len(refs) // 100), "seconds": time.perf_counter() - start}; history.append(row); print(json.dumps(row), flush=True)
    result = {"model": args.model, "batch_size": args.batch_size, "epochs": args.epochs, "device": str(device), "history": history, "parameter_report": model.parameter_report()}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save({"config": model.config.__dict__, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, args.output.with_suffix(".pt")); args.output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__": main()
