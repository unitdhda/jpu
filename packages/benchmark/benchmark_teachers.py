#!/usr/bin/env python3
"""Benchmark jpu against its Sudachi and GiNZA teacher pipeline.

The benchmark always evaluates on independent gold JSONL. Sudachi-only and
GiNZA-only rows report only the fields those analyzers provide; the combined
teacher row is the comparable full surface task.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from data.annotation.mappings import map_bunsetsu_role, map_inflections, map_particle, map_sudachi_pos
from data.schemas.canonical import LEVELS
from .metrics import score_corpus
from jp_lexer.decoder import decode_logits
from jp_lexer.features import collate_records
from jp_lexer.model import JpLexer, ModelConfig


def _sudachi_prediction(text: str, sudachi: Any, tokenizer: Any) -> dict[str, Any]:
    def tokenize(mode: Any) -> list[dict[str, Any]]:
        result = []
        cursor = 0
        for morpheme in sudachi.tokenize(text, mode):
            surface = morpheme.surface()
            # Match the production teacher exporter: Sudachi may emit a
            # zero-width empty morpheme at sentence end.
            if not surface and morpheme.begin() == morpheme.end():
                continue
            end = cursor + len(surface)
            if not surface or text[cursor:end] != surface:
                raise ValueError(f"Sudachi surface mismatch at {cursor}: {surface!r}")
            pos = list(morpheme.part_of_speech())
            result.append({"start": cursor, "end": end,
                           "atom_type": map_sudachi_pos(pos),
                           "function": map_particle(surface, pos),
                           "inflections": map_inflections(
                               surface, morpheme.normalized_form(),
                               pos[5] if len(pos) > 5 and pos[5] != "*" else "")})
            cursor = end
        if cursor != len(text):
            raise ValueError(f"Sudachi did not cover text: {cursor}/{len(text)}")
        return result

    a = tokenize(tokenizer.Tokenizer.SplitMode.A)
    b = tokenize(tokenizer.Tokenizer.SplitMode.B)
    return {"text": text, "boundaries": {
                "a": [x["end"] for x in a], "b": [x["end"] for x in b],
                "bunsetsu": [len(text)], "clause": [len(text)], "sentence": [len(text)]},
            "spans": a, "b_spans": [{"start": x["start"], "end": x["end"]} for x in b],
            "bunsetsu": [{"start": 0, "end": len(text), "role": "UNKNOWN"}]}


def _ginza_prediction(text: str, doc: Any, ginza: Any) -> dict[str, Any]:
    raw_spans = list(ginza.bunsetu_spans(doc))
    chunks = []
    cursor = 0
    for index, span in enumerate(raw_spans):
        gap = text[cursor:span.start_char]
        if gap and not gap.isspace():
            raise ValueError(f"GiNZA uncovered text at {cursor}: {gap!r}")
        start = cursor
        end = span.end_char
        members = list(span)
        head = next((token for token in members if token.dep_.lower() == "root"), None)
        if head is None:
            head = next((token for token in members
                         if token.head.i < span.start or token.head.i >= span.end), members[0])
        topic = any(token.text in {"は", "も"} for token in members)
        predicate = head.pos_ in {"VERB", "ADJ"} or head.dep_.lower() == "root"
        chunks.append({"start": start, "end": end,
                       "role": map_bunsetsu_role(head.dep_, "TOPIC" if topic else None, predicate)})
        cursor = end
    tail = text[cursor:]
    if tail and not tail.isspace():
        raise ValueError(f"GiNZA did not cover text at {cursor}: {tail!r}")
    if tail and chunks:
        chunks[-1]["end"] = len(text)
    if not chunks:
        raise ValueError("GiNZA returned no bunsetsu")
    return {"text": text, "boundaries": {
                "a": [len(text)], "b": [len(text)],
                "bunsetsu": [x["end"] for x in chunks],
                "clause": [len(text)], "sentence": [len(text)]},
            "spans": [{"start": 0, "end": len(text), "atom_type": "OTHER",
                       "function": None, "inflections": []}],
            "b_spans": [{"start": 0, "end": len(text)}], "bunsetsu": chunks}


def _combine(sudachi: Mapping[str, Any], ginza: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(sudachi)
    result["boundaries"] = dict(sudachi["boundaries"])
    result["boundaries"].update({key: ginza["boundaries"][key] for key in ("bunsetsu", "clause", "sentence")})
    result["bunsetsu"] = ginza["bunsetsu"]
    return result


def _scoped(scores: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {key: scores[key] for key in fields}


def _load(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _load_model(checkpoint: Path, device_name: str) -> JpLexer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(device_name)
    model = JpLexer(ModelConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _model_predictions(rows: list[dict[str, Any]], model: JpLexer,
                       batch_size: int, device_name: str) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            batch = collate_records(group, model.config.codepoint_buckets, model.config.bigram_buckets)
            inputs = {key: value.to(device) for key, value in batch.inputs.items()}
            outputs = model(inputs)
            for index, row in enumerate(group):
                length = len(row["text"])
                one = {key: value[index, :length] for key, value in outputs.items()}
                result.append(decode_logits(row["text"], one))
    return result


def benchmark_dataset(rows: list[dict[str, Any]], nlp: Any, ginza: Any,
                      sudachi: Any, tokenizer: Any, checkpoint: Path,
                      batch_size: int, device_name: str) -> dict[str, Any]:
    sudachi_all = []
    ginza_all = []
    failures = []
    ginza_start = time.perf_counter()
    docs = list(nlp.pipe((row["text"] for row in rows), batch_size=64, n_process=1))
    ginza_pipe_ms = (time.perf_counter() - ginza_start) * 1000

    sudachi_start = time.perf_counter()
    for index, row in enumerate(rows):
        try:
            sudachi_all.append(_sudachi_prediction(row["text"], sudachi, tokenizer))
        except Exception as error:
            sudachi_all.append(None)
            failures.append({"index": index, "text": row["text"], "reason": str(error)})
    sudachi_ms = (time.perf_counter() - sudachi_start) * 1000

    ginza_conversion_start = time.perf_counter()
    for index, (row, doc, sudachi_prediction) in enumerate(zip(rows, docs, sudachi_all)):
        if sudachi_prediction is None:
            ginza_all.append(None)
            continue
        try:
            ginza_all.append(_ginza_prediction(row["text"], doc, ginza))
        except Exception as error:
            ginza_all.append(None)
            failures.append({"index": index, "text": row["text"], "reason": str(error)})
    ginza_ms = ginza_pipe_ms + (time.perf_counter() - ginza_conversion_start) * 1000

    keep = [index for index, prediction in enumerate(sudachi_all)
            if prediction is not None and ginza_all[index] is not None]
    rows = [rows[index] for index in keep]
    sudachi_predictions = [sudachi_all[index] for index in keep]
    ginza_predictions = [ginza_all[index] for index in keep]
    combined = [_combine(sudachi, ginza) for sudachi, ginza in zip(sudachi_predictions, ginza_predictions)]
    model = _load_model(checkpoint, device_name)
    model_start = time.perf_counter()
    model_predictions = _model_predictions(rows, model, batch_size, device_name)
    model_ms = (time.perf_counter() - model_start) * 1000
    all_model = score_corpus(rows, model_predictions)
    input_records = len(sudachi_all)
    def runtime(milliseconds: float, measured_batch_size: int) -> dict[str, float | int]:
        return {"wall_ms": milliseconds, "sentences_per_second":
                input_records / (milliseconds / 1000) if milliseconds else 0.0,
                "ms_per_sentence": milliseconds / input_records if input_records else 0.0,
                "batch_size": measured_batch_size,
                "measurement": "model inference wall time divided by all input sentences; model load excluded"}
    runtimes = {"jpu": runtime(model_ms, batch_size), "sudachi": runtime(sudachi_ms, 1),
                "ginza": runtime(ginza_ms, 64), "sudachi+ginza": runtime(sudachi_ms + ginza_ms, batch_size)}
    all_teacher = score_corpus(rows, combined)
    all_sudachi = score_corpus(rows, sudachi_predictions)
    all_ginza = score_corpus(rows, ginza_predictions)
    return {
        "input_records": input_records, "records": len(rows), "teacher_failures": failures,
        "device": device_name,
        "systems": {
            "jpu": {**all_model, "runtime": runtimes["jpu"]},
            "sudachi+ginza": {**all_teacher, "runtime": runtimes["sudachi+ginza"]},
            "sudachi": {
                "scope": ["boundaries.a", "boundaries.b", "atom", "inflection", "particle_function"],
                "runtime": runtimes["sudachi"],
                "scores": {"boundaries": {"a": all_sudachi["boundaries"]["a"], "b": all_sudachi["boundaries"]["b"]},
                           "atom": all_sudachi["atom"], "inflection": all_sudachi["inflection"],
                           "particle_function": all_sudachi["particle_function"]}},
            "ginza": {
                "scope": ["boundaries.bunsetsu", "bunsetsu_role"],
                "runtime": runtimes["ginza"],
                "scores": {"boundaries": {"bunsetsu": all_ginza["boundaries"]["bunsetsu"]},
                           "bunsetsu_role": all_ginza["bunsetsu_role"]}},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwdlc-test", type=Path, default=Path("artifacts/gold/kwdlc/test.jsonl"))
    parser.add_argument("--ud-test", type=Path, default=Path("artifacts/gold/ud-gsd/test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="local checkpoint; checkpoints are not redistributed")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="device for JPU inference timing; scores are device-independent")
    parser.add_argument("--output", type=Path, default=Path("artifacts/teacher-benchmark.json"))
    args = parser.parse_args()

    from sudachipy import dictionary, tokenizer
    import ginza
    import spacy
    sudachi = dictionary.Dictionary().create()
    nlp = spacy.load("ja_ginza", disable=["ner"])
    versions = {name: importlib.metadata.version(name)
                for name in ("SudachiPy", "SudachiDict-core", "spacy", "ginza", "ja-ginza")}
    datasets = {}
    for name, path in (("kwdlc-test", args.kwdlc_test), ("ud-gsd-test", args.ud_test)):
        rows = _load(path, args.limit)
        datasets[name] = benchmark_dataset(rows, nlp, ginza, sudachi, tokenizer,
                                            args.checkpoint, args.batch_size, args.device)
    report = {
        "schema": "jpu.teacher-benchmark.v1",
        "teacher_versions": versions,
        "checkpoint": str(args.checkpoint),
        "metric_contract": {
            "boundaries": "exact character-gap endpoint F1",
            "atom_function_role": "character-position weighted label F1; span labels are expanded across covered codepoints",
            "inflection": "character-position weighted multilabel F1",
            "complete_tree": "exact equality of boundaries and labeled spans",
        },
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
