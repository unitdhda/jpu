"""Deterministic Unicode features and canonical-record batch targets.

The runtime has no dictionary or teacher dependency. Hashing is deliberately
implemented with integer arithmetic rather than Python's randomized hash.
Offsets throughout this module are Unicode-codepoint offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from data.schemas.canonical import (ATOM_TYPES, BUNSETSU_ROLES, INFLECTIONS,
                                    PARTICLE_FUNCTIONS)

SCRIPT_NAMES = ("HIRAGANA", "KATAKANA", "KANJI", "LATIN", "DIGIT",
                "PUNCT", "SYMBOL", "WHITESPACE", "OTHER")
PUNCT_CLASSES = ("NONE", "SENTENCE", "OPEN", "CLOSE", "COMMA")
DIGIT_CLASSES = ("NONE", "ASCII", "FULLWIDTH", "OTHER")

# One-hot feature layout is part of the checkpoint contract.
CAT_FEATURE_NAMES = tuple(
    "script:" + x for x in SCRIPT_NAMES
) + ("script_transition", "small_kana", "prolonged_mark", "iteration_mark") + tuple(
    "punct:" + x for x in PUNCT_CLASSES
) + tuple("digit:" + x for x in DIGIT_CLASSES)


def _mix64(value: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def hash_bucket(value: int, buckets: int) -> int:
    """Stable bucket in ``[0, buckets)`` for a Unicode/codepoint key."""
    if buckets <= 0 or buckets & (buckets - 1):
        raise ValueError("bucket count must be a positive power of two")
    return _mix64(value) & (buckets - 1)


def _script(ch: str) -> str:
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309F:
        return "HIRAGANA"
    if 0x30A0 <= cp <= 0x30FF:
        return "KATAKANA"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "KANJI"
    if ch.isascii() and ch.isalpha():
        return "LATIN"
    if ch.isdigit():
        return "DIGIT"
    category = unicodedata.category(ch)
    if category.startswith("P"):
        return "PUNCT"
    if category.startswith("S"):
        return "SYMBOL"
    if ch.isspace():
        return "WHITESPACE"
    return "OTHER"


def _punct_class(ch: str) -> str:
    if ch in "。！？!?．": return "SENTENCE"
    if ch in "「『（【〔〈《": return "OPEN"
    if ch in "」』）】〕〉》": return "CLOSE"
    if ch in "、，,": return "COMMA"
    return "NONE"


def _digit_class(ch: str) -> str:
    if "0" <= ch <= "9": return "ASCII"
    if "０" <= ch <= "９": return "FULLWIDTH"
    return "OTHER" if _script(ch) == "DIGIT" else "NONE"


def categorical_features(text: str) -> torch.Tensor:
    """Return deterministic ``[len(text), len(CAT_FEATURE_NAMES)]`` one-hots."""
    index = {name: i for i, name in enumerate(CAT_FEATURE_NAMES)}
    rows = torch.zeros((len(text), len(CAT_FEATURE_NAMES)), dtype=torch.float32)
    previous = None
    for i, ch in enumerate(text):
        rows[i, index["script:" + _script(ch)]] = 1.0
        if previous is not None and previous != _script(ch):
            rows[i, index["script_transition"]] = 1.0
        if ch in "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ":
            rows[i, index["small_kana"]] = 1.0
        if ch in "ーｰ":
            rows[i, index["prolonged_mark"]] = 1.0
        if ch in "々〆ヽヾゝゞー":
            rows[i, index["iteration_mark"]] = 1.0
        rows[i, index["punct:" + _punct_class(ch)]] = 1.0
        rows[i, index["digit:" + _digit_class(ch)]] = 1.0
        previous = _script(ch)
    return rows


def encode_text(text: str, codepoint_buckets: int = 8192,
                bigram_buckets: int = 8192) -> Dict[str, torch.Tensor]:
    """Encode raw text without normalization or a learned character vocab."""
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    chars = list(text)
    codepoints = torch.tensor([hash_bucket(ord(ch), codepoint_buckets) for ch in chars], dtype=torch.long)
    bigrams = []
    for i, ch in enumerate(chars):
        right = ord(chars[i + 1]) if i + 1 < len(chars) else 0
        # Include position direction and a sentinel for the final character.
        bigrams.append(hash_bucket((ord(ch) << 21) ^ right ^ 0x9E3779B9, bigram_buckets))
    return {"codepoints": codepoints, "bigrams": torch.tensor(bigrams, dtype=torch.long),
            "categorical": categorical_features(text), "length": torch.tensor(len(text))}


@dataclass
class ModelBatch:
    inputs: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]


def _span_value(target: torch.Tensor, start: int, end: int, value: int) -> None:
    target[start:end] = value


def record_targets(record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Expand span labels to character positions and boundaries to gaps."""
    text = record["text"]
    n = len(text)
    boundary = torch.zeros((n, 5), dtype=torch.float32)
    for level, column in zip(("a", "b", "bunsetsu", "clause", "sentence"), range(5)):
        for end in record["boundaries"][level]:
            boundary[end - 1, column] = 1.0
    atom = torch.zeros(n, dtype=torch.long)
    function = torch.full((n,), PARTICLE_FUNCTIONS.index("UNKNOWN"), dtype=torch.long)
    function_mask = torch.zeros(n, dtype=torch.bool)
    inflection = torch.zeros((n, len(INFLECTIONS)), dtype=torch.float32)
    inflection_mask = torch.zeros(n, dtype=torch.bool)
    atom_index = {x: i for i, x in enumerate(ATOM_TYPES)}
    function_index = {x: i for i, x in enumerate(PARTICLE_FUNCTIONS)}
    infl_index = {x: i for i, x in enumerate(INFLECTIONS)}
    for span in record["spans"]:
        start, end = span["start"], span["end"]
        atom[start:end] = atom_index[span["atom_type"]]
        if span["atom_type"] == "PARTICLE":
            function_mask[start:end] = True
            function[start:end] = function_index[span.get("function") or "UNKNOWN"]
        if span["atom_type"] in {"VERB_STEM", "AUXILIARY", "COPULA", "ADJECTIVE"}:
            inflection_mask[start:end] = True
            for label in span.get("inflections", []):
                inflection[start:end, infl_index[label]] = 1.0
    role = torch.zeros(n, dtype=torch.long)
    role_index = {x: i for i, x in enumerate(BUNSETSU_ROLES)}
    for span in record["bunsetsu"]:
        _span_value(role, span["start"], span["end"], role_index[span["role"]])
    return {"boundaries": boundary, "atom": atom, "function": function,
            "function_mask": function_mask, "inflection": inflection,
            "inflection_mask": inflection_mask, "role": role}


def collate_records(records: Sequence[Mapping[str, Any]], codepoint_buckets: int = 8192,
                    bigram_buckets: int = 8192) -> ModelBatch:
    """Pad a list of canonical records and preserve a boolean valid mask."""
    if not records:
        raise ValueError("cannot collate an empty batch")
    encoded = [encode_text(r["text"], codepoint_buckets, bigram_buckets) for r in records]
    targets = [record_targets(r) for r in records]
    max_len = max(x["codepoints"].numel() for x in encoded)
    batch_size = len(records)
    cp = torch.zeros((batch_size, max_len), dtype=torch.long)
    bg = torch.zeros((batch_size, max_len), dtype=torch.long)
    cats = torch.zeros((batch_size, max_len, len(CAT_FEATURE_NAMES)), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    result: Dict[str, torch.Tensor] = {}
    for i, (features, labels) in enumerate(zip(encoded, targets)):
        length = features["codepoints"].numel()
        cp[i, :length] = features["codepoints"]
        bg[i, :length] = features["bigrams"]
        cats[i, :length] = features["categorical"]
        mask[i, :length] = True
        for key, value in labels.items():
            if key.endswith("_mask"):
                result.setdefault(key, torch.zeros((batch_size, max_len), dtype=torch.bool))[i, :length] = value
            elif value.ndim == 1:
                result.setdefault(key, torch.zeros((batch_size, max_len), dtype=value.dtype))[i, :length] = value
            else:
                result.setdefault(key, torch.zeros((batch_size, max_len, value.shape[-1]), dtype=value.dtype))[i, :length] = value
    result["mask"] = mask
    return ModelBatch({"codepoints": cp, "bigrams": bg, "categorical": cats, "mask": mask}, result)
