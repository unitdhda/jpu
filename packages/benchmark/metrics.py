"""Metrics for the jpu teacher/model comparison."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from data.schemas.canonical import ATOM_TYPES, BUNSETSU_ROLES, INFLECTIONS, PARTICLE_FUNCTIONS, LEVELS


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _merge_label_counts(target: dict[str, Counter], source: Mapping[str, Counter]) -> None:
    for label, counter in source.items():
        target[label].update(counter)


def boundary_counts(gold: Mapping[str, Any], predicted: Mapping[str, Any]) -> dict[str, Counter]:
    result = {}
    for level in LEVELS:
        expected, actual = set(gold["boundaries"][level]), set(predicted["boundaries"][level])
        result[level] = Counter(tp=len(expected & actual), fp=len(actual - expected), fn=len(expected - actual))
    return result


def _char_values(record: Mapping[str, Any], field: str, default: Any) -> list[Any]:
    values = [default for _ in record["text"]]
    for span in record.get(field, []):
        for index in range(span["start"], span["end"]):
            values[index] = span.get("atom_type" if field == "spans" else "role", default)
    return values


def _char_functions(record: Mapping[str, Any]) -> list[str | None]:
    values: list[str | None] = [None for _ in record["text"]]
    for span in record.get("spans", []):
        if span.get("atom_type") == "PARTICLE":
            for index in range(span["start"], span["end"]):
                values[index] = span.get("function") or "UNKNOWN"
    return values


def _char_inflections(record: Mapping[str, Any]) -> list[set[str] | None]:
    values: list[set[str] | None] = [None for _ in record["text"]]
    for span in record.get("spans", []):
        if span.get("atom_type") in {"VERB_STEM", "AUXILIARY", "COPULA", "ADJECTIVE"}:
            flags = set(span.get("inflections", []))
            for index in range(span["start"], span["end"]):
                values[index] = flags
    return values


def _classification_counts(gold: Sequence[Any], predicted: Sequence[Any], labels: Iterable[str]) -> dict[str, Counter]:
    counters = {label: Counter() for label in labels}
    for expected, actual in zip(gold, predicted):
        if expected is None:
            continue
        for label in counters:
            if expected == label and actual == label:
                counters[label]["tp"] += 1
            elif expected == label:
                counters[label]["fn"] += 1
            elif actual == label:
                counters[label]["fp"] += 1
    return counters


def _macro(counters: Mapping[str, Counter]) -> dict[str, Any]:
    scores = {label: _prf(c["tp"], c["fp"], c["fn"]) for label, c in counters.items()}
    active = [score["f1"] for label, score in scores.items()
              if counters[label]["tp"] + counters[label]["fn"] + counters[label]["fp"] > 0]
    return {"macro_f1": sum(active) / len(active) if active else 0.0, "by_label": scores}


def _multilabel_scores(gold: Sequence[set[str] | None], predicted: Sequence[set[str] | None]) -> dict[str, Any]:
    counters = {label: Counter() for label in INFLECTIONS}
    for expected, actual in zip(gold, predicted):
        if expected is None:
            continue
        actual = actual or set()
        for label in counters:
            if label in expected and label in actual:
                counters[label]["tp"] += 1
            elif label in expected:
                counters[label]["fn"] += 1
            elif label in actual:
                counters[label]["fp"] += 1
    total = Counter()
    for counter in counters.values(): total.update(counter)
    return {"micro": _prf(total["tp"], total["fp"], total["fn"]), **_macro(counters)}


def _signature(record: Mapping[str, Any]) -> tuple:
    spans = tuple((x["start"], x["end"], x.get("atom_type"), x.get("function"), tuple(x.get("inflections", [])))
                   for x in record.get("spans", []))
    chunks = tuple((x["start"], x["end"], x.get("role")) for x in record.get("bunsetsu", []))
    b_spans = tuple((x["start"], x["end"]) for x in record.get("b_spans", []))
    boundaries = tuple(tuple(record["boundaries"][level]) for level in LEVELS)
    return boundaries, spans, b_spans, chunks


def score_corpus(gold_records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(gold_records) != len(predictions):
        raise ValueError("gold/prediction lengths differ")
    boundaries = {level: Counter() for level in LEVELS}
    atom_counts = {label: Counter() for label in ATOM_TYPES}
    function_counts = {label: Counter() for label in PARTICLE_FUNCTIONS}
    role_counts = {label: Counter() for label in BUNSETSU_ROLES}
    inflection_gold: list[set[str] | None] = []
    inflection_predicted: list[set[str] | None] = []
    tree_matches = 0
    for gold, predicted in zip(gold_records, predictions):
        for level, count in boundary_counts(gold, predicted).items(): boundaries[level].update(count)
        _merge_label_counts(atom_counts, _classification_counts(_char_values(gold, "spans", "OTHER"), _char_values(predicted, "spans", "OTHER"), ATOM_TYPES))
        _merge_label_counts(function_counts, _classification_counts(_char_functions(gold), _char_functions(predicted), PARTICLE_FUNCTIONS))
        _merge_label_counts(role_counts, _classification_counts(_char_values(gold, "bunsetsu", "UNKNOWN"), _char_values(predicted, "bunsetsu", "UNKNOWN"), BUNSETSU_ROLES))
        inflection_gold.extend(_char_inflections(gold))
        inflection_predicted.extend(_char_inflections(predicted))
        tree_matches += _signature(gold) == _signature(predicted)
    atom_total = Counter(); function_total = Counter(); role_total = Counter()
    for counters, total in ((atom_counts, atom_total), (function_counts, function_total), (role_counts, role_total)):
        for counter in counters.values(): total.update(counter)
    return {
        "sentences": len(gold_records),
        "boundaries": {level: _prf(c["tp"], c["fp"], c["fn"]) for level, c in boundaries.items()},
        "atom": {"micro": _prf(atom_total["tp"], atom_total["fp"], atom_total["fn"]), **_macro(atom_counts)},
        "inflection": _multilabel_scores(inflection_gold, inflection_predicted),
        "particle_function": _macro(function_counts),
        "bunsetsu_role": _macro(role_counts),
        "complete_tree_exact_match": tree_matches / len(gold_records) if gold_records else 0.0,
    }
