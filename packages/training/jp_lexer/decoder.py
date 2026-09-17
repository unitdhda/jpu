"""Deterministic hierarchical composer for model predictions."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from data.schemas.canonical import ATOM_TYPES, BUNSETSU_ROLES, INFLECTIONS, LEVELS, PARTICLE_FUNCTIONS


def nested_boundaries(predicted: Mapping[str, Iterable[int]], length: int) -> Dict[str, List[int]]:
    """Make predicted endpoints nested, retain only valid gaps, and close text."""
    if length < 1:
        raise ValueError("length must be positive")
    sets = {level: {int(x) for x in predicted.get(level, ()) if 1 <= int(x) <= length} for level in LEVELS}
    sets["sentence"].add(length)
    # A high boundary is necessarily every lower boundary as well.
    for high, lower_levels in (("sentence", LEVELS[:-1]), ("clause", LEVELS[:3]),
                               ("bunsetsu", LEVELS[:2]), ("b", ["a"])):
        for lower in lower_levels:
            sets[lower].update(sets[high])
    return {level: sorted(sets[level]) for level in LEVELS}


def _partition(start: int, end: int, endpoints: Sequence[int]) -> List[Dict[str, int]]:
    points = [x for x in sorted(set(endpoints)) if start < x <= end]
    result=[]; cursor=start
    for point in points:
        result.append({"start": cursor, "end": point}); cursor=point
    if cursor != end:
        result.append({"start": cursor, "end": end})
    return result


def _label_at(logits: Optional[torch.Tensor], index: int, labels: Sequence[str], multilabel=False):
    if logits is None:
        return None
    row = logits[index].detach().cpu()
    if multilabel:
        return [labels[i] for i, value in enumerate(torch.sigmoid(row)) if value >= .5]
    return labels[int(row.argmax())]


def decode_logits(text: str, outputs: Optional[Mapping[str, torch.Tensor]] = None,
                  threshold: float = .5) -> Dict[str, Any]:
    """Decode one unbatched model output into spans and a nested tree."""
    outputs = outputs or {}
    boundary_logits = outputs.get("boundaries")
    predicted = {}
    if boundary_logits is None:
        predicted = {level: [len(text)] for level in LEVELS}
    else:
        probabilities = torch.sigmoid(boundary_logits.detach().cpu())
        for column, level in enumerate(LEVELS):
            predicted[level] = [i + 1 for i, p in enumerate(probabilities[:, column]) if float(p) >= threshold]
    boundaries = nested_boundaries(predicted, len(text))
    atom_logits = outputs.get("atom"); function_logits = outputs.get("function")
    inflection_logits = outputs.get("inflection"); role_logits = outputs.get("role")
    atoms=[]
    for span in _partition(0, len(text), boundaries["a"]):
        i=span["end"]-1
        atom_type = _label_at(atom_logits, i, ATOM_TYPES) or "OTHER"
        atoms.append({**span, "atom_type": atom_type,
                      # These heads are masked during training and are not
                      # semantically attached to inapplicable spans at decode.
                      "function": _label_at(function_logits, i, PARTICLE_FUNCTIONS)
                                   if atom_type == "PARTICLE" else None,
                      "inflections": _label_at(inflection_logits, i, INFLECTIONS, True)
                                    if atom_type in {"VERB_STEM", "AUXILIARY", "COPULA", "ADJECTIVE"} else []})
    b_spans = _partition(0, len(text), boundaries["b"])
    chunks=[]
    for span in _partition(0, len(text), boundaries["bunsetsu"]):
        i=span["end"]-1
        chunks.append({**span, "role": _label_at(role_logits, i, BUNSETSU_ROLES) or "UNKNOWN"})
    tree = _tree(text, boundaries, atoms, b_spans, chunks)
    return {"text": text, "boundaries": boundaries, "spans": atoms,
            "b_spans": b_spans, "bunsetsu": chunks, "tree": tree}


def _node(kind: str, start: int, end: int, text: str, children=None, **attrs):
    result={"type":kind,"start":start,"end":end,"text":text[start:end]}
    if attrs: result.update(attrs)
    result["children"] = children or []
    return result


def _tree(text: str, boundaries: Mapping[str, Sequence[int]], atoms: Sequence[Mapping[str, Any]],
          b_spans: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def atom_node(span):
        return _node("A", span["start"], span["end"], text, atom_type=span["atom_type"],
                     function=span["function"], inflections=span["inflections"])
    def b_node(span):
        children=[atom_node(x) for x in atoms if span["start"] <= x["start"] and x["end"] <= span["end"]]
        return _node("B", span["start"], span["end"], text, children)
    def chunk_node(span):
        children=[b_node(x) for x in b_spans if span["start"] <= x["start"] and x["end"] <= span["end"]]
        # In well-formed output every B lies in one chunk; use A fallback if a
        # caller supplies manually inconsistent endpoints.
        if not children:
            children=[atom_node(x) for x in atoms if span["start"] <= x["start"] and x["end"] <= span["end"]]
        return _node("BUNSETSU", span["start"], span["end"], text, children, role=span["role"])
    clauses=[]
    for clause in _partition(0, len(text), boundaries["clause"]):
        children=[chunk_node(x) for x in chunks if clause["start"] <= x["start"] and x["end"] <= clause["end"]]
        clauses.append(_node("CLAUSE", clause["start"], clause["end"], text, children))
    sentences=[]
    for sentence in _partition(0, len(text), boundaries["sentence"]):
        children=[x for x in clauses if sentence["start"] <= x["start"] and x["end"] <= sentence["end"]]
        sentences.append(_node("SENTENCE", sentence["start"], sentence["end"], text, children))
    return sentences[0] if len(sentences)==1 else _node("DOCUMENT", 0, len(text), text, sentences)


def decode(text: str, outputs: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, Any]:
    """Public alias with tensors accepted as returned by ``JpLexer``."""
    return decode_logits(text, outputs)
