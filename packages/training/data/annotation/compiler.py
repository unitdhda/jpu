"""Compile raw teacher JSON into canonical records.

This module has no runtime dependency on Sudachi or GiNZA. Teacher adapters
export JSON first; compilation and alignment are deterministic and testable.
"""

from collections import Counter
from typing import Any, Dict, Mapping, Optional, Tuple

from data.alignment.align import align_surfaces, check_endpoint_hierarchy, endpoints
from data.annotation.mappings import (MAPPING_VERSION, map_bunsetsu_role,
                                      map_inflections, map_particle,
                                      map_sudachi_pos)
from data.schemas.canonical import make_record, validate_record


def compile_example(raw: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    text = raw.get("text", "")
    if not isinstance(text, str) or not (4 <= len(text) <= 128):
        return None, _failure(raw, "length_out_of_range", "raw", "length=%d" % len(text))
    required = ("sudachi_a", "sudachi_b", "ginza_bunsetsu")
    aligned = {}
    for name in required:
        if name not in raw:
            return None, _failure(raw, "teacher_missing", name, "")
        aligned[name], failure = align_surfaces(text, raw[name], name)
        if failure:
            return None, _failure(raw, failure.reason, failure.teacher, failure.detail)

    a_spans = []
    for token in aligned["sudachi_a"]:
        pos = token.get("pos", [])
        atom_type = map_sudachi_pos(pos)
        a_spans.append({
            "start": token["start"], "end": token["end"],
            "atom_type": atom_type,
            "function": map_particle(token["surface"], pos, token.get("context")),
            "inflections": map_inflections(token["surface"], token.get("normalized_form", ""), token.get("inflection_form", "")),
        })
    b_spans = [{"start": x["start"], "end": x["end"]} for x in aligned["sudachi_b"]]
    chunks = []
    for chunk in aligned["ginza_bunsetsu"]:
        chunks.append({
            "start": chunk["start"], "end": chunk["end"],
            "role": map_bunsetsu_role(chunk.get("dependency", ""), chunk.get("particle_function"), bool(chunk.get("is_predicate"))),
        })

    a_ends, b_ends, chunk_ends = endpoints(a_spans), endpoints(b_spans), endpoints(chunks)
    clause_ends = sorted(set(raw.get("clause_boundaries", [len(text)])))
    sentence_ends = [len(text)]
    disagreement = check_endpoint_hierarchy(a_ends, b_ends, chunk_ends, clause_ends, sentence_ends)
    if disagreement:
        return None, _failure(raw, "teacher_boundary_disagreement", "cross_teacher", disagreement)

    record = make_record(
        text, str(raw.get("source", "unknown")), str(raw.get("document_id", "")),
        {"a": a_ends, "b": b_ends, "bunsetsu": chunk_ends,
         "clause": clause_ends, "sentence": sentence_ends},
        a_spans, chunks, b_spans,
        {"mapping_version": MAPPING_VERSION,
         "teacher_versions": dict(raw.get("teacher_versions", {})),
         "alignment_adjustments": list(raw.get("alignment_adjustments", []))},
    )
    errors = validate_record(record)
    if errors:
        return None, _failure(raw, "canonical_invariant_failure", "compiler", ";".join(errors))
    return record, None


def _failure(raw: Mapping[str, Any], reason: str, teacher: str, detail: str) -> Dict[str, Any]:
    return {"source": raw.get("source"), "document_id": raw.get("document_id"),
            "text": raw.get("text"), "reason": reason, "teacher": teacher,
            "detail": detail}
