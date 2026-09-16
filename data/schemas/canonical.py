"""Canonical Phase 1 record and invariant validation.

All offsets are Python Unicode-codepoint offsets into ``text``.  A boundary is
an endpoint after a character, so valid values are in ``1..len(text)``.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence

ATOM_TYPES = (
    "NOUN_LIKE", "PROPER_LIKE", "VERB_STEM", "ADJECTIVE", "ADVERB",
    "PARTICLE", "AUXILIARY", "COPULA", "PREFIX_SUFFIX", "NUMBER",
    "PUNCT_SYMBOL", "OTHER",
)
PARTICLE_FUNCTIONS = (
    "TOPIC", "SUBJECT_CASE", "OBJECT", "LOCATION", "DIRECTION", "TIME",
    "COMITATIVE", "QUOTATIVE", "GENITIVE", "CONNECTIVE", "OTHER", "UNKNOWN",
)
INFLECTIONS = (
    "PAST", "NEGATIVE", "POLITE", "TE_FORM", "CONDITIONAL", "VOLITIONAL",
    "PASSIVE", "CAUSATIVE", "POTENTIAL", "ASPECTUAL",
)
BUNSETSU_ROLES = (
    "TOPIC", "CASED_NOMINAL", "PREDICATE", "MODIFIER", "CONNECTIVE",
    "ADVERBIAL", "PUNCT", "OTHER", "UNKNOWN",
)
LEVELS = ("a", "b", "bunsetsu", "clause", "sentence")


def _sorted_unique(values: Iterable[int]) -> List[int]:
    return sorted(set(values))


def make_record(text: str, source: str, document_id: str,
                boundaries: Mapping[str, Iterable[int]],
                spans: Sequence[Mapping[str, Any]],
                bunsetsu: Sequence[Mapping[str, Any]],
                b_spans: Sequence[Mapping[str, Any]] = (),
                metadata: Mapping[str, Any] = None) -> Dict[str, Any]:
    """Create a normalized, readable canonical record.

    This function intentionally does not repair invalid input.  Call
    ``validate_record`` before writing records to a training shard.
    """
    record = {
        "schema_version": "jpu.phase1.v1",
        "text": text,
        "source": source,
        "document_id": document_id,
        "boundaries": {level: _sorted_unique(boundaries.get(level, ())) for level in LEVELS},
        "spans": [dict(span) for span in spans],
        "b_spans": [dict(span) for span in b_spans],
        "bunsetsu": [dict(span) for span in bunsetsu],
    }
    if metadata:
        record["metadata"] = dict(metadata)
    return record


def validate_record(record: Mapping[str, Any]) -> List[str]:
    """Return all invariant violations; an empty list means valid."""
    errors = []
    text = record.get("text")
    if not isinstance(text, str) or not text:
        return ["text_missing_or_empty"]
    n = len(text)

    boundaries = record.get("boundaries")
    if not isinstance(boundaries, Mapping):
        return ["boundaries_missing"]
    previous = None
    for level in LEVELS:
        values = boundaries.get(level)
        if not isinstance(values, list):
            errors.append("%s_boundaries_not_list" % level)
            continue
        if values != sorted(set(values)):
            errors.append("%s_boundaries_not_sorted_unique" % level)
        if any(not isinstance(x, int) or x < 1 or x > n for x in values):
            errors.append("%s_boundary_out_of_range" % level)
        current = set(values)
        if previous is not None and not current.issubset(previous):
            errors.append("nesting_%s_not_subset_of_previous" % level)
        previous = current

    spans = record.get("spans")
    if not isinstance(spans, list):
        errors.append("spans_missing_or_not_list")
    else:
        errors.extend(_validate_spans(spans, n, "a"))
        if spans and sorted(span["end"] for span in spans) != boundaries.get("a", []):
            errors.append("a_spans_do_not_match_a_boundaries")

    b_spans = record.get("b_spans", [])
    if not isinstance(b_spans, list):
        errors.append("b_spans_not_list")
    else:
        errors.extend(_validate_spans(b_spans, n, "b"))
        if b_spans and sorted(span["end"] for span in b_spans) != boundaries.get("b", []):
            errors.append("b_spans_do_not_match_b_boundaries")

    bunsetsu = record.get("bunsetsu")
    if not isinstance(bunsetsu, list):
        errors.append("bunsetsu_missing_or_not_list")
    else:
        errors.extend(_validate_spans(bunsetsu, n, "bunsetsu"))
        if bunsetsu and sorted(span["end"] for span in bunsetsu) != boundaries.get("bunsetsu", []):
            errors.append("bunsetsu_spans_do_not_match_bunsetsu_boundaries")
        for span in bunsetsu:
            if span.get("role") not in BUNSETSU_ROLES:
                errors.append("invalid_bunsetsu_role")

    # A/B/chunk endpoints must be endpoints from the corresponding hierarchy.
    for child, parent in (("a", "b"), ("b", "bunsetsu"),
                          ("bunsetsu", "clause"), ("clause", "sentence")):
        child_values = set(boundaries.get(child, []))
        parent_values = set(boundaries.get(parent, []))
        if not child_values.issuperset(parent_values):
            errors.append("%s_boundaries_missing_in_%s" % (parent, child))

    return errors


def _validate_spans(spans: Sequence[Mapping[str, Any]], n: int, kind: str) -> List[str]:
    errors = []
    last_end = 0
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= n):
            errors.append("%s_span_out_of_range" % kind)
            continue
        if start != last_end:
            errors.append("%s_spans_not_contiguous" % kind)
        last_end = end
        if kind == "a" and span.get("atom_type") not in ATOM_TYPES:
            errors.append("invalid_atom_type")
        if kind == "a":
            function = span.get("function")
            if function is not None and function not in PARTICLE_FUNCTIONS:
                errors.append("invalid_particle_function")
            inflections = span.get("inflections", [])
            if not isinstance(inflections, list) or any(x not in INFLECTIONS for x in inflections):
                errors.append("invalid_inflections")
    if spans and last_end != n:
        errors.append("%s_spans_do_not_cover_text" % kind)
    return errors
