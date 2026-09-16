"""Strict monotonic alignment of teacher tokens to raw Unicode text."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AlignmentFailure:
    reason: str
    teacher: str
    token_index: int
    detail: str


def align_surfaces(text: str, tokens: Sequence[Mapping[str, Any]], teacher: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[AlignmentFailure]]:
    """Align exact token surfaces without normalization or guessed offsets.

    Teachers may provide start/end offsets. They are accepted only if their
    raw slice exactly equals the surface and begins at the current cursor.
    Otherwise exact concatenation is attempted. Any disagreement is fatal.
    """
    cursor = 0
    aligned = []
    for index, token in enumerate(tokens):
        surface = token.get("surface")
        if not isinstance(surface, str) or not surface:
            return None, AlignmentFailure("empty_surface", teacher, index, repr(surface))
        supplied_start, supplied_end = token.get("start"), token.get("end")
        if supplied_start is not None or supplied_end is not None:
            if not isinstance(supplied_start, int) or not isinstance(supplied_end, int):
                return None, AlignmentFailure("invalid_teacher_offsets", teacher, index, repr((supplied_start, supplied_end)))
            if supplied_start != cursor:
                return None, AlignmentFailure("non_monotonic_offset", teacher, index, "expected %d, got %d" % (cursor, supplied_start))
            if text[supplied_start:supplied_end] != surface:
                return None, AlignmentFailure("surface_offset_mismatch", teacher, index, repr(surface))
            end = supplied_end
        else:
            end = cursor + len(surface)
            if text[cursor:end] != surface:
                return None, AlignmentFailure("surface_mismatch", teacher, index, "at raw offset %d: %r" % (cursor, surface))
        item = dict(token)
        item["start"], item["end"] = cursor, end
        aligned.append(item)
        cursor = end
    if cursor != len(text):
        return None, AlignmentFailure("teacher_does_not_cover_text", teacher, len(tokens), "covered %d/%d codepoints" % (cursor, len(text)))
    return aligned, None


def endpoints(spans: Sequence[Mapping[str, Any]]) -> List[int]:
    return [span["end"] for span in spans]


def check_endpoint_hierarchy(a: Sequence[int], b: Sequence[int], bunsetsu: Sequence[int], clause: Sequence[int], sentence: Sequence[int]) -> Optional[str]:
    levels = (("a", set(a)), ("b", set(b)), ("bunsetsu", set(bunsetsu)),
              ("clause", set(clause)), ("sentence", set(sentence)))
    for (fine_name, fine), (coarse_name, coarse) in zip(levels, levels[1:]):
        missing = coarse - fine
        if missing:
            return "%s endpoints absent from %s: %s" % (coarse_name, fine_name, sorted(missing))
    return None
