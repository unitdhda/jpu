"""Streaming Phase 1 corpus statistics."""
from collections import Counter
import unicodedata


def script(ch):
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309f: return "HIRAGANA"
    if 0x30a0 <= cp <= 0x30ff: return "KATAKANA"
    if 0x4e00 <= cp <= 0x9fff or 0x3400 <= cp <= 0x4dbf: return "KANJI"
    if ch.isascii() and ch.isalpha(): return "LATIN"
    if ch.isdigit(): return "DIGIT"
    cat = unicodedata.category(ch)
    if cat.startswith("P"): return "PUNCT"
    if cat.startswith("S"): return "SYMBOL"
    if ch.isspace(): return "WHITESPACE"
    return "OTHER"


class CorpusStats:
    def __init__(self):
        self.total = self.aligned = 0
        self.drop_reasons = Counter()
        self.atom_types = Counter(); self.functions = Counter(); self.inflections = Counter()
        self.bunsetsu_roles = Counter(); self.boundaries = Counter(); self.lengths = Counter(); self.scripts = Counter()

    def add_success(self, record):
        self.total += 1; self.aligned += 1
        self.lengths[len(record["text"])] += 1
        self.scripts.update(script(c) for c in record["text"])
        for level, values in record["boundaries"].items(): self.boundaries[level] += len(values)
        for span in record["spans"]:
            self.atom_types[span["atom_type"]] += 1
            if span["function"] is not None: self.functions[span["function"]] += 1
            self.inflections.update(span["inflections"])
        self.bunsetsu_roles.update(x["role"] for x in record["bunsetsu"])

    def add_failure(self, failure):
        self.total += 1; self.drop_reasons[failure["reason"]] += 1

    def as_dict(self):
        dropped = self.total - self.aligned
        return {"total_sentences": self.total, "successfully_aligned": self.aligned,
                "dropped": dropped, "drop_percentage": (100.0*dropped/self.total if self.total else 0.0),
                "drop_reasons": dict(self.drop_reasons), "label_distribution": {
                    "atom_type": dict(self.atom_types), "function": dict(self.functions),
                    "inflection": dict(self.inflections), "bunsetsu_role": dict(self.bunsetsu_roles)},
                "boundary_distribution": dict(self.boundaries),
                "sentence_length_distribution": dict(sorted(self.lengths.items())),
                "script_distribution": dict(self.scripts)}
