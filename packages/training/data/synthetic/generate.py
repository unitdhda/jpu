"""Deterministic synthetic verb morphology.

Splits are assigned by lemma family, never by generated form. These records are
kept identifiable as synthetic so a batch sampler can cap them at 10--15%.
"""
import hashlib
from typing import Dict, Iterable, List

GODAN_K = {
    "dictionary": ("く", []), "past": ("いた", ["PAST"]),
    "negative": ("かない", ["NEGATIVE"]), "negative_past": ("かなかった", ["NEGATIVE", "PAST"]),
    "conditional": ("けば", ["CONDITIONAL"]), "volitional": ("こう", ["VOLITIONAL"]),
    "causative": ("かせる", ["CAUSATIVE"]), "passive": ("かれる", ["PASSIVE"]),
    "causative_passive": ("かせられる", ["CAUSATIVE", "PASSIVE"]),
}
ICHIDAN = {
    "dictionary": ("る", []), "past": ("た", ["PAST"]),
    "negative": ("ない", ["NEGATIVE"]), "negative_past": ("なかった", ["NEGATIVE", "PAST"]),
    "conditional": ("れば", ["CONDITIONAL"]), "volitional": ("よう", ["VOLITIONAL"]),
    "causative": ("させる", ["CAUSATIVE"]), "passive": ("られる", ["PASSIVE"]),
    "causative_passive": ("させられる", ["CAUSATIVE", "PASSIVE"]),
}


def lemma_split(lemma: str, train=80, dev=10) -> str:
    bucket = int.from_bytes(hashlib.sha256(lemma.encode("utf-8")).digest()[:4], "big") % 100
    return "train" if bucket < train else ("dev" if bucket < train + dev else "test")


def generate_lemma(lemma: str, conjugation: str) -> List[Dict]:
    endings = GODAN_K if conjugation == "godan_k" else ICHIDAN if conjugation == "ichidan" else None
    if endings is None: raise ValueError("unsupported conjugation: %s" % conjugation)
    stem = lemma[:-1]
    result = []
    for paradigm, (ending, flags) in endings.items():
        text = stem + ending
        # Keep labels at a useful coarse granularity: lexical stem + complete
        # inflection chain. Natural teacher data later determines finer splits.
        spans = [{"start": 0, "end": len(stem), "atom_type": "VERB_STEM", "function": None, "inflections": []},
                 {"start": len(stem), "end": len(text), "atom_type": "AUXILIARY", "function": None, "inflections": flags}]
        result.append({"text": text, "source": "synthetic-morphology-v1", "document_id": "lemma:"+lemma,
                       "lemma": lemma, "split": lemma_split(lemma), "paradigm": paradigm,
                       "spans": spans, "synthetic_batch_max_fraction": 0.15})
    return result


def generate(lexicon: Iterable[Dict]) -> List[Dict]:
    return [record for entry in lexicon for record in generate_lemma(entry["lemma"], entry["conjugation"])]
