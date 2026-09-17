"""Explicit, versioned teacher-label mappings for Phase 1."""

from typing import Iterable, Mapping, Optional, Sequence, Set

MAPPING_VERSION = "v1"

# Sudachi POS is a six-field tuple. Mapping is deliberately driven by the
# documented first/second fields rather than fuzzy matching label strings.
SUDACHI_POS1 = {
    "名詞": "NOUN_LIKE", "代名詞": "NOUN_LIKE", "動詞": "VERB_STEM",
    "形容詞": "ADJECTIVE", "形状詞": "ADJECTIVE", "副詞": "ADVERB",
    "助詞": "PARTICLE", "助動詞": "AUXILIARY", "接頭辞": "PREFIX_SUFFIX",
    "接尾辞": "PREFIX_SUFFIX", "連体詞": "OTHER", "接続詞": "OTHER",
    "感動詞": "OTHER", "補助記号": "PUNCT_SYMBOL", "記号": "PUNCT_SYMBOL",
    "空白": "OTHER", "UNK": "OTHER",
}
SUDACHI_NOUN_POS2 = {"固有名詞": "PROPER_LIKE", "数詞": "NUMBER"}

PARTICLE_SURFACE = {
    "は": "TOPIC", "も": "TOPIC", "が": "SUBJECT_CASE", "を": "OBJECT",
    "へ": "DIRECTION", "まで": "DIRECTION", "と": "COMITATIVE",
    "より": "OTHER", "の": "GENITIVE", "って": "QUOTATIVE",
    "て": "CONNECTIVE", "で": "CONNECTIVE", "ながら": "CONNECTIVE",
}
# Context-free labels only where the surface is not strongly ambiguous.
PARTICLE_POS2 = {
    "係助詞": "TOPIC", "格助詞": "UNKNOWN", "接続助詞": "CONNECTIVE",
    "終助詞": "OTHER", "副助詞": "OTHER", "準体助詞": "OTHER",
}

# Exact normalized-form labels supplied by Sudachi/UniDic. Surface fallback is
# separately explicit, making uncertain labels auditable.
INFLECTION_NORMALIZED = {
    "ない": {"NEGATIVE"}, "ぬ": {"NEGATIVE"}, "ん": {"NEGATIVE"},
    "た": {"PAST"}, "だ": {"PAST"}, "ます": {"POLITE"},
    "れる": {"PASSIVE"}, "られる": {"PASSIVE"}, "せる": {"CAUSATIVE"},
    "させる": {"CAUSATIVE"}, "たい": {"ASPECTUAL"}, "いる": {"ASPECTUAL"},
}
INFLECTION_FORM = {
    "仮定形": {"CONDITIONAL"}, "仮定形-一般": {"CONDITIONAL"},
    "仮定形-融合": {"CONDITIONAL"}, "意志推量形": {"VOLITIONAL"},
    "連用形-促音便": {"TE_FORM"}, "連用形-イ音便": {"TE_FORM"},
    "連用形-撥音便": {"TE_FORM"}, "連用形-ウ音便": {"TE_FORM"},
}
INFLECTION_SURFACE = {
    "なかっ": {"NEGATIVE"}, "なかった": {"NEGATIVE", "PAST"},
    "ました": {"POLITE", "PAST"}, "ません": {"POLITE", "NEGATIVE"},
    "ませんでした": {"POLITE", "NEGATIVE", "PAST"},
}

# GiNZA/Universal Dependencies fields used for conservative role derivation.
DEPENDENCY_TO_ROLE = {
    "punct": "PUNCT", "root": "PREDICATE", "advcl": "CONNECTIVE",
    "acl": "MODIFIER", "advmod": "ADVERBIAL", "obl": "CASED_NOMINAL",
    "nsubj": "CASED_NOMINAL", "obj": "CASED_NOMINAL", "iobj": "CASED_NOMINAL",
    "nmod": "MODIFIER", "compound": "MODIFIER", "cc": "CONNECTIVE",
    "conj": "CONNECTIVE",
}


def map_sudachi_pos(pos: Sequence[str]) -> str:
    if not pos:
        return "OTHER"
    if pos[0] == "名詞" and len(pos) > 1 and pos[1] in SUDACHI_NOUN_POS2:
        return SUDACHI_NOUN_POS2[pos[1]]
    if pos[0] == "助動詞" and len(pos) > 4 and pos[4] in {"助動詞-ダ", "助動詞-デス"}:
        return "COPULA"
    return SUDACHI_POS1.get(pos[0], "OTHER")


def map_particle(surface: str, pos: Sequence[str], context: Optional[Mapping] = None) -> Optional[str]:
    if not pos or pos[0] != "助詞":
        return None
    # Explicit contextual disambiguation for common ambiguous particles.
    if context:
        case = context.get("case")
        if surface == "に" and case in {"LOC", "DIR", "TIME"}:
            return {"LOC": "LOCATION", "DIR": "DIRECTION", "TIME": "TIME"}[case]
        if surface == "と" and case == "QUOTE":
            return "QUOTATIVE"
        if surface == "で" and case == "LOC":
            return "LOCATION"
    if surface in {"に", "で"}:
        return "UNKNOWN"
    return PARTICLE_SURFACE.get(surface, PARTICLE_POS2.get(pos[1] if len(pos) > 1 else "", "UNKNOWN"))


def map_inflections(surface: str, normalized_form: str = "", inflection_form: str = "") -> list:
    flags: Set[str] = set(INFLECTION_NORMALIZED.get(normalized_form, set()))
    flags.update(INFLECTION_FORM.get(inflection_form, set()))
    flags.update(INFLECTION_SURFACE.get(surface, set()))
    return sorted(flags)


def map_bunsetsu_role(dependency: str, particle_function: Optional[str] = None,
                       is_predicate: bool = False) -> str:
    if particle_function == "TOPIC":
        return "TOPIC"
    if is_predicate:
        return "PREDICATE"
    return DEPENDENCY_TO_ROLE.get(dependency.lower(), "UNKNOWN")
