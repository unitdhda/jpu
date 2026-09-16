"""Deterministic document fingerprints for conservative near-deduplication."""
import hashlib
import re
from typing import Iterable


def comparison_text(text: str) -> str:
    """Normalize only for comparison; never use this as emitted training text."""
    return re.sub(r"\s+", " ", text).strip()


def simhash64(text: str, ngram: int = 3) -> int:
    """Character-ngram SimHash suitable for candidate duplicate detection."""
    value = comparison_text(text)
    if not value:
        return 0
    features = (value[i:i + ngram] for i in range(max(1, len(value) - ngram + 1)))
    weights = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8,
                                 person=b"jplex-v1").digest()
        bits = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if bits & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")


def lsh_bands(fingerprint: int, width: int = 16):
    """Return four exact-band keys; distance <=3 guarantees a shared band."""
    mask = (1 << width) - 1
    return [(band, (fingerprint >> (band * width)) & mask) for band in range(64 // width)]
