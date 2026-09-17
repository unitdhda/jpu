"""Raw-document ingestion and exact-offset Japanese sentence extraction."""
import hashlib
import re
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

# Closing punctuation/quotes stay with the sentence. Newlines also delimit
# fragments, but are not emitted. Text itself is never Unicode-normalized.
_SENTENCE_END = set("。！？!?．")
_CLOSERS = set("」』）】〕〉》〗〙〛”)’\"'")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_for_document(source: str, document_id: str,
                       train: int = 98, dev: int = 1) -> str:
    """Stable split assignment; all sentences in a document stay together."""
    bucket = int(stable_hash(source + "\0" + document_id)[:8], 16) % 100
    return "train" if bucket < train else ("dev" if bucket < train + dev else "test")


def dedup_key(text: str) -> str:
    """Conservative comparison key only; emitted text remains byte-for-byte."""
    comparison = re.sub(r"\s+", " ", text).strip()
    return stable_hash(comparison)


def extract_sentences(document: Mapping, min_length: int = 4,
                      max_length: int = 128) -> Iterator[Dict]:
    """Yield raw slices with codepoint offsets and no orthographic rewriting."""
    source = document.get("source")
    document_id = document.get("document_id")
    text = document.get("text")
    if not all(isinstance(x, str) and x for x in (source, document_id, text)):
        raise ValueError("document requires non-empty string source, document_id, and text")
    start = 0
    i = 0
    sentence_index = 0
    while i < len(text):
        boundary = text[i] in _SENTENCE_END or text[i] in "\r\n"
        end = i + 1
        if text[i] in _SENTENCE_END:
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
        if boundary:
            raw_start, raw_end = start, end
            # Trim surrounding whitespace by moving offsets, never by changing
            # characters inside the selected slice.
            while raw_start < raw_end and text[raw_start].isspace(): raw_start += 1
            while raw_end > raw_start and text[raw_end - 1].isspace(): raw_end -= 1
            sentence = text[raw_start:raw_end]
            if min_length <= len(sentence) <= max_length:
                yield {"text": sentence, "source": source,
                       "document_id": document_id, "sentence_index": sentence_index,
                       "document_start": raw_start, "document_end": raw_end,
                       "split": split_for_document(source, document_id),
                       "dedup_key": dedup_key(sentence)}
                sentence_index += 1
            start = end
        i = max(i + 1, end)
    if start < len(text):
        raw_start, raw_end = start, len(text)
        while raw_start < raw_end and text[raw_start].isspace(): raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace(): raw_end -= 1
        sentence = text[raw_start:raw_end]
        if min_length <= len(sentence) <= max_length:
            yield {"text": sentence, "source": source,
                   "document_id": document_id, "sentence_index": sentence_index,
                   "document_start": raw_start, "document_end": raw_end,
                   "split": split_for_document(source, document_id),
                   "dedup_key": dedup_key(sentence)}
