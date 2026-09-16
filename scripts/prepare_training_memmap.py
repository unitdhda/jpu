#!/usr/bin/env python3
"""Convert canonical JSONL into fixed-length memory-mapped training arrays."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from data.schemas.canonical import ATOM_TYPES, BUNSETSU_ROLES, INFLECTIONS, PARTICLE_FUNCTIONS

BUCKETS = (32, 64, 128)
FIELDS = {
    "codepoints": (np.uint16, ()), "bigrams": (np.uint16, ()), "categorical": (np.uint8, (22,)),
    "mask": (np.uint8, ()), "boundaries": (np.uint8, (5,)), "atom": (np.uint8, ()),
    "function": (np.uint8, ()), "function_mask": (np.uint8, ()),
    "inflection": (np.uint8, (10,)), "inflection_mask": (np.uint8, ()), "role": (np.uint8, ()),
}
ATOM_INDEX = {x: i for i, x in enumerate(ATOM_TYPES)}
FUNCTION_INDEX = {x: i for i, x in enumerate(PARTICLE_FUNCTIONS)}
INFL_INDEX = {x: i for i, x in enumerate(INFLECTIONS)}
ROLE_INDEX = {x: i for i, x in enumerate(BUNSETSU_ROLES)}

def mix64(value):
    value &= 0xffffffffffffffff; value ^= value >> 30; value = (value * 0xBF58476D1CE4E5B9) & 0xffffffffffffffff
    value ^= value >> 27; value = (value * 0x94D049BB133111EB) & 0xffffffffffffffff
    return (value ^ (value >> 31)) & 0xffffffffffffffff

def hash_bucket(value, buckets): return mix64(value) & (buckets - 1)
def split(record):
    key = str(record["source"]) + "\0" + str(record["document_id"])
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 100
    return "train" if bucket < 85 else "dev" if bucket < 95 else "test"
def length_bucket(n): return next((b for b in BUCKETS if n <= b), None)
def script(ch):
    cp = ord(ch)
    if 0x3040 <= cp <= 0x309f: return 0
    if 0x30a0 <= cp <= 0x30ff: return 1
    if 0x4e00 <= cp <= 0x9fff or 0x3400 <= cp <= 0x4dbf: return 2
    if ch.isascii() and ch.isalpha(): return 3
    if ch.isdigit(): return 4
    if ch.isspace(): return 7
    if __import__('unicodedata').category(ch).startswith('P'): return 5
    if __import__('unicodedata').category(ch).startswith('S'): return 6
    return 8

def encode_arrays(text, buckets):
    chars = list(text); n = len(chars); cp = np.empty(n, np.uint16); bg = np.empty(n, np.uint16); cat = np.zeros((n, 22), np.uint8); previous = -1
    for i, ch in enumerate(chars):
        current = script(ch); cp[i] = hash_bucket(ord(ch), buckets); right = ord(chars[i + 1]) if i + 1 < n else 0
        bg[i] = hash_bucket((ord(ch) << 21) ^ right ^ 0x9E3779B9, buckets); cat[i, current] = 1
        if previous >= 0 and previous != current: cat[i, 9] = 1
        if ch in 'ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ': cat[i, 10] = 1
        if ch in 'ーｰ': cat[i, 11] = 1
        if ch in '々〆ヽヾゝゞー': cat[i, 12] = 1
        cat[i, 13 + (1 if ch in '。！？!?．' else 2 if ch in '「『（【〔〈《' else 3 if ch in '」』）】〕〉》' else 4 if ch in '、，,' else 0)] = 1
        digit = 1 if '0' <= ch <= '9' else 2 if '０' <= ch <= '９' else 3 if current == 4 else 0
        cat[i, 18 + digit] = 1; previous = current
    return cp, bg, cat

def target_arrays(record, n, bucket):
    boundary = np.zeros((bucket, 5), np.uint8)
    for level, column in zip(('a', 'b', 'bunsetsu', 'clause', 'sentence'), range(5)):
        for end in record['boundaries'][level]: boundary[end - 1, column] = 1
    atom = np.zeros(bucket, np.uint8); function = np.full(bucket, FUNCTION_INDEX['UNKNOWN'], np.uint8); function_mask = np.zeros(bucket, np.uint8)
    inflection = np.zeros((bucket, 10), np.uint8); inflection_mask = np.zeros(bucket, np.uint8); role = np.zeros(bucket, np.uint8)
    for span in record['spans']:
        start, end = span['start'], span['end']; kind = span['atom_type']; atom[start:end] = ATOM_INDEX[kind]
        if kind == 'PARTICLE': function_mask[start:end] = 1; function[start:end] = FUNCTION_INDEX[span.get('function') or 'UNKNOWN']
        if kind in {'VERB_STEM', 'AUXILIARY', 'COPULA', 'ADJECTIVE'}:
            inflection_mask[start:end] = 1
            for label in span.get('inflections', []): inflection[start:end, INFL_INDEX[label]] = 1
    for span in record['bunsetsu']: role[span['start']:span['end']] = ROLE_INDEX[span['role']]
    mask = np.zeros(bucket, np.uint8); mask[:n] = 1
    return {'boundaries': boundary, 'atom': atom, 'function': function, 'function_mask': function_mask, 'inflection': inflection, 'inflection_mask': inflection_mask, 'role': role, 'mask': mask}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data', type=Path); ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--models', nargs='+', default=['50k', '150k']); args = ap.parse_args()
    counts = {(s, b): 0 for s in ('train', 'dev', 'test') for b in BUCKETS}
    started = time.monotonic(); last_report = started; lines = 0
    with args.data.open(encoding='utf-8') as f:
        for line in f:
            lines += 1; r = json.loads(line); b = length_bucket(len(r['text']))
            if b: counts[(split(r), b)] += 1
            now = time.monotonic()
            if now - last_report >= 15:
                print(json.dumps({'phase': 'count', 'lines': lines, 'rate_per_sec': lines / max(0.001, now - started)}), flush=True); last_report = now
    print(json.dumps({'phase': 'count_complete', 'counts': {f'{s}-{b}': n for (s, b), n in counts.items() if n}}), flush=True)
    for model_name in args.models:
        cp_buckets = 4096 if model_name == '50k' else 8192; root = args.output_dir / model_name; root.mkdir(parents=True, exist_ok=True); maps = {}; cursors = {}
        for s in ('train', 'dev', 'test'):
            for b in BUCKETS:
                if not counts[(s, b)]: continue
                cursors[(s, b)] = 0
                for field, (dtype, tail) in FIELDS.items(): maps[(s, b, field)] = np.lib.format.open_memmap(root / f'{s}-{b}-{field}.npy', mode='w+', dtype=dtype, shape=(counts[(s, b)], b, *tail))
        started = time.monotonic(); last_report = started
        with args.data.open(encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                r = json.loads(line); n = len(r['text']); b = length_bucket(n)
                if b is None: continue
                s = split(r); i = cursors[(s, b)]; cursors[(s, b)] += 1; cp, bg, cat = encode_arrays(r['text'], cp_buckets)
                maps[(s, b, 'codepoints')][i, :n] = cp; maps[(s, b, 'bigrams')][i, :n] = bg; maps[(s, b, 'categorical')][i, :n] = cat
                for field, values in target_arrays(r, n, b).items(): maps[(s, b, field)][i] = values
                now = time.monotonic()
                if now - last_report >= 15:
                    print(json.dumps({'phase': 'encode', 'model': model_name, 'lines': line_number, 'rate_per_sec': line_number / max(0.001, now - started)}), flush=True); last_report = now
        for mmap in maps.values(): mmap.flush()
        (root / 'metadata.json').write_text(json.dumps({'model': model_name, 'counts': {f'{s}-{b}': counts[(s, b)] for s in ('train', 'dev', 'test') for b in BUCKETS}}, indent=2) + '\n')

if __name__ == '__main__': main()
