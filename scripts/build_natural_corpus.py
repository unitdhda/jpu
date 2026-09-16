#!/usr/bin/env python3
"""Build deterministic sentence subsets from normalized document JSONL.

Input documents use {source, document_id, text}. Acquisition-specific adapters
must preserve stable source IDs before invoking this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

# Permit the documented direct invocation from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.acquisition.extract import extract_sentences
from data.acquisition.dedup import hamming_distance, lsh_bands, simhash64


def parse_input(value):
    if "=" not in value: raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    source, path = value.split("=", 1)
    if not source or not path: raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    return source, Path(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", type=parse_input, required=True)
    p.add_argument("--quota", action="append", type=parse_input, required=True,
                   help="SOURCE=COUNT (COUNT is parsed as integer)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True,
                   help="temporary/restartable SQLite dedup database")
    p.add_argument("--near-duplicate-distance", type=int, default=3,
                   help="maximum 64-bit SimHash distance (default: 3)")
    p.add_argument("--near-duplicate-min-chars", type=int, default=80,
                   help="minimum document length for SimHash filtering")
    args = p.parse_args()
    quotas = {}
    try: quotas = {source: int(str(count)) for source, count in args.quota}
    except ValueError: p.error("quota count must be an integer")
    inputs = dict(args.input)
    missing = set(quotas) - set(inputs)
    if missing: p.error("quota sources missing inputs: " + ", ".join(sorted(missing)))
    for source, path in inputs.items():
        if not path.exists(): p.error("input does not exist: %s" % path)

    args.database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(args.database))
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("DROP TABLE IF EXISTS sentences; DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS lsh;"
                     "CREATE TABLE sentences (dedup_key TEXT PRIMARY KEY, source TEXT, sample_key TEXT, record TEXT);"
                     "CREATE TABLE documents (document_key TEXT PRIMARY KEY, fingerprint TEXT, source TEXT, document_id TEXT);"
                     "CREATE TABLE lsh (band INTEGER, band_value INTEGER, document_key TEXT);"
                     "CREATE INDEX lsh_lookup ON lsh(band, band_value);")
    counts = {"documents": {}, "candidate_sentences": {}, "duplicates": {}, "invalid_documents": {},
              "exact_duplicate_documents": {}, "near_duplicate_documents": {}}
    for source, path in args.input:  # command-line order defines duplicate precedence
        counts["documents"][source] = counts["candidate_sentences"][source] = 0
        counts["duplicates"][source] = counts["invalid_documents"][source] = 0
        counts["exact_duplicate_documents"][source] = counts["near_duplicate_documents"][source] = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    doc = json.loads(line); doc["source"] = source
                    if not isinstance(doc.get("text"), str) or not isinstance(doc.get("document_id"), str):
                        raise ValueError("invalid document")
                    document_key = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
                    if db.execute("SELECT 1 FROM documents WHERE document_key=?", (document_key,)).fetchone():
                        counts["exact_duplicate_documents"][source] += 1
                        continue
                    fingerprint = simhash64(doc["text"])
                    near_duplicate = False
                    if len(doc["text"]) >= args.near_duplicate_min_chars:
                        candidates = set()
                        for band, value in lsh_bands(fingerprint):
                            candidates.update(x[0] for x in db.execute(
                                "SELECT document_key FROM lsh WHERE band=? AND band_value=?", (band, value)))
                        for candidate in candidates:
                            prior = db.execute("SELECT fingerprint FROM documents WHERE document_key=?", (candidate,)).fetchone()
                            if prior and hamming_distance(fingerprint, int(prior[0], 16)) <= args.near_duplicate_distance:
                                near_duplicate = True; break
                    if near_duplicate:
                        counts["near_duplicate_documents"][source] += 1
                        continue
                    db.execute("INSERT INTO documents VALUES (?,?,?,?)", (document_key, "%016x" % fingerprint, source, doc["document_id"]))
                    if len(doc["text"]) >= args.near_duplicate_min_chars:
                        db.executemany("INSERT INTO lsh VALUES (?,?,?)", [(b, v, document_key) for b, v in lsh_bands(fingerprint)])
                    records = extract_sentences(doc)
                    counts["documents"][source] += 1
                    for record in records:
                        counts["candidate_sentences"][source] += 1
                        sample_key = hashlib.sha256((source+"\0"+record["document_id"]+"\0"+str(record["sentence_index"])).encode()).hexdigest()
                        before = db.total_changes
                        db.execute("INSERT OR IGNORE INTO sentences VALUES (?,?,?,?)",
                                   (record["dedup_key"], source, sample_key, json.dumps(record, ensure_ascii=False)))
                        if db.total_changes == before: counts["duplicates"][source] += 1
                except (ValueError, json.JSONDecodeError, TypeError):
                    counts["invalid_documents"][source] += 1
        db.commit()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = {}
    with args.output.open("w", encoding="utf-8") as out:
        for source in sorted(quotas):
            rows = db.execute("SELECT record FROM sentences WHERE source=? ORDER BY sample_key LIMIT ?", (source, quotas[source]))
            selected[source] = 0
            for (record,) in rows:
                out.write(record + "\n"); selected[source] += 1
    report = {"requested": quotas, "selected": selected, **counts,
              "shortfall": {s: quotas[s]-selected[s] for s in quotas}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    db.close()
    if any(report["shortfall"].values()):
        print("one or more source quotas were not met; see report", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__": raise SystemExit(main())
