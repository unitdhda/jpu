#!/usr/bin/env python3
"""Normalize production source exports into document JSONL.

FineWeb2 input is Parquet and requires the optional ``pyarrow`` package.
Wikipedia input is JSONL from WikiExtractor (``id``, ``text``, optionally
``revid``/``url``/``title``). This adapter never downloads data.
"""
import argparse
import glob
import json
from pathlib import Path
import sys


def write_record(out, source, document_id, text, provenance):
    if not isinstance(document_id, (str, int)) or not isinstance(text, str) or not text:
        return False
    out.write(json.dumps({"source":source, "document_id":str(document_id), "text":text,
                          "provenance":provenance}, ensure_ascii=False)+"\n")
    return True


def normalize_fineweb(paths, out):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("FineWeb2 Parquet conversion requires optional dependency pyarrow; install it in a controlled acquisition environment")
    written = invalid = 0
    columns = ["id", "text", "url", "dump", "date", "file_path", "language", "language_score", "language_script"]
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        required = {"id", "text"}
        if not required.issubset(available):
            raise SystemExit("%s lacks required columns %s" % (path, sorted(required - available)))
        selected = [x for x in columns if x in available]
        for batch in parquet.iter_batches(batch_size=1024, columns=selected):
            for row in batch.to_pylist():
                provenance = {k: row.get(k) for k in selected if k not in {"id", "text"}}
                if write_record(out, "fineweb2-ja", row.get("id"), row.get("text"), provenance): written += 1
                else: invalid += 1
    return written, invalid


def normalize_wikiextractor(paths, out):
    written = invalid = 0
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                try: row = json.loads(line)
                except json.JSONDecodeError: invalid += 1; continue
                provenance = {k: row.get(k) for k in ("title", "url", "revid") if k in row}
                if write_record(out, "jawiki", row.get("id"), row.get("text"), provenance): written += 1
                else: invalid += 1
    return written, invalid


def main():
    p=argparse.ArgumentParser(); p.add_argument("source",choices=("fineweb2-ja","jawiki")); p.add_argument("inputs",nargs="+")
    p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); args=p.parse_args()
    paths=sorted({item for pattern in args.inputs for item in glob.glob(pattern, recursive=True) if Path(item).is_file()})
    if not paths: p.error("no input files matched")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as out:
        written,invalid=(normalize_fineweb(paths,out) if args.source=="fineweb2-ja" else normalize_wikiextractor(paths,out))
    args.report.write_text(json.dumps({"source":args.source,"input_files":paths,"written":written,"invalid":invalid},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0 if written else 2

if __name__=="__main__": raise SystemExit(main())
