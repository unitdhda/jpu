#!/usr/bin/env python3
"""Compile teacher JSONL to canonical JSONL, failures, and statistics."""
import argparse, json
from pathlib import Path
import sys

# Permit the documented direct invocation from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.annotation.compiler import compile_example
from data.annotation.stats import CorpusStats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.failures, args.stats): path.parent.mkdir(parents=True, exist_ok=True)
    stats = CorpusStats()
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as good, args.failures.open("w", encoding="utf-8") as bad:
        for line_number, line in enumerate(source, 1):
            try: raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failure = {"reason": "invalid_json", "teacher": "input", "detail": "%d: %s" % (line_number, exc)}
                stats.add_failure(failure); bad.write(json.dumps(failure, ensure_ascii=False)+"\n"); continue
            record, failure = compile_example(raw)
            if failure:
                stats.add_failure(failure); bad.write(json.dumps(failure, ensure_ascii=False)+"\n")
            else:
                stats.add_success(record); good.write(json.dumps(record, ensure_ascii=False)+"\n")
    args.stats.write_text(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__": main()
