#!/usr/bin/env python3
"""Verify completed annotation shards and merge in source order."""
import argparse,hashlib,json
from pathlib import Path

def lines(path):
 with path.open("rb") as f:return sum(x.count(b"\n") for x in iter(lambda:f.read(8*1024*1024),b""))
def main():
 p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("--work-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--failures",type=Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text());report=[]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("wb") as out,a.failures.open("wb") as bad:
  for s in m["shards"]:
   idx=s["index"];success=a.work_dir/("part-%02d.teacher.jsonl"%idx);failure=a.work_dir/("part-%02d.failures.jsonl"%idx);checkpoint=success.with_suffix(success.suffix+".checkpoint.json")
   if not all(x.exists() for x in (success,failure,checkpoint)):raise SystemExit("incomplete shard %02d"%idx)
   state=json.loads(checkpoint.read_text());ok=lines(success);failed=lines(failure)
   if state["input_lines"]!=s["lines"] or ok!=state["successes"] or failed!=state["failures"] or ok+failed!=s["lines"]:raise SystemExit("count mismatch in shard %02d"%idx)
   with success.open("rb") as src:
    for chunk in iter(lambda:src.read(8*1024*1024),b""):out.write(chunk)
   with failure.open("rb") as src:
    for chunk in iter(lambda:src.read(8*1024*1024),b""):bad.write(chunk)
   report.append({"index":idx,"input":s["lines"],"success":ok,"failure":failed})
 print(json.dumps({"input":sum(x["input"] for x in report),"success":sum(x["success"] for x in report),"failure":sum(x["failure"] for x in report),"shards":report}))
if __name__=="__main__":main()
