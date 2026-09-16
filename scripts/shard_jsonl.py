#!/usr/bin/env python3
"""Split JSONL into contiguous, checksum-recorded shards."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--shards",type=int,required=True);p.add_argument("--total",type=int,required=True);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True);base=a.total//a.shards;extra=a.total%a.shards;manifest=[];seen=0
 with a.input.open("rb") as src:
  for index in range(a.shards):
   expected=base+(1 if index<extra else 0);path=a.output_dir/("part-%02d.jsonl"%index);digest=hashlib.sha256();count=0
   with path.open("wb") as out:
    for _ in range(expected):
     line=src.readline()
     if not line:raise SystemExit("input ended at %d, expected %d"%(seen,a.total))
     out.write(line);digest.update(line);count+=1;seen+=1
   manifest.append({"index":index,"path":str(path),"start_line":seen-count+1,"end_line":seen,"lines":count,"sha256":digest.hexdigest()})
  if src.readline():raise SystemExit("input has more than declared --total lines")
 (a.output_dir/"manifest.json").write_text(json.dumps({"source":str(a.input),"total":seen,"shards":manifest},indent=2)+"\n")
 print(json.dumps({"total":seen,"shards":a.shards,"manifest":str(a.output_dir/"manifest.json")}))
if __name__=="__main__":main()
