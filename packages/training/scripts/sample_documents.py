#!/usr/bin/env python3
"""Deterministically select documents by stable source/document hash."""
import argparse, hashlib, heapq, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("output",type=Path); p.add_argument("--count",type=int,required=True)
    args=p.parse_args(); heap=[]; invalid=0; total=0
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            total+=1
            try:
                row=json.loads(line); key=(str(row["source"])+"\0"+str(row["document_id"])).encode(); rank=hashlib.sha256(key).hexdigest()
            except (json.JSONDecodeError,KeyError,TypeError): invalid+=1; continue
            # Keep lexicographically smallest hashes with an inverted integer max-heap.
            item=(-int(rank,16),line)
            if len(heap)<args.count: heapq.heappush(heap,item)
            elif item>heap[0]: heapq.heapreplace(heap,item)
    selected=sorted(((-rank,line) for rank,line in heap),key=lambda x:x[0])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as out:
        for _,line in selected: out.write(line if line.endswith("\n") else line+"\n")
    print(json.dumps({"input":total,"invalid":invalid,"selected":len(selected),"output":str(args.output)}))

if __name__=="__main__": main()
