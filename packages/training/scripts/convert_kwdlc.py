#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.annotation.gold_kwdlc import parse_knp,convert_sentence

def main():
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("split_ids",type=Path);p.add_argument("output",type=Path);p.add_argument("--failures",type=Path,required=True);a=p.parse_args()
 ids={x.strip() for x in a.split_ids.read_text().splitlines() if x.strip()}; files={x.stem:x for x in (a.root/"knp").glob("**/*.knp")};missing=ids-set(files)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.failures.parent.mkdir(parents=True,exist_ok=True);ok=bad=0
 with a.output.open("w",encoding="utf8") as out,a.failures.open("w",encoding="utf8") as fail:
  for doc_id in sorted(ids&set(files)):
   with files[doc_id].open(encoding="utf8") as src:
    for sid,tokens,bs,buns in parse_knp(src):
     try:out.write(json.dumps(convert_sentence(sid,tokens,bs,buns,doc_id),ensure_ascii=False)+"\n");ok+=1
     except Exception as e:fail.write(json.dumps({"document_id":doc_id,"sentence_id":sid,"reason":type(e).__name__,"detail":str(e)},ensure_ascii=False)+"\n");bad+=1
 print(json.dumps({"documents_requested":len(ids),"documents_missing":len(missing),"converted":ok,"failed":bad}))
 return 0 if not missing and not bad else 2
if __name__=="__main__":raise SystemExit(main())
