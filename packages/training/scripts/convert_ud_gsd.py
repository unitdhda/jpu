#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.annotation.gold_ud import convert_sentence,parse_conllu

def main():
 p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("output",type=Path);p.add_argument("--failures",type=Path,required=True);a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.failures.parent.mkdir(parents=True,exist_ok=True)
 ok=bad=0
 with a.input.open(encoding="utf8") as src,a.output.open("w",encoding="utf8") as out,a.failures.open("w",encoding="utf8") as failures:
  for comments,tokens in parse_conllu(src):
   try:out.write(json.dumps(convert_sentence(comments,tokens),ensure_ascii=False)+"\n");ok+=1
   except Exception as e:failures.write(json.dumps({"sent_id":comments.get("sent_id"),"reason":type(e).__name__,"detail":str(e)},ensure_ascii=False)+"\n");bad+=1
 print(json.dumps({"converted":ok,"failed":bad}))
 return 0 if not bad else 2
if __name__=="__main__":raise SystemExit(main())
