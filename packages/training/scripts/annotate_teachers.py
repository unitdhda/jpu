#!/usr/bin/env python3
"""Offline Sudachi A/B + GiNZA exporter with checkpointed progress."""
import argparse, importlib.metadata, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sudachi_tokens(tokenizer_obj, mode, text):
    rows=[]; adjustments=[]
    for index,m in enumerate(tokenizer_obj.tokenize(text, mode)):
        if m.begin() == m.end() and not m.surface():
            adjustments.append({"teacher":"sudachi_"+str(mode).rsplit(".",1)[-1].lower(),"token":index,
                                "offset":m.begin(),"reason":"drop_zero_width_empty_morpheme"})
            continue
        if m.begin() == m.end() or not m.surface(): raise ValueError("invalid Sudachi morpheme at %d"%index)
        pos=list(m.part_of_speech())
        rows.append({"surface":m.surface(),"start":m.begin(),"end":m.end(),"pos":pos,
                     "normalized_form":m.normalized_form(),
                     "inflection_form":pos[5] if len(pos)>5 and pos[5]!="*" else ""})
    return rows,adjustments


def ginza_chunks(doc,ginza,text):
    rows=[];adjustments=[];spans=list(ginza.bunsetu_spans(doc))
    for index,span in enumerate(spans):
        members=list(span);head=next((t for t in members if t.dep_.lower()=="root"),None)
        if head is None: head=next((t for t in members if t.head.i<span.start or t.head.i>=span.end),members[0])
        surfaces={t.text for t in members if t.pos_ in {"ADP","PART"}}
        particle_function="TOPIC" if surfaces & {"は","も"} else None
        start=0 if index==0 else rows[-1]["end"]
        end=spans[index+1].start_char if index+1<len(spans) else len(text)
        if start!=span.start_char or end!=span.end_char:
            gap=text[span.end_char:end] if end>span.end_char else text[start:span.start_char]
            if gap and not gap.isspace(): raise ValueError("GiNZA uncovered non-whitespace text %r"%gap)
            adjustments.append({"chunk":index,"teacher_start":span.start_char,"teacher_end":span.end_char,
                                "raw_start":start,"raw_end":end,"reason":"attach_skipped_whitespace"})
        rows.append({"surface":text[start:end],"start":start,"end":end,"dependency":head.dep_,
                     "is_predicate":head.pos_ in {"VERB","ADJ"} or head.dep_.lower()=="root",
                     "particle_function":particle_function})
    return rows,adjustments


def count_lines(path):
    with path.open("rb") as source:return sum(chunk.count(b"\n") for chunk in iter(lambda:source.read(8*1024*1024),b""))


def load_checkpoint(path,output,failures):
    if not path.exists(): return {"input_lines":0,"output_offset":0,"failure_offset":0,"successes":0,"failures":0}
    state=json.loads(path.read_text(encoding="utf8"))
    for target,key in ((output,"output_offset"),(failures,"failure_offset")):
        if not target.exists(): raise SystemExit("checkpoint exists but output is missing: %s"%target)
        with target.open("r+b") as handle:handle.truncate(state[key])
    return state


def save_checkpoint(path,state):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(state,indent=2)+"\n",encoding="utf8");os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--failures",type=Path,required=True);p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--processes",type=int,default=1,help="spaCy worker processes (benchmark before increasing)")
    p.add_argument("--checkpoint",type=Path);p.add_argument("--log-every",type=int,default=1000)
    p.add_argument("--total",type=int,help="known input count; otherwise count lines before loading teachers")
    args=p.parse_args();checkpoint=args.checkpoint or args.output.with_suffix(args.output.suffix+".checkpoint.json")
    total=args.total if args.total is not None else count_lines(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.failures.parent.mkdir(parents=True,exist_ok=True);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    state=load_checkpoint(checkpoint,args.output,args.failures);resuming=state["input_lines"]>0
    print("teacher annotation: total=%d resume_at=%d checkpoint=%s"%(total,state["input_lines"],checkpoint),file=sys.stderr,flush=True)
    from sudachipy import dictionary,tokenizer
    import spacy,ginza
    sudachi=dictionary.Dictionary().create();nlp=spacy.load("ja_ginza",disable=["ner"])
    versions={name:importlib.metadata.version(name) for name in ("SudachiPy","SudachiDict-core","spacy","ginza","ja-ginza")}
    started=time.monotonic();run_start=state["input_lines"];last_logged=state["input_lines"]
    with args.input.open(encoding="utf8") as source,args.output.open("a" if resuming else "w",encoding="utf8") as out,args.failures.open("a" if resuming else "w",encoding="utf8") as bad:
        for _ in range(state["input_lines"]):
            if not source.readline():raise SystemExit("checkpoint exceeds input length")
        while True:
            raw=[]
            for _ in range(args.batch_size):
                line=source.readline()
                if not line:break
                raw.append((state["input_lines"]+len(raw)+1,line))
            if not raw:break
            valid=[]
            for line_number,line in raw:
                try:
                    row=json.loads(line);text=row["text"]
                    if not isinstance(text,str):raise ValueError("text is not a string")
                    valid.append((line_number,row))
                except (json.JSONDecodeError,KeyError,ValueError,TypeError) as exc:
                    bad.write(json.dumps({"line":line_number,"reason":"invalid_input","detail":str(exc)})+"\n");state["failures"]+=1
            docs=nlp.pipe((row["text"] for _,row in valid),batch_size=args.batch_size,n_process=args.processes)
            for (line_number,row),doc in zip(valid,docs):
                try:
                    text=row["text"];a,aa=sudachi_tokens(sudachi,tokenizer.Tokenizer.SplitMode.A,text);b,ba=sudachi_tokens(sudachi,tokenizer.Tokenizer.SplitMode.B,text);chunks,ga=ginza_chunks(doc,ginza,text)
                    result={"text":text,"source":row["source"],"document_id":row["document_id"],"sudachi_a":a,"sudachi_b":b,
                            "ginza_bunsetsu":chunks,"clause_boundaries":[len(text)],"alignment_adjustments":aa+ba+ga,"teacher_versions":versions}
                    out.write(json.dumps(result,ensure_ascii=False)+"\n");state["successes"]+=1
                except Exception as exc:
                    bad.write(json.dumps({"line":line_number,"source":row.get("source"),"document_id":row.get("document_id"),"reason":"annotation_failure","detail":"%s: %s"%(type(exc).__name__,exc)},ensure_ascii=False)+"\n");state["failures"]+=1
            state["input_lines"]+=len(raw);out.flush();bad.flush();os.fsync(out.fileno());os.fsync(bad.fileno())
            state["output_offset"]=out.buffer.tell();state["failure_offset"]=bad.buffer.tell();save_checkpoint(checkpoint,state)
            if state["input_lines"]-last_logged>=args.log_every or state["input_lines"]==total:
                elapsed=time.monotonic()-started;done=state["input_lines"]
                print("progress %d/%d (%.2f%%) success=%d failure=%d elapsed=%.1fs rate=%.1f sent/s eta=%.1fs"%(done,total,100*done/total if total else 100,state["successes"],state["failures"],elapsed,(done-run_start)/elapsed if elapsed else 0,(total-done)*elapsed/max(1,done-run_start)),file=sys.stderr,flush=True);last_logged=done
    print("teacher annotation complete: success=%d failure=%d"%(state["successes"],state["failures"]),file=sys.stderr,flush=True)

if __name__=="__main__":main()
