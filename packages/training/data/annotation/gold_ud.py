"""Converter for pinned UD Japanese GSD CoNLL-U releases."""
from typing import Dict, Iterable, Iterator, List
from data.annotation.mappings import map_bunsetsu_role
from data.schemas.canonical import make_record, validate_record

UPOS_ATOM={"NOUN":"NOUN_LIKE","PRON":"NOUN_LIKE","PROPN":"PROPER_LIKE","VERB":"VERB_STEM",
 "ADJ":"ADJECTIVE","ADV":"ADVERB","ADP":"PARTICLE","SCONJ":"PARTICLE","CCONJ":"PARTICLE",
 "PART":"PARTICLE","AUX":"AUXILIARY","NUM":"NUMBER","PUNCT":"PUNCT_SYMBOL","SYM":"PUNCT_SYMBOL"}
SURFACE_FUNCTION={"は":"TOPIC","が":"SUBJECT_CASE","を":"OBJECT","へ":"DIRECTION","の":"GENITIVE",
 "て":"CONNECTIVE","で":"UNKNOWN","に":"UNKNOWN","と":"UNKNOWN"}

def misc_dict(value):
    return {x.split("=",1)[0]:x.split("=",1)[1] for x in value.split("|") if "=" in x}

def parse_conllu(lines: Iterable[str]):
    comments={}; tokens=[]
    for raw in lines:
        line=raw.rstrip("\n")
        if not line:
            if tokens: yield comments,tokens
            comments={};tokens=[];continue
        if line.startswith("#"):
            if " = " in line:
                k,v=line[2:].split(" = ",1);comments[k]=v
            continue
        fields=line.split("\t")
        if len(fields)==10 and fields[0].isdigit(): tokens.append(fields)
    if tokens: yield comments,tokens

def convert_sentence(comments,tokens,source="ud-ja-gsd"):
    text=comments.get("text") or "".join(t[1] for t in tokens)
    spans=[];cursor=0;starts_b=[];starts_l=[]
    for i,t in enumerate(tokens):
        surface=t[1]
        if text[cursor:cursor+len(surface)]!=surface: raise ValueError("token/raw mismatch at %d"%cursor)
        misc=misc_dict(t[9]); start=cursor;cursor+=len(surface)
        atom=UPOS_ATOM.get(t[3],"OTHER")
        if atom=="AUXILIARY" and "助動詞-ダ" in t[4]: atom="COPULA"
        infl=[]; feats=t[5]
        if "Tense=Past" in feats or "助動詞-タ" in t[4]: infl.append("PAST")
        if "Polarity=Neg" in feats or "助動詞-ナイ" in t[4] or "助動詞-ヌ" in t[4]: infl.append("NEGATIVE")
        if "助動詞-マス" in t[4]: infl.append("POLITE")
        spans.append({"start":start,"end":cursor,"atom_type":atom,
                      "function":SURFACE_FUNCTION.get(surface,"UNKNOWN") if atom=="PARTICLE" else None,
                      "inflections":sorted(set(infl))})
        if misc.get("LUWBILabel")=="B": starts_l.append(i)
        if misc.get("BunsetuBILabel")=="B": starts_b.append(i)
    if cursor!=len(text): raise ValueError("tokens do not cover raw text")
    def grouped(starts):
        starts=starts or [0]; result=[]
        for j,start_i in enumerate(starts):
            end_i=starts[j+1] if j+1<len(starts) else len(tokens)
            result.append((spans[start_i]["start"],spans[end_i-1]["end"],start_i,end_i))
        return result
    b_spans=[{"start":a,"end":b} for a,b,_,_ in grouped(starts_l)]
    chunks=[]
    for a,b,i,j in grouped(starts_b):
        head=next((t for t in tokens[i:j] if t[7].lower()=="root"),tokens[i])
        topic=any(t[1] in {"は","も"} for t in tokens[i:j])
        role=map_bunsetsu_role(head[7],"TOPIC" if topic else None,head[3] in {"VERB","ADJ"})
        chunks.append({"start":a,"end":b,"role":role})
    record=make_record(text,source,comments.get("newdoc id",comments.get("sent_id","")),
      {"a":[s["end"] for s in spans],"b":[s["end"] for s in b_spans],
       "bunsetsu":[s["end"] for s in chunks],"clause":[len(text)],"sentence":[len(text)]},
      spans,chunks,b_spans,{"gold_mapping_version":"ud-ja-gsd.v1","sent_id":comments.get("sent_id")})
    errors=validate_record(record)
    if errors: raise ValueError(";".join(errors))
    return record
