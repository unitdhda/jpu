"""Strict KWDLC KNP-format converter for independent evaluation."""
from pathlib import Path
from data.schemas.canonical import make_record,validate_record

POS_MAP={"名詞":"NOUN_LIKE","指示詞":"NOUN_LIKE","動詞":"VERB_STEM","形容詞":"ADJECTIVE","副詞":"ADVERB",
 "助詞":"PARTICLE","助動詞":"AUXILIARY","判定詞":"COPULA","接頭辞":"PREFIX_SUFFIX","接尾辞":"PREFIX_SUFFIX",
 "特殊":"PUNCT_SYMBOL","未定義語":"OTHER","連体詞":"OTHER","接続詞":"OTHER","感動詞":"OTHER"}
FUNCTION={"は":"TOPIC","が":"SUBJECT_CASE","を":"OBJECT","へ":"DIRECTION","から":"DIRECTION","まで":"DIRECTION",
 "と":"COMITATIVE","の":"GENITIVE","て":"CONNECTIVE","で":"UNKNOWN","に":"UNKNOWN"}

def parse_knp(lines):
    sentence=None; tokens=[]; b_starts=[]; bun_starts=[]
    for raw in lines:
        line=raw.rstrip("\n")
        if line.startswith("# S-ID:"):
            if sentence is not None and tokens: yield sentence,tokens,b_starts,bun_starts
            sentence=line.split()[1].split(":",1)[1];tokens=[];b_starts=[];bun_starts=[]
        elif line.startswith("*"):
            bun_starts.append(len(tokens))
        elif line.startswith("+"):
            b_starts.append(len(tokens))
        elif line=="EOS":
            if sentence is not None and tokens: yield sentence,tokens,b_starts,bun_starts
            sentence=None;tokens=[];b_starts=[];bun_starts=[]
        elif line and not line.startswith("#"):
            f=line.split()
            if len(f)>=10: tokens.append({"surface":f[0],"lemma":f[2],"pos":f[3],"subpos":f[5],"conj_type":f[7],"conj_form":f[9]})
    if sentence is not None and tokens: yield sentence,tokens,b_starts,bun_starts

def _groups(starts,n):
    starts=sorted(set(starts or [0]));return [(x,starts[i+1] if i+1<len(starts) else n) for i,x in enumerate(starts)]

def convert_sentence(sentence_id,tokens,b_starts,bun_starts,document_id):
    text="".join(t["surface"] for t in tokens);cursor=0;spans=[]
    for t in tokens:
        start=cursor;cursor+=len(t["surface"]);infl=[]
        if t["lemma"] in {"ない","ぬ","ず"}:infl.append("NEGATIVE")
        if t["lemma"] in {"た","だ"} or "タ形" in t["conj_form"]:infl.append("PAST")
        if t["lemma"]=="ます":infl.append("POLITE")
        if "条件" in t["conj_form"] or "仮定" in t["conj_form"]:infl.append("CONDITIONAL")
        if t["lemma"] in {"れる","られる"}:infl.append("PASSIVE")
        if t["lemma"] in {"せる","させる"}:infl.append("CAUSATIVE")
        spans.append({"start":start,"end":cursor,"atom_type":POS_MAP.get(t["pos"],"OTHER"),
          "function":FUNCTION.get(t["surface"],"UNKNOWN") if t["pos"]=="助詞" else None,"inflections":sorted(set(infl))})
    def materialize(starts):return [(spans[i]["start"],spans[j-1]["end"],i,j) for i,j in _groups(starts,len(tokens))]
    b_spans=[{"start":a,"end":b} for a,b,_,_ in materialize(b_starts)]
    chunks=[]
    for a,b,i,j in materialize(bun_starts):
        group=tokens[i:j];surfaces={t["surface"] for t in group};poses={t["pos"] for t in group}
        if "は" in surfaces:role="TOPIC"
        elif "動詞" in poses or "形容詞" in poses or "判定詞" in poses:role="PREDICATE"
        elif poses=={"特殊"}:role="PUNCT"
        elif poses & {"名詞","指示詞"} and poses & {"助詞"}:role="CASED_NOMINAL"
        else:role="UNKNOWN"
        chunks.append({"start":a,"end":b,"role":role})
    record=make_record(text,"kwdlc",document_id,{"a":[x["end"] for x in spans],"b":[x["end"] for x in b_spans],
      "bunsetsu":[x["end"] for x in chunks],"clause":[len(text)],"sentence":[len(text)]},spans,chunks,b_spans,
      {"gold_mapping_version":"kwdlc.v1","sentence_id":sentence_id})
    errors=validate_record(record)
    if errors:raise ValueError(";".join(errors))
    return record
