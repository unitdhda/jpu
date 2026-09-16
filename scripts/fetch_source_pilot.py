#!/usr/bin/env python3
"""Fetch small, auditable source pilots using only Python's standard library.

This is not the production downloader. It verifies schemas and extraction before
large acquisition. Responses and revisions are hashed into a provenance report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

USER_AGENT = "jpu-research/0.1 (Phase 1 dataset pilot)"
FINEWEB_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FINEWEB_URL = ("https://datasets-server.huggingface.co/first-rows?" +
               urllib.parse.urlencode({"dataset":"HuggingFaceFW/fineweb-2", "config":"jpn_Jpan", "split":"test"}))
WIKI_API = "https://ja.wikipedia.org/w/api.php"


def request_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read(), dict(response.headers)


def fetch_fineweb(limit):
    raw, headers = request_bytes(FINEWEB_URL)
    payload = json.loads(raw)
    docs = []
    for item in payload.get("rows", [])[:limit]:
        row = item["row"]
        docs.append({"source":"fineweb2-ja", "document_id":str(row["id"]), "text":row["text"],
                     "provenance":{"url":row.get("url"), "dump":row.get("dump"),
                                   "date":row.get("date"), "file_path":row.get("file_path"),
                                   "language_score":row.get("language_score"), "row_idx":item.get("row_idx")}})
    return docs, {"endpoint":FINEWEB_URL, "dataset_revision":FINEWEB_REVISION,
                  "response_sha256":hashlib.sha256(raw).hexdigest(), "rows_available":len(payload.get("rows", [])),
                  "rows_selected":len(docs)}


def wiki_query(params):
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    raw, headers = request_bytes(url)
    return json.loads(raw), raw, url


def fetch_wikipedia(limit):
    listing, listing_raw, listing_url = wiki_query({"action":"query", "format":"json", "formatversion":"2",
        "list":"allpages", "apnamespace":"0", "aplimit":str(limit), "apfilterredir":"nonredirects"})
    titles = [x["title"] for x in listing["query"]["allpages"]]
    # Intro-only extraction supports a multi-page request and avoids both the
    # whole-article exlimit=1 restriction and abusive request rates. This API
    # sample validates schemas only; production text comes from the pinned dump.
    pages, pages_raw, _ = wiki_query({"action":"query", "format":"json", "formatversion":"2",
        "prop":"extracts|revisions", "explaintext":"1", "exintro":"1", "exsectionformat":"plain",
        "rvprop":"ids|timestamp", "titles":"|".join(titles)})
    docs = []
    revisions = []
    for page in pages.get("query", {}).get("pages", []):
        text = page.get("extract", "")
        revision = (page.get("revisions") or [{}])[0]
        if not text: continue
        docs.append({"source":"jawiki", "document_id":str(page["pageid"]), "text":text,
                     "provenance":{"title":page["title"], "revision_id":revision.get("revid"),
                                   "revision_timestamp":revision.get("timestamp")}})
        revisions.append({"page_id":page["pageid"], "revision_id":revision.get("revid")})
    return docs, {"endpoint":WIKI_API, "selection":"first namespace-0 non-redirect page intros by title",
                  "listing_url":listing_url, "listing_response_sha256":hashlib.sha256(listing_raw).hexdigest(),
                  "pages_response_sha256":hashlib.sha256(pages_raw).hexdigest(), "pages_selected":len(docs),
                  "revisions":revisions}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records: out.write(json.dumps(record, ensure_ascii=False)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=Path(".data/pilot")); p.add_argument("--limit",type=int,default=20)
    args=p.parse_args()
    if not 1 <= args.limit <= 50: p.error("--limit must be between 1 and 50")
    fineweb, fineweb_meta=fetch_fineweb(args.limit); wiki, wiki_meta=fetch_wikipedia(args.limit)
    write_jsonl(args.output_dir/"fineweb2-ja.documents.jsonl",fineweb)
    write_jsonl(args.output_dir/"jawiki.documents.jsonl",wiki)
    report={"schema_version":"jpu.source-pilot.v1", "fineweb2-ja":fineweb_meta, "jawiki":wiki_meta,
            "files":{"fineweb2-ja.documents.jsonl":hashlib.sha256((args.output_dir/"fineweb2-ja.documents.jsonl").read_bytes()).hexdigest(),
                     "jawiki.documents.jsonl":hashlib.sha256((args.output_dir/"jawiki.documents.jsonl").read_bytes()).hexdigest()}}
    (args.output_dir/"provenance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"fineweb_documents":len(fineweb),"wikipedia_documents":len(wiki),"output":str(args.output_dir)},ensure_ascii=False))

if __name__=="__main__": main()
