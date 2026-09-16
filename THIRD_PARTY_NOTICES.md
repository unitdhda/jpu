# Third-party notices

JPU source code is MIT-licensed. Training inputs, teacher software, and
independent evaluation corpora retain their own licenses and terms.

## Inspiration

JPU is an independent experimental project inspired by Shu Ding's
[`gpu-lexer`](https://github.com/shuding/gpu-lexer), which is MIT-licensed.
JPU does not include gpu-lexer source code, model weights, training corpus, or
other project assets; the relationship is design inspiration and attribution
only.

## Natural-text training sources

The default corpus recipe is:

- 70% `HuggingFaceFW/fineweb-2`, configuration `jpn_Jpan` — ODC-By-1.0
  according to the dataset metadata.
- 30% Japanese Wikipedia dated dumps — CC-BY-SA-4.0; source revisions may
  also carry GFDL terms.

Source revisions, dump identifiers, hashes, and attribution requirements are
recorded in `data/acquisition/sources.v1.json`. Source text is not redistributed
by this repository. Users building a corpus must acquire it directly and
comply with the applicable terms.

## Teacher and annotation software

SudachiPy, Sudachi dictionaries, spaCy, GiNZA, and their Japanese models are
offline annotation dependencies only. Their exact versions are pinned in
`requirements/phase1-teachers.lock.txt`; consult each installed distribution's
license and notice files when redistributing an environment or derived data.

## Evaluation corpora

KWDLC, UD Japanese GSD, and NPCMJ are independent corpora with independent
terms. Their status and source URLs are recorded in
`data/acquisition/sources.v1.json`. Evaluation records and converted text must
not be redistributed unless the corpus terms permit it.

## Synthetic morphology

The checked-in seed generator emits deterministic generated forms and labels;
it does not redistribute copied source documents. Any future authoritative
lexicon or conjugation resource used to expand it must be pinned in the source
manifest with its license and attribution before a corpus release. The
synthetic generator records its mapping/version in each release manifest.
