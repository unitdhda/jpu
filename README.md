# JPU

JPU is an experimental project inspired by Shu Ding's
[`gpu-lexer`](https://github.com/shuding/gpu-lexer). It is an independent
implementation with an independent Japanese corpus and teacher pipeline.

JPU is a tiny Japanese surface-grammar analyzer designed for browser and
small-device deployment. It combines a compact PyTorch model with a
deterministic hierarchical composer:

```text
Unicode codepoints → boundary/label predictions → valid nested surface tree
```

It predicts coarse morphology, particle functions, inflection flags, and
bunsetsu-level roles. It does not predict dependencies, semantics, omitted
subjects, readings, translation, or free-form text.

## Repository layout

```text
apps/                 application/package entry points
packages/             workspace metadata for core, training, and evaluation
web/public/models/    exported WebGPU weight assets
data/                 acquisition, alignment, canonical labels, augmentation
manifests/            versioned corpus, split, holdout, and environment manifests
schemas/              readable release schemas
jp_lexer/             Python model, feature, loss, and decoder implementation
scripts/              corpus, annotation, training, and export commands
eval/                 metrics, teacher benchmarks, and error analysis
tests/                deterministic unit tests
```

The package boundaries mirror the gpu-lexer-style separation between app,
core/runtime, training data, benchmark, and release metadata while preserving
JPU's existing Python implementation and Japanese teacher pipeline.

## Development

Create the pinned Python environment for the phase being worked on, then run:

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

## Corpus build

The reproducible Phase 1 flow is deliberately offline-teacher based:

```text
source documents
  → normalized document JSONL
  → document-level deduplication and split assignment
  → Unicode-preserving sentence extraction
  → Sudachi A/B + GiNZA annotation
  → strict alignment and canonical validation
  → readable JSONL plus failure/statistics reports
```

Natural corpus construction is fail-closed. It does not silently substitute a
third-party dataset for the pinned 70/30 FineWeb2 Japanese/Wikipedia recipe.
See `data/corpus.json`, `manifests/source-corpora.json`, and
`data/acquisition/sources.v1.json` before acquiring source text.

## License and data terms

JPU code is MIT-licensed. Training and evaluation corpora are not automatically
licensed by this repository; their source-specific terms and attribution rules
are documented in `THIRD_PARTY_NOTICES.md`.
