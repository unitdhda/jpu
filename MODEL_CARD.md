# JPU model card

## Model

JPU is an experimental Japanese surface-grammar analyzer. It predicts
character-gap morphology and shallow chunk structure, then uses a deterministic
composer to construct a valid nested tree. It is not a semantic parser,
translator, dependency parser, or grammar corrector.

The primary deployment variants are approximately 50k and 150k learned
parameters. They operate directly on Unicode codepoints and do not require
Sudachi, GiNZA, a runtime dictionary, or an LLM.

## Training data

The default pseudo-gold recipe is 70% FineWeb2 Japanese and 30% Japanese
Wikipedia, split by document. Sudachi Mode A/B and GiNZA provide offline
teacher labels. Deterministic synthetic morphology is a minority augmentation.
The current provenance and acquisition status are recorded in
`data/acquisition/sources.v1.json` and `manifests/`.

## Evaluation

The current production 150k checkpoint reports teacher-aligned independent-gold
results as follows:

| corpus | A F1 | B F1 | bunsetsu F1 | atom macro-F1 | inflection macro-F1 | particle macro-F1 | role macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KWDLC test | 94.0 | 58.5 | 95.0 | 51.3 | 14.8 | 60.0 | 31.1 |
| UD Japanese GSD test | 98.2 | 89.4 | 97.6 | 74.3 | 28.1 | 56.8 | 60.3 |

These are aggregate comparisons against converted annotations, not claims of
semantic correctness. The 150k model remains weak on B segmentation,
inflections, and contextual chunk roles. Reproduce the comparison with
`eval/benchmark_teachers.py` and the sealed evaluation inputs.

## Limitations and risks

- Teacher disagreement and ontology ambiguity limit pseudo-gold quality.
- Results vary by domain, script mixture, sentence length, and rare morphology.
- Particle functions are coarse surface categories, not semantic roles.
- The current clause supervision is limited and should not be treated as full
  syntactic parsing.
- Model files are experimental checkpoints, not safety-critical analyzers.

## Intended use

Research on compact Japanese morphology and shallow syntax, browser demos, and
reproducible model-size experiments. Do not use it as the sole source for
high-stakes linguistic, legal, educational, or accessibility decisions.
