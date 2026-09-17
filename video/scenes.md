# How JPU Reads a Japanese Sentence

## Overview

- **Topic**: How JPU converts raw Japanese Unicode into a nested surface tree.
- **Hook**: A 134,832-parameter model runs without a dictionary or tokenizer.
- **Audience**: Developers and Japanese NLP researchers; no ML background required.
- **Length**: approximately 50–55 seconds, silent, 16:9.
- **Core idea**: learned local/long-range evidence predicts labels and nested gap boundaries; deterministic code enforces a valid tree.

## Scene 1 — Raw Unicode in

Show `昨日は友達と映画を見に行かなかった。` centered on black. Captions:
`Japanese text in`, then `raw Unicode input · no runtime dictionary`. Move the
glyphs into evenly spaced Unicode-codepoint positions without adding token-cell
borders, morphing the glyphs, changing the text, or changing offsets.

## Scene 2 — Deterministic features

Reveal all three colored feature rows together:

1. stable codepoint hash buckets;
2. stable right-bigram hash buckets;
3. script/punctuation/digit categorical flags.

Emphasize `exact Unicode codepoint offsets` and `deterministic on CPU`. Do not
imply a character vocabulary or orthographic normalization.

## Scene 3 — Context encoder

On WebGPU, look up the learned 8d codepoint and 4d right-bigram embeddings,
concatenate the 22 categorical flags, and project the resulting 34d vector to
64 channels per codepoint. Fade the CPU feature view out, then fade in exactly
64 separated points; do not morph feature marks into channel points. Hold this
view long enough to read.

Show the five residual depthwise/separable convolutions as five rows of 33
square nodes. For dilations `1 · 2 · 4 · 8 · 16`, highlight the three sampled
nodes `i−d`, `i`, and `i+d` in each row. State the 63-codepoint stacked receptive
field. Then run electric-blue arrows left-to-right and right-to-left for the
cheap diagonal affine scan. Caption: `local patterns + sentence context`.

## Scene 4 — Parallel heads

Use a vertical chart: put the contextual Japanese character strip above and the
five category boxes in one row below it:

1. five gap-boundary levels;
2. A-unit atom type;
3. particle/function;
4. inflection flags;
5. bunsetsu role.

After both layers appear, highlight `見` (not the final punctuation mark) and
draw arrows from that character to the five category boxes.

Show five independent boundary scores after every character: A, B, bunsetsu,
clause, and sentence. Do not imply that the logits themselves are ordinal; the
deterministic closure in the next scene makes higher accepted boundaries imply
all lower levels. The resulting span ends receive the coarse label-head output.
Do not show dependency heads or semantic roles.

## Scene 5 — Deterministic closure and composition

Take intentionally incomplete/noisy predicted boundary markers and close them
upward:

`sentence ⊆ clause ⊆ bunsetsu ⊆ B ⊆ A`, with each level colored to match its
boundary-marker row.

Then partition the sentence into A spans, B spans, bunsetsu, clause, and
sentence. Explicit caption: `valid hierarchy by construction` and
`no learned parser · no beam search`.

## Scene 6 — Surface tree out

Reveal the production 150k checkpoint's decoded analysis, including its errors:

- `昨日 / は` → topic bunsetsu;
- `友達 / と` → modifier bunsetsu;
- `映画 / を` → cased-nominal bunsetsu;
- `見 / に` → cased-nominal bunsetsu, with particle function `UNKNOWN`;
- `行か / なかっ / た / 。` → predicate bunsetsu.

Its `なかっ` span predicts `NEGATIVE` and `TE_FORM`. Sudachi Mode B retains the
Mode A boundaries for this sentence; the production checkpoint reproduces
those boundaries and the GiNZA bunsetsu splits exactly. Do not silently replace
checkpoint errors with an idealized analysis.

Finish on the highlighted sentence and compact tree. Footer:
`134,832 parameters · packed FP16 · WebGPU`.

## Scene 7 — Closing card

Fade to a centered closing card:

- bold white `JPU`;
- `tiny Japanese surface grammar on WebGPU`, with a larger vertical gap below
the title;
- `raw Unicode · 134,832 parameters · deterministic composition`;
- `github.com/unitdhda/gpu-jpu`.

## Visual system

- black background;
- off-white structure/text;
- muted gray secondary labels;
- electric blue learned activations;
- cyan boundaries;
- violet atom types;
- green particle functions;
- orange inflections;
- gold bunsetsu roles.

Use PragmataPro Mono Liga for Latin labels and Noto Sans JP for Japanese text.
Semantic colors appear only at model outputs; internal activations remain blue.

## Accuracy guardrails

- Input is Unicode codepoints, not words, bytes, or dictionary tokens.
- Hash collisions are possible and expected.
- The encoder is five residual depthwise/separable convolution blocks followed
  by a diagonal bidirectional affine scan; it is not a Transformer.
- Five separate heads predict boundaries and coarse labels.
- Particle functions are coarse surface functions, not semantic roles.
- Higher boundaries imply lower boundaries; the decoder enforces this.
- The composer is deterministic and does not perform learned shift/reduce or
  dependency parsing.
- Sudachi and GiNZA are offline teachers only and never appear in runtime.
- Displayed labels illustrate model intent and are not a claim of perfect output.
