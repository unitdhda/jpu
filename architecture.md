# JPU architecture

JPU analyzes raw Unicode codepoints without normalization, a word vocabulary,
or a runtime dictionary.

```text
codepoint hash       → learned embedding [T, 8]
right-bigram hash    → learned embedding [T, 4]
categorical features → deterministic     [T, 22]
concatenate + project                    [T, 64]
5 residual separable convolutions        [T, 64]
bidirectional diagonal affine scan       [T, 64]
5 independent output heads
```

The kernel-3 convolutions use dilations `1, 2, 4, 8, 16`, giving a
63-codepoint receptive field before the sentence-wide scan. The primary model
has 134,832 learned parameters.

The heads emit five independent boundary logits plus atom type, particle
function, inflection flags, and bunsetsu role. The deterministic decoder
thresholds boundary logits, closes higher accepted levels into every lower
level, partitions spans, and emits:

```text
SENTENCE → CLAUSE → BUNSETSU → B → A
```

No learned parser, beam search, Sudachi, GiNZA, or runtime dictionary is used
in inference. The browser runtime executes the learned operations directly in
WGSL using packed FP16 weights and FP32 computation.
