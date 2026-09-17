# JPU

JPU is an experimental Japanese surface-segmentation and shallow grammatical-
labeling model powered directly by WebGPU. A 134,832-parameter model predicts
character-gap boundaries, coarse surface labels, limited inflection flags, and
shallow chunk roles; deterministic code composes those predictions into a valid
nested tree. It is not a full morphological or semantic analyzer.

```js
import { CustomWebGpuLexer } from "jpu";

const analyzer = await CustomWebGpuLexer.create({ modelSize: "150k" });
const result = await analyzer.analyze("昨日は映画を見た。");
```

The browser runtime requires WebGPU and a secure context. It has no Sudachi,
GiNZA, ONNX Runtime, dictionary, Transformer, or LLM dependency.

## How it works

The CPU computes stable codepoint and right-bigram hash buckets plus 22 Unicode
categories. WebGPU looks up compact embeddings, projects each codepoint to 64
channels, applies five residual separable convolutions and a bidirectional
diagonal affine scan, then evaluates five output heads for boundaries, atom
types, particle functions, inflection flags, and bunsetsu roles. A deterministic
decoder closes boundary levels and emits `SENTENCE → CLAUSE → BUNSETSU → B → A`.

The promoted runtime uses 134,832 packed-FP16 weights. See
[architecture.md](architecture.md) for tensor shapes and composition details.

## Accuracy

The primary 150k checkpoint reaches the following independent converted-gold
scores:

| corpus | A F1 | B F1 | bunsetsu F1 | atom macro-F1 | inflection macro-F1 | particle macro-F1 | role macro-F1 | tree exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KWDLC | 94.0 | 58.5 | 95.0 | 51.3 | 14.8 | 60.0 | 31.1 | 0.0 |
| UD Japanese GSD | 98.2 | 89.4 | 97.6 | 74.3 | 28.1 | 56.8 | 60.3 | 1.1 |

JPU remains weak on B segmentation, inflection labels, and contextual chunk
roles. Boundary scores are exact character-gap endpoint F1; label scores are
character-position weighted, and complete-tree match requires exact equality of
boundaries and labeled spans. These numbers measure agreement with converted
annotations, not semantic correctness. See [MODEL_CARD.md](MODEL_CARD.md).

### CPU and GPU reference

For the 150k model on the sample sentence, warm Safari WebGPU measured 4.00 ms
for batch 1, 7.00 ms for batch 16, and 20.00 ms for batch 64. The corresponding
Python/PyTorch CPU timings were 4.73 ms, 61.78 ms, and 255.44 ms, or 4.73,
3.86, and 3.99 ms per sentence. At batch 64 this is approximately 12.8× higher
throughput on WebGPU. These are same-sample reference timings, not independent-
gold corpus results; CPU uses FP32 and GPU uses packed FP16 weights, and browser
timers are approximate.

## Development

```sh
corepack pnpm install
corepack pnpm test
corepack pnpm build:website
```

Python model and corpus tests require the dependencies in
`packages/training/requirements-torch.txt`.

## Repository

- `packages/core`: publishable JavaScript/WebGPU runtime
- `packages/training`: corpus compiler, PyTorch model, training, and export
- `packages/benchmark`: independent metrics and teacher comparisons
- `apps/website`: interactive demo and promoted model assets
- `video`: Manim explainer source and storyboard

Corpora, checkpoints, generated training arrays, and local runs are intentionally
ignored. Provenance manifests are tracked instead.

JPU is an independent project inspired by Shu Ding's
[`gpu-lexer`](https://github.com/vercel-labs/gpu-lexer); it does not reuse that
project's code, weights, or corpus.

## License

MIT. Source corpora, teacher software, evaluation sets, and fonts retain their
own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
