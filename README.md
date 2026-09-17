# JPU

<video controls>
     <source
 src="https://raw.githubusercontent.com/unitdhda/jpu/main/apps/website/public/jpu-pipeline.mp4"
 type="video/mp4">
</video>

JPU is an experimental Japanese surface-grammar analyzer powered directly by
WebGPU. A 134,832-parameter model predicts morphology and shallow structure at
Unicode character gaps; deterministic code composes those predictions into a
valid nested tree.

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
diagonal affine scan, then evaluates five output heads. A deterministic decoder
closes boundary levels and emits `SENTENCE → CLAUSE → BUNSETSU → B → A`.

The promoted runtime uses 134,832 packed-FP16 weights. See
[architecture.md](architecture.md) for tensor shapes and composition details.

## Accuracy

The primary 150k checkpoint reaches the following independent converted-gold
scores:

| corpus | A F1 | B F1 | bunsetsu F1 | atom macro-F1 | particle macro-F1 | role macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| KWDLC | 94.0 | 58.5 | 95.0 | 51.3 | 60.0 | 31.1 |
| UD Japanese GSD | 98.2 | 89.4 | 97.6 | 74.3 | 56.8 | 60.3 |

JPU remains weak on B segmentation, inflection labels, and contextual chunk
roles. These numbers measure agreement with converted annotations, not semantic
correctness. See [MODEL_CARD.md](MODEL_CARD.md).

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
