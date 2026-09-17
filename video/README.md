# JPU pipeline video

Silent 16:9 Manim explainer for JPU's production inference path: raw Unicode
codepoints, deterministic hash/script features, five residual dilated
convolution blocks, bidirectional diagonal scan, parallel prediction heads, and
deterministic hierarchical composition.

The storyboard and accuracy guardrails are in [`scenes.md`](scenes.md).

## Fonts

Japanese text uses Noto Sans JP from
[google/fonts](https://github.com/google/fonts/tree/main/ofl/notosansjp), under
the SIL Open Font License 1.1 included at
`assets/fonts/OFL-NotoSansJP.txt`. The 9 MiB font binary is not committed; fetch
the pinned file and verify its checksum with `python3 fetch_font.py`.

Latin labels use **PragmataPro Mono Liga**. Pragmata Pro is commercially
licensed and is deliberately not included. The renderer searches these local
paths:

```text
$JPU_PRAGMATA_FONT
~/Library/Fonts/PragmataPro_Mono_R_liga_0903.ttf
~/Library/Fonts/Pragmata Pro Mono Regular.otf
```

Set `JPU_PRAGMATA_FONT` if your licensed copy is elsewhere.

## Render

From this directory:

```sh
python3 fetch_font.py
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/manim -ql jpu_pipeline.py JpuPipeline
.venv/bin/manim -qh --fps 30 jpu_pipeline.py JpuPipeline
```

The final command creates the 1920×1080, 30 fps site asset. The video has no
audio track.

JPU is an independent experimental project inspired by Shu Ding's
[gpu-lexer](https://github.com/vercel-labs/gpu-lexer). The video concept follows
that project's concise pipeline-explainer format but depicts JPU's different
Unicode, convolutional, multihead, and deterministic-composer architecture.
