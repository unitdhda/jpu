"""Silent Manim explainer for JPU's production surface-analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from manim import *
import manimpango


HERE = Path(__file__).resolve().parent
JP_FONT_FILE = HERE / "assets/fonts/NotoSansJP-Variable.ttf"
PRAGMATA_CANDIDATES = [
    Path(os.environ["JPU_PRAGMATA_FONT"]).expanduser() if os.environ.get("JPU_PRAGMATA_FONT") else None,
    Path.home() / "Library/Fonts/PragmataPro_Mono_R_liga_0903.ttf",
    Path.home() / "Library/Fonts/Pragmata Pro Mono Regular.otf",
]
PRAGMATA_FILE = next((path for path in PRAGMATA_CANDIDATES if path and path.is_file()), None)
if PRAGMATA_FILE is None:
    raise RuntimeError("Pragmata Pro not found; set JPU_PRAGMATA_FONT to your licensed font file")
for font_file in (PRAGMATA_FILE, JP_FONT_FILE):
    if not manimpango.register_font(str(font_file)):
        raise RuntimeError(f"could not register font: {font_file}")

LATIN_FONT = "PragmataPro Mono Liga"
JP_FONT = "Noto Sans JP"
REFERENCE_SIZE = 48
SENTENCE = "昨日は友達と映画を見に行かなかった。"

BG = "#000000"
WHITE = "#F5F5F5"
OFF_WHITE = "#D4D4D4"
MUTED = "#858585"
LINE = "#5E5E5E"
PANEL = "#0A0A0A"
BLUE = "#3B82F6"
CYAN = "#67E8F9"
VIOLET = "#C792EA"
GREEN = "#A6E22E"
ORANGE = "#F78C6C"
GOLD = "#FFCB6B"
CORAL = "#F07178"

ATOM_SPANS = [
    (0, 2, "NOUN_LIKE"), (2, 3, "TOPIC"),
    (3, 5, "NOUN_LIKE"), (5, 6, "COMITATIVE"),
    (6, 8, "NOUN_LIKE"), (8, 9, "OBJECT"),
    (9, 10, "VERB_STEM"), (10, 11, "UNKNOWN"),
    (11, 13, "VERB_STEM"), (13, 16, "NEGATIVE + TE_FORM"),
    (16, 17, "PAST"), (17, 18, "PUNCT_SYMBOL"),
]
A_ENDS = [end for _, end, _ in ATOM_SPANS]
# Sudachi Mode B happens to retain every Mode A boundary in this sentence.
B_ENDS = list(A_ENDS)
# GiNZA: 昨日は / 友達と / 映画を / 見に / 行かなかった。
BUNSETSU_ENDS = [3, 6, 9, 11, 18]
CLAUSE_ENDS = [18]
SENTENCE_ENDS = [18]


def latin(text, size=24, color=WHITE, **kwargs):
    return Text(
        text,
        font=LATIN_FONT,
        font_size=REFERENCE_SIZE,
        color=color,
        disable_ligatures=False,
        **kwargs,
    ).scale(size / REFERENCE_SIZE)


def japanese(text, size=36, color=WHITE, **kwargs):
    return Text(
        text,
        font=JP_FONT,
        font_size=REFERENCE_SIZE,
        color=color,
        disable_ligatures=True,
        **kwargs,
    ).scale(size / REFERENCE_SIZE)


def heading(text):
    return latin(text, 28).move_to(UP * 3.35)


def footnote(text, y=-3.45, color=MUTED):
    return latin(text, 16, color).move_to(UP * y)


def stable_values(key: str, count=8):
    state = 2166136261
    for character in key:
        state = ((state ^ ord(character)) * 16777619) & 0xFFFFFFFF
    values = []
    for index in range(count):
        state = (1664525 * (state ^ index) + 1013904223) & 0xFFFFFFFF
        values.append(0.22 + 0.75 * ((state >> 8) & 255) / 255)
    return values


class CharacterCell(VGroup):
    def __init__(self, character, width=0.57, height=0.68):
        self.character = character
        # Invisible geometry preserves stable character spacing without
        # implying that runtime input is boxed or tokenized.
        self.box = Rectangle(
            width=width,
            height=height,
            stroke_width=0,
            fill_opacity=0,
        )
        self.box.set_z_index(-1)
        self.glyph = japanese(character, 25).set_z_index(1)
        super().__init__(self.box, self.glyph)


class ActivationRow(VGroup):
    def __init__(self, key, count=8, width=1.5, height=0.24, color=BLUE):
        values = stable_values(key, count)
        gap = 0.035
        cell_width = (width - gap * (count - 1)) / count
        cells = []
        for value in values:
            cells.append(RoundedRectangle(
                width=cell_width,
                height=height,
                corner_radius=0.025,
                stroke_width=0,
                fill_color=color,
                fill_opacity=value,
            ))
        super().__init__(*cells)
        self.arrange(RIGHT, buff=gap)


class LabelBox(VGroup):
    def __init__(self, text, color, width=2.15, height=0.58, size=16):
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.12,
            stroke_color=color,
            stroke_width=1.4,
            fill_color=PANEL,
            fill_opacity=1,
        )
        label = latin(text, size, color)
        if label.width > width * 0.86:
            label.scale_to_fit_width(width * 0.86)
        super().__init__(box, label)


def character_strip(scale=1.0):
    result = VGroup(*[CharacterCell(ch) for ch in SENTENCE]).arrange(RIGHT, buff=0.055)
    if result.width > 12.75:
        result.scale_to_fit_width(12.75)
    return result.scale(scale)


def feature_matrix(chars, key, y, color=BLUE):
    rows = VGroup()
    for index, cell in enumerate(chars):
        values = stable_values(f"{key}:{index}:{cell.character}", 4)
        column = VGroup(*[
            Square(0.095, stroke_width=0, fill_color=color, fill_opacity=value)
            for value in values
        ]).arrange(UP, buff=0.018)
        column.move_to([cell.get_x(), y, 0])
        rows.add(column)
    return rows


def horizontal_arrow(start_x, end_x, y, color=BLUE):
    return Arrow(
        [start_x, y, 0], [end_x, y, 0],
        buff=0,
        color=color,
        stroke_width=4,
        max_tip_length_to_length_ratio=0.035,
    )


def boundary_markers(chars, endpoints, y, color, height=0.26):
    marks = VGroup()
    for endpoint in endpoints:
        right = chars[endpoint - 1].get_right()[0]
        marks.add(Line([right, y - height / 2, 0], [right, y + height / 2, 0], color=color, stroke_width=3))
    return marks


def span_brackets(chars, spans, y, color):
    group = VGroup()
    for start, end, label in spans:
        left = chars[start].get_left()[0]
        right = chars[end - 1].get_right()[0]
        line = Line([left, y, 0], [right, y, 0], color=color, stroke_width=2)
        line.add(Line([left, y, 0], [left, y + 0.09, 0], color=color, stroke_width=2))
        line.add(Line([right, y, 0], [right, y + 0.09, 0], color=color, stroke_width=2))
        text = latin(label, 10, color).next_to(line, DOWN, buff=0.07)
        if text.width > max(0.2, right - left - 0.03):
            text.scale_to_fit_width(max(0.2, right - left - 0.03))
        group.add(VGroup(line, text))
    return group


def enclosing_span(chars, start, end, y, height, color, label, horizontal_padding=0.05):
    left = chars[start].get_left()[0] - horizontal_padding
    right = chars[end - 1].get_right()[0] + horizontal_padding
    rect = RoundedRectangle(
        width=right - left,
        height=height,
        corner_radius=0.1,
        stroke_color=color,
        stroke_width=1.4,
        fill_opacity=0,
    ).move_to([(left + right) / 2, y, 0])
    name = latin(label, 10, color)
    if name.width > rect.width - 0.24:
        name.scale_to_fit_width(max(0.18, rect.width - 0.24))
    name.move_to(rect.get_corner(UL) + RIGHT * (0.14 + name.width / 2) + DOWN * 0.14)
    return VGroup(rect, name)


class JpuPipeline(Scene):
    slowdown = 1.42

    def setup(self):
        self.camera.background_color = BG

    def play(self, *animations, **kwargs):
        kwargs["run_time"] = kwargs.get("run_time", 1) * self.slowdown
        kwargs.setdefault("rate_func", smooth)
        return super().play(*animations, **kwargs)

    def wait(self, duration=1, **kwargs):
        return super().wait(duration * self.slowdown, **kwargs)

    def construct(self):
        # Scene 1: raw Unicode input.
        current_heading = heading("Japanese text in")
        sentence = japanese(SENTENCE, 46).move_to(UP * 0.25)
        self.play(FadeIn(current_heading, shift=DOWN * 0.08), AddTextLetterByLetter(sentence), run_time=1.25)
        runtime_note = footnote("raw Unicode input · no runtime dictionary", y=-1.35)
        model_note = footnote("134,832 learned parameters", y=-1.78, color=OFF_WHITE)
        self.play(FadeIn(runtime_note), FadeIn(model_note), run_time=0.55)
        self.wait(0.55)

        chars = character_strip().move_to(DOWN * 0.1)
        target_glyphs = VGroup(*[cell.glyph for cell in chars])
        next_heading = heading("Raw Unicode codepoints")
        offset_note = footnote("exact codepoint offsets · text is never normalized", y=-1.35)
        self.play(
            FadeOut(current_heading),
            FadeOut(runtime_note),
            FadeOut(model_note),
            FadeIn(next_heading),
            FadeIn(offset_note),
            AnimationGroup(*[
                ReplacementTransform(sentence[index], target_glyphs[index])
                for index in range(len(SENTENCE))
            ], lag_ratio=0),
            run_time=0.9,
        )
        self.remove(sentence)
        self.remove(*target_glyphs)
        self.add(chars)
        current_heading = next_heading
        self.wait(0.45)

        # Scene 2: deterministic feature extraction.
        self.play(chars.animate.shift(UP * 1.95 + RIGHT * 0.9), run_time=0.55)
        cp_label = latin("codepoint hash bucket", 14, CYAN).move_to(LEFT * 5.75 + UP * 0.65)
        bg_label = latin("right-bigram hash bucket", 14, VIOLET).move_to(LEFT * 5.75 + DOWN * 0.15)
        cat_label = latin("script + flags · 22", 14, GREEN).move_to(LEFT * 5.75 + DOWN * 0.95)
        cp = feature_matrix(chars, "cp", 0.65, CYAN)
        bg = feature_matrix(chars, "bg", -0.15, VIOLET)
        cat = feature_matrix(chars, "cat", -0.95, GREEN)
        deterministic = footnote("stable hashes · compact categories · no character vocabulary", y=-2.0)
        next_heading = heading("Deterministic CPU features")
        self.play(
            FadeOut(current_heading), FadeIn(next_heading),
            FadeOut(offset_note),
            FadeIn(cp_label), FadeIn(bg_label), FadeIn(cat_label),
            LaggedStart(*[FadeIn(column, shift=UP * 0.05) for column in cp], lag_ratio=0.025),
            LaggedStart(*[FadeIn(column, shift=UP * 0.05) for column in bg], lag_ratio=0.025),
            LaggedStart(*[FadeIn(column, shift=UP * 0.05) for column in cat], lag_ratio=0.025),
            FadeIn(deterministic),
            run_time=0.9,
        )
        current_heading = next_heading
        self.wait(0.65)

        # Scene 3: five dilated convolution blocks.
        projected = VGroup(*[
            Dot(
                radius=0.055,
                color=BLUE,
                fill_opacity=value,
                stroke_width=0,
            )
            for value in stable_values("projected:64", 64)
        ]).arrange_in_grid(rows=8, cols=8, buff=0.12)
        encoder_frame = RoundedRectangle(
            width=13.55, height=5.65, corner_radius=0.16,
            stroke_color=LINE, stroke_width=1.2,
        ).move_to(DOWN * 0.12)
        webgpu = latin("WebGPU", 15, MUTED).move_to(encoder_frame.get_corner(UL) + RIGHT * 0.64 + DOWN * 0.32)
        next_heading = heading("Embed + project each codepoint to 64 channels")
        projection_chars = character_strip(0.82).move_to(UP * 1.35)
        projected.move_to(DOWN * 0.72)
        projection_arrow = Arrow(
            projection_chars.get_bottom(), projected.get_top(),
            buff=0.24, color=BLUE, stroke_width=3,
        )
        self.play(
            FadeOut(VGroup(
                current_heading, chars, cp, bg, cat,
                cp_label, bg_label, cat_label, deterministic,
            )),
            run_time=0.4,
        )
        self.play(
            FadeIn(next_heading),
            FadeIn(encoder_frame), FadeIn(webgpu),
            FadeIn(projection_chars), FadeIn(projected),
            GrowArrow(projection_arrow),
            run_time=0.8,
        )
        current_heading = next_heading
        self.wait(2.0)

        dilation_values = (1, 2, 4, 8, 16)
        convolution_rows = VGroup(*[
            VGroup(*[
                Square(
                    0.16,
                    stroke_color="#1D355D",
                    stroke_width=1.1,
                    fill_color="#0B1730",
                    fill_opacity=0.55,
                )
                for _ in range(33)
            ]).arrange(RIGHT, buff=0.07)
            for _ in dilation_values
        ]).arrange(DOWN, buff=0.22).move_to(RIGHT * 0.55 + UP * 0.48)
        dilation_labels = VGroup(*[
            latin(f"dilation {d}", 14, BLUE).next_to(row, LEFT, buff=0.3)
            for d, row in zip(dilation_values, convolution_rows)
        ])
        tap_note = footnote(
            "kernel taps i−d · i · i+d  →  stacked receptive field: 63 codepoints",
            y=-1.5,
            color=OFF_WHITE,
        )
        conv_note = footnote(
            "depthwise context · 1×1 channel mixing · residual update",
            y=-1.92,
            color=MUTED,
        )
        next_heading = heading("5 residual separable convolutions")
        self.play(
            FadeOut(VGroup(
                current_heading, projection_chars, projection_arrow, projected,
            )),
            run_time=0.35,
        )
        self.play(
            FadeIn(next_heading), FadeIn(dilation_labels), FadeIn(convolution_rows),
            FadeIn(tap_note), FadeIn(conv_note),
            run_time=0.7,
        )
        current_heading = next_heading
        center = 16
        for index, dilation in enumerate(dilation_values):
            sampled = [center - dilation, center, center + dilation]
            self.play(
                *[
                    convolution_rows[index][position].animate
                    .set_fill(BLUE, opacity=0.95)
                    .set_stroke(BLUE, width=1.5)
                    for position in sampled
                ],
                Indicate(dilation_labels[index], color=BLUE, scale_factor=1.05),
                run_time=0.45,
            )

        conv_group = VGroup(
            dilation_labels, convolution_rows, tap_note, conv_note,
        )
        scan_forward = horizontal_arrow(-5.8, 5.8, 0.4)
        scan_backward = horizontal_arrow(5.8, -5.8, -0.35)
        scan_labels = VGroup(
            latin("forward diagonal scan", 15, BLUE).next_to(scan_forward, UP, buff=0.12),
            latin("backward diagonal scan", 15, BLUE).next_to(scan_backward, DOWN, buff=0.12),
        )
        next_heading = heading("Cheap context in both directions")
        self.play(
            FadeOut(VGroup(current_heading, conv_group)),
            run_time=0.35,
        )
        self.play(
            FadeIn(next_heading),
            GrowArrow(scan_forward), FadeIn(scan_labels[0]),
            run_time=0.7,
        )
        current_heading = next_heading
        self.play(GrowArrow(scan_backward), FadeIn(scan_labels[1]), run_time=0.8)
        context_note = footnote("local patterns + sentence-wide ordered context", y=-1.55, color=OFF_WHITE)
        self.play(FadeIn(context_note), run_time=0.4)
        self.wait(0.5)

        # Scene 4: five parallel heads.
        context_strip = character_strip(0.62).move_to(UP * 1.35)
        head_specs = [
            ("five boundary levels", CYAN),
            ("A-unit atom type", VIOLET),
            ("particle / function", GREEN),
            ("inflection flags", ORANGE),
            ("bunsetsu role", GOLD),
        ]
        heads = VGroup(*[
            LabelBox(label, color, width=2.25, height=0.62, size=14)
            for label, color in head_specs
        ])
        heads.arrange(RIGHT, buff=0.16).move_to(DOWN * 0.9)
        anchor = context_strip[9].get_bottom()
        fan = VGroup(*[
            Arrow(
                anchor, head.get_top(), buff=0.08, color=color,
                stroke_width=1.6, max_tip_length_to_length_ratio=0.08,
            )
            for head, (_, color) in zip(heads, head_specs)
        ])
        next_heading = heading("Five prediction heads in parallel")
        old_encoder = VGroup(scan_forward, scan_backward, scan_labels, encoder_frame, webgpu)
        self.play(
            FadeOut(VGroup(current_heading, old_encoder, context_note)),
            run_time=0.35,
        )
        self.play(FadeIn(next_heading), FadeIn(context_strip), run_time=0.55)
        self.play(
            LaggedStart(*[FadeIn(head, shift=UP * 0.08) for head in heads], lag_ratio=0.08),
            run_time=0.65,
        )
        self.play(
            Indicate(context_strip[9].glyph, color=CYAN, scale_factor=1.08),
            LaggedStart(*[GrowArrow(arrow) for arrow in fan], lag_ratio=0.08),
            run_time=0.8,
        )
        current_heading = next_heading
        self.wait(0.65)

        # Predicted gap levels and coarse labels.
        output_chars = character_strip(0.82).move_to(UP * 1.15)
        markers = VGroup(
            boundary_markers(output_chars, A_ENDS, 0.72, VIOLET, 0.18),
            boundary_markers(output_chars, B_ENDS, 0.49, CYAN, 0.22),
            boundary_markers(output_chars, BUNSETSU_ENDS, 0.22, GOLD, 0.27),
            boundary_markers(output_chars, CLAUSE_ENDS, -0.1, GREEN, 0.32),
            boundary_markers(output_chars, SENTENCE_ENDS, -0.47, WHITE, 0.38),
        )
        level_names = VGroup(*[
            latin(text, 13, color) for text, color in
            (("A", VIOLET), ("B", CYAN), ("bunsetsu", GOLD), ("clause", GREEN), ("sentence", WHITE))
        ]).arrange(DOWN, aligned_edge=RIGHT, buff=0.11).move_to(LEFT * 5.9 + UP * 0.13)
        next_heading = heading("Score five boundary levels at every character gap")
        self.play(
            FadeOut(VGroup(current_heading, context_strip, fan, heads)),
            run_time=0.35,
        )
        self.play(
            FadeIn(next_heading),
            FadeIn(output_chars),
            LaggedStart(*[FadeIn(row) for row in markers], lag_ratio=0.14),
            FadeIn(level_names),
            run_time=1.0,
        )
        current_heading = next_heading
        hierarchy_note = footnote(
            "five independent scores per gap · A, B, bunsetsu, clause, sentence",
            y=-1.45,
            color=OFF_WHITE,
        )
        masked_note = footnote(
            "the resulting span ends receive atom, function, inflection, and bunsetsu labels",
            y=-1.82,
        )
        self.play(FadeIn(hierarchy_note), FadeIn(masked_note), run_time=0.5)
        self.wait(0.45)

        # Scene 5: deterministic upward closure and partitioning.
        closure = VGroup(
            latin("sentence", 23, WHITE),
            latin("⊆", 23, MUTED),
            latin("clause", 23, GREEN),
            latin("⊆", 23, MUTED),
            latin("bunsetsu", 23, GOLD),
            latin("⊆", 23, MUTED),
            latin("B", 23, CYAN),
            latin("⊆", 23, MUTED),
            latin("A", 23, VIOLET),
        ).arrange(RIGHT, buff=0.16).move_to(UP * 2.15)
        next_heading = heading("Deterministic boundary closure")
        self.play(
            FadeOut(VGroup(current_heading, level_names, masked_note, hierarchy_note)),
            run_time=0.35,
        )
        self.play(
            FadeIn(next_heading),
            output_chars.animate.shift(DOWN * 1.15),
            markers.animate.shift(DOWN * 1.15),
            FadeIn(closure, shift=DOWN * 0.08),
            run_time=0.8,
        )
        current_heading = next_heading
        self.play(
            LaggedStart(*[Indicate(row, color=CYAN, scale_factor=1.0) for row in reversed(markers)], lag_ratio=0.15),
            run_time=1.05,
        )
        valid_note = footnote("valid hierarchy by construction · UNKNOWN remains available", y=-1.72, color=OFF_WHITE)
        parser_note = footnote("no learned parser · no beam search", y=-2.08)
        self.play(FadeIn(valid_note), FadeIn(parser_note), run_time=0.5)
        self.wait(0.45)

        # Build nested span boxes from the same ordered characters.
        tree_chars = character_strip(0.82).move_to(UP * 0.05)
        atoms = span_brackets(tree_chars, ATOM_SPANS, -0.46, VIOLET)
        b_ranges = [(start, end, "B") for start, end, _ in ATOM_SPANS]
        b_spans = span_brackets(tree_chars, b_ranges, -0.9, CYAN)
        # Labels below are the production 150k checkpoint's decoded output,
        # including its non-gold MODIFIER and CASED_NOMINAL decisions.
        bunsetsu_ranges = [
            (0, 3, "TOPIC"),
            (3, 6, "MODIFIER"),
            (6, 9, "CASED_NOMINAL"),
            (9, 11, "CASED_NOMINAL"),
            (11, 18, "PREDICATE"),
        ]
        bunsetsu_boxes = VGroup(*[
            enclosing_span(
                tree_chars, start, end, 0.0, 2.15, GOLD, label,
                horizontal_padding=-0.04,
            )
            for start, end, label in bunsetsu_ranges
        ])
        clause_box = enclosing_span(
            tree_chars, 0, 18, 0.03, 2.6, GREEN, "CLAUSE",
            horizontal_padding=0.18,
        )
        sentence_box = enclosing_span(
            tree_chars, 0, 18, 0.06, 3.05, WHITE, "SENTENCE",
            horizontal_padding=0.34,
        )
        next_heading = heading("Compose A → B → bunsetsu → clause → sentence")
        self.play(
            FadeOut(VGroup(current_heading, output_chars, markers, closure, valid_note, parser_note)),
            run_time=0.35,
        )
        self.play(FadeIn(next_heading), FadeIn(tree_chars), run_time=0.55)
        current_heading = next_heading
        self.play(
            LaggedStart(*[Create(item[0]) for item in atoms], lag_ratio=0.08),
            LaggedStart(*[FadeIn(item[1]) for item in atoms], lag_ratio=0.08),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[Create(item[0]) for item in b_spans], lag_ratio=0.1),
            LaggedStart(*[FadeIn(item[1]) for item in b_spans], lag_ratio=0.1),
            run_time=1.0,
        )
        self.play(LaggedStart(*[Create(box) for box in bunsetsu_boxes], lag_ratio=0.12), run_time=1.05)
        self.play(Create(clause_box), run_time=0.55)
        self.play(Create(sentence_box), run_time=0.55)
        self.wait(0.8)

        # Scene 6: clean output and deployment statement.
        color_map = {
            (0, 2): VIOLET, (2, 3): GREEN, (3, 5): VIOLET, (5, 6): GREEN,
            (6, 8): VIOLET, (8, 9): GREEN, (9, 10): CORAL, (10, 11): GREEN,
            (11, 13): CORAL, (13, 16): ORANGE, (16, 17): ORANGE, (17, 18): OFF_WHITE,
        }
        final_chars = character_strip(1.0).move_to(UP * 1.15)
        for start, end, _ in ATOM_SPANS:
            for index in range(start, end):
                final_chars[index].glyph.set_color(color_map[(start, end)])
        final_atoms = span_brackets(final_chars, ATOM_SPANS, 0.66, VIOLET)
        summary = VGroup(
            LabelBox("TOPIC  昨日は", GOLD, width=1.9, size=12),
            LabelBox("MODIFIER  友達と", GOLD, width=2.25, size=12),
            LabelBox("CASED_NOMINAL  映画を", GOLD, width=2.7, size=12),
            LabelBox("CASED_NOMINAL  見に", GOLD, width=2.45, size=12),
            LabelBox("PREDICATE  行かなかった。", GOLD, width=3.35, size=12),
        ).arrange(RIGHT, buff=0.1).move_to(DOWN * 1.05)
        next_heading = heading("Production 150k surface tree out")
        self.play(
            FadeOut(VGroup(current_heading, tree_chars, atoms, b_spans, bunsetsu_boxes, clause_box, sentence_box)),
            run_time=0.35,
        )
        self.play(
            FadeIn(next_heading),
            FadeIn(final_chars), FadeIn(final_atoms),
            LaggedStart(*[FadeIn(item, shift=RIGHT * 0.08) for item in summary], lag_ratio=0.1),
            run_time=1.0,
        )
        current_heading = next_heading
        footer = footnote("134,832 parameters · packed FP16 · WebGPU", y=-3.18, color=CYAN)
        inspiration = footnote("experimental project inspired by Shu Ding's gpu-lexer", y=-3.53, color=MUTED)
        self.play(FadeIn(footer), FadeIn(inspiration), run_time=0.6)
        self.wait(1.6)

        # Scene 7: closing card.
        closing_title = latin("JPU", 68, WHITE, weight="BOLD").set_stroke(WHITE, width=1.2).move_to(UP * 1.25)
        closing_tagline = latin(
            "tiny Japanese surface grammar on WebGPU",
            25,
            WHITE,
        ).move_to(DOWN * 0.15)
        closing_detail = latin(
            "raw Unicode · 134,832 parameters · deterministic composition",
            17,
            OFF_WHITE,
        ).move_to(DOWN * 0.85)
        closing_url = latin(
            "github.com/unitdhda/gpu-jpu",
            18,
            MUTED,
        ).move_to(DOWN * 1.65)
        self.play(
            FadeOut(VGroup(
                current_heading, final_chars, final_atoms, summary,
                footer, inspiration,
            )),
            run_time=0.45,
        )
        self.play(
            FadeIn(closing_title, shift=UP * 0.08),
            FadeIn(closing_tagline),
            FadeIn(closing_detail),
            FadeIn(closing_url),
            run_time=0.8,
        )
        self.wait(2.5)
