import json
import tempfile
import unittest
from pathlib import Path

from data.acquisition.extract import dedup_key, extract_sentences, split_for_document
from data.acquisition.dedup import hamming_distance, lsh_bands, simhash64


class ExtractionTests(unittest.TestCase):
    def test_preserves_raw_slices_and_offsets(self):
        text = "  昨日は晴れた。\n「映画を見る？」  次です！"
        rows = list(extract_sentences({"source":"x", "document_id":"d", "text":text}))
        self.assertEqual([x["text"] for x in rows], ["昨日は晴れた。", "「映画を見る？」", "次です！"])
        for row in rows:
            self.assertEqual(text[row["document_start"]:row["document_end"]], row["text"])

    def test_length_filter_uses_codepoints(self):
        rows = list(extract_sentences({"source":"x", "document_id":"d", "text":"短い。これは十分な長さです。"}))
        self.assertEqual([x["text"] for x in rows], ["これは十分な長さです。"])

    def test_document_split_is_stable(self):
        first = split_for_document("source", "doc")
        self.assertEqual(first, split_for_document("source", "doc"))
        rows = list(extract_sentences({"source":"source", "document_id":"doc", "text":"最初の文です。二番目の文です。"}))
        self.assertEqual({x["split"] for x in rows}, {first})

    def test_dedup_comparison_does_not_rewrite_output(self):
        self.assertEqual(dedup_key("同じ 文です。"), dedup_key("同じ   文です。"))

    def test_invalid_document_fails(self):
        with self.assertRaises(ValueError):
            list(extract_sentences({"source":"x", "document_id":"", "text":"文章です。"}))


class NearDedupTests(unittest.TestCase):
    def test_whitespace_variant_is_near_duplicate(self):
        left = "これは 比較対象となる 日本語文書です。" * 10
        right = "これは   比較対象となる\n日本語文書です。" * 10
        self.assertEqual(hamming_distance(simhash64(left), simhash64(right)), 0)

    def test_unrelated_documents_are_distant(self):
        left = "日本語の形態論を評価する文章です。" * 10
        right = "宇宙望遠鏡が遠方銀河を撮影しました。" * 10
        self.assertGreater(hamming_distance(simhash64(left), simhash64(right)), 3)

    def test_distance_three_guarantees_shared_band(self):
        left = 0x1234567890ABCDEF
        right = left ^ 1 ^ 2 ^ 4
        self.assertTrue(set(lsh_bands(left)) & set(lsh_bands(right)))


class ManifestTests(unittest.TestCase):
    def test_default_recipe_and_optional_edu_ablation_are_explicit(self):
        path = Path(__file__).parents[1] / "data/acquisition/sources.v1.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        sources = manifest["default_recipe"]
        self.assertEqual({x["id"]: x["target_fraction"] for x in sources},
                         {"fineweb2-ja": 0.70, "jawiki": 0.30})
        self.assertAlmostEqual(sum(x["target_fraction"] for x in sources), 1.0)
        wiki = next(x for x in sources if x["id"] == "jawiki")
        self.assertEqual(wiki["revision"], "20260901")
        self.assertEqual(len(wiki["dump_sha1"]), 40)
        ablation = manifest["optional_ablations"][0]
        self.assertEqual(ablation["dataset"], "hotchpotch/fineweb-2-edu-japanese")
        self.assertNotEqual(ablation["status"], "approved")
        self.assertNotIn(ablation["id"], {x["id"] for x in sources})

if __name__ == "__main__": unittest.main()
