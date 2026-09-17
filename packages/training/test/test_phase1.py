import unittest

from data.alignment.align import align_surfaces
from data.annotation.compiler import compile_example
from data.annotation.mappings import map_bunsetsu_role, map_inflections, map_particle, map_sudachi_pos
from data.schemas.canonical import validate_record
from data.synthetic.generate import generate_lemma, lemma_split


class MappingTests(unittest.TestCase):
    def test_pos(self):
        self.assertEqual(map_sudachi_pos(["名詞", "固有名詞"]), "PROPER_LIKE")
        self.assertEqual(map_sudachi_pos(["名詞", "数詞"]), "NUMBER")
        self.assertEqual(map_sudachi_pos(["動詞", "一般"]), "VERB_STEM")
        self.assertEqual(map_sudachi_pos(["助詞", "格助詞"]), "PARTICLE")
        self.assertEqual(map_sudachi_pos(["助動詞", "*", "*", "*", "助動詞-ダ", "終止形-一般"]), "COPULA")

    def test_particle_ambiguity_is_unknown(self):
        self.assertEqual(map_particle("に", ["助詞", "格助詞"]), "UNKNOWN")
        self.assertEqual(map_particle("に", ["助詞", "格助詞"], {"case": "TIME"}), "TIME")
        self.assertEqual(map_particle("と", ["助詞", "格助詞"], {"case": "QUOTE"}), "QUOTATIVE")
        self.assertEqual(map_particle("を", ["助詞", "格助詞"]), "OBJECT")

    def test_morphology_and_roles(self):
        self.assertEqual(map_inflections("なかっ", "ない"), ["NEGATIVE"])
        self.assertEqual(map_inflections("た", "た"), ["PAST"])
        self.assertEqual(map_inflections("なけれ", "ない", "仮定形-一般"), ["CONDITIONAL", "NEGATIVE"])
        self.assertEqual(map_bunsetsu_role("nsubj", "TOPIC"), "TOPIC")
        self.assertEqual(map_bunsetsu_role("root", is_predicate=True), "PREDICATE")


class AlignmentTests(unittest.TestCase):
    def test_exact_unicode_offsets(self):
        aligned, failure = align_surfaces("映画を見た。", [{"surface": "映画"}, {"surface": "を"}, {"surface": "見"}, {"surface": "た"}, {"surface": "。"}], "test")
        self.assertIsNone(failure)
        self.assertEqual([(x["start"], x["end"]) for x in aligned], [(0,2),(2,3),(3,4),(4,5),(5,6)])

    def test_never_searches_ahead(self):
        aligned, failure = align_surfaces("映画を見た。", [{"surface": "映画"}, {"surface": "見た"}], "test")
        self.assertIsNone(aligned); self.assertEqual(failure.reason, "surface_mismatch")


class CompilerTests(unittest.TestCase):
    def fixture(self):
        return {"text": "昨日は映画を見た。", "source": "unit", "document_id": "doc-1",
          "sudachi_a": [
            {"surface":"昨日","pos":["名詞","普通名詞"]}, {"surface":"は","pos":["助詞","係助詞"]},
            {"surface":"映画","pos":["名詞","普通名詞"]}, {"surface":"を","pos":["助詞","格助詞"]},
            {"surface":"見","pos":["動詞","一般"]}, {"surface":"た","pos":["助動詞","*"] ,"normalized_form":"た"},
            {"surface":"。","pos":["補助記号","句点"]}],
          "sudachi_b": [{"surface":"昨日"},{"surface":"は"},{"surface":"映画"},{"surface":"を"},{"surface":"見た"},{"surface":"。"}],
          "ginza_bunsetsu": [{"surface":"昨日は","dependency":"nsubj","particle_function":"TOPIC"},
             {"surface":"映画を","dependency":"obj"},{"surface":"見た","dependency":"root","is_predicate":True},
             {"surface":"。","dependency":"punct"}]}

    def test_compile_and_nesting(self):
        record, failure = compile_example(self.fixture())
        self.assertIsNone(failure); self.assertEqual(validate_record(record), [])
        self.assertEqual(record["bunsetsu"][0]["role"], "TOPIC")
        self.assertEqual(record["spans"][-2]["inflections"], ["PAST"])

    def test_disagreement_is_recorded(self):
        raw = self.fixture(); raw["sudachi_b"] = [{"surface": raw["text"]}]
        record, failure = compile_example(raw)
        self.assertIsNone(record); self.assertEqual(failure["reason"], "teacher_boundary_disagreement")


class SyntheticTests(unittest.TestCase):
    def test_paradigms(self):
        forms = {x["paradigm"]: x for x in generate_lemma("書く", "godan_k")}
        self.assertEqual(forms["causative_passive"]["text"], "書かせられる")
        self.assertEqual(forms["negative_past"]["spans"][-1]["inflections"], ["NEGATIVE", "PAST"])

    def test_family_split_has_no_leakage(self):
        self.assertEqual({x["split"] for x in generate_lemma("食べる", "ichidan")}, {lemma_split("食べる")})

if __name__ == "__main__": unittest.main()
