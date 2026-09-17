import copy
import json
from pathlib import Path
import unittest

import torch

from data.schemas.canonical import validate_record
from jp_lexer.decoder import decode_logits, nested_boundaries
from jp_lexer.features import CAT_FEATURE_NAMES, collate_records, encode_text, hash_bucket
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer, ModelConfig
from train.train import document_split, split_records


class Phase2Fixture(unittest.TestCase):
    def record(self):
        return {"text": "昨日は映画を見た。", "source": "test", "document_id": "d",
          "boundaries": {"a": [2, 3, 5, 6, 7, 8, 9], "b": [2, 3, 5, 6, 7, 8, 9],
                          "bunsetsu": [3, 6, 9], "clause": [9], "sentence": [9]},
          "spans": [{"start":0,"end":2,"atom_type":"NOUN_LIKE","function":None,"inflections":[]},
                    {"start":2,"end":3,"atom_type":"PARTICLE","function":"TOPIC","inflections":[]},
                    {"start":3,"end":5,"atom_type":"NOUN_LIKE","function":None,"inflections":[]},
                    {"start":5,"end":6,"atom_type":"PARTICLE","function":"OBJECT","inflections":[]},
                    {"start":6,"end":7,"atom_type":"VERB_STEM","function":None,"inflections":[]},
                    {"start":7,"end":8,"atom_type":"AUXILIARY","function":None,"inflections":["PAST"]},
                    {"start":8,"end":9,"atom_type":"PUNCT_SYMBOL","function":None,"inflections":[]}],
          "b_spans": [], "bunsetsu": [{"start":0,"end":3,"role":"TOPIC"},
                    {"start":3,"end":6,"role":"CASED_NOMINAL"},
                    {"start":6,"end":9,"role":"PREDICATE"}]}


class FeatureTests(Phase2Fixture):
    def test_hash_and_unicode_offsets_are_stable(self):
        first=encode_text("食べる🙂")
        second=encode_text("食べる🙂")
        self.assertTrue(torch.equal(first["codepoints"], second["codepoints"]))
        self.assertEqual(first["codepoints"].numel(), 4)
        self.assertEqual(len(first["categorical"]), 4)
        self.assertEqual(hash_bucket(ord("食"), 8192), int(first["codepoints"][0]))

    def test_padding_and_masks(self):
        short={"text":"映画。", "source":"test", "document_id":"short",
               "boundaries":{"a":[2,3],"b":[2,3],"bunsetsu":[3],"clause":[3],"sentence":[3]},
               "spans":[{"start":0,"end":2,"atom_type":"NOUN_LIKE","function":None,"inflections":[]},
                        {"start":2,"end":3,"atom_type":"PUNCT_SYMBOL","function":None,"inflections":[]}],
               "b_spans":[], "bunsetsu":[{"start":0,"end":3,"role":"PREDICATE"}]}
        batch=collate_records([self.record(), short])
        self.assertEqual(tuple(batch.inputs["codepoints"].shape), (2, 9))
        self.assertEqual(batch.inputs["mask"].tolist()[1], [True, True, True, False, False, False, False, False, False])
        self.assertFalse(batch.targets["function_mask"][1, 3:].any())
        self.assertTrue(torch.equal(batch.targets["mask"], batch.inputs["mask"]))


class SplitTests(Phase2Fixture):
    def test_document_split_is_85_10_5_and_never_mixes_documents(self):
        rows=[]
        for i in range(1000):
            row=self.record(); row["document_id"]="doc-%d" % i; rows.append(row)
        splits=split_records(rows)
        self.assertEqual(sum(map(len, splits.values())), 1000)
        self.assertEqual(len({document_split(x) for x in rows}), 3)
        self.assertGreaterEqual(len(splits["train"]), 800)
        self.assertLessEqual(len(splits["train"]), 900)
        self.assertGreaterEqual(len(splits["dev"]), 50)
        self.assertLessEqual(len(splits["dev"]), 150)
        self.assertGreaterEqual(len(splits["test"]), 20)
        self.assertLessEqual(len(splits["test"]), 100)


class ModelTests(Phase2Fixture):
    def test_shapes_masked_loss_and_serialization(self):
        model=JpLexer(ModelConfig(hidden=16, dilations=(1, 2)))
        batch=collate_records([self.record()])
        outputs=model(batch.inputs)
        self.assertEqual(outputs["boundaries"].shape, (1, 9, 5))
        losses=multitask_loss(outputs, batch.targets)
        self.assertTrue(torch.isfinite(losses["total"]))
        state=copy.deepcopy(model.state_dict())
        restored=JpLexer(model.config); restored.load_state_dict(state)
        for key in outputs:
            self.assertTrue(torch.allclose(outputs[key], restored(batch.inputs)[key]))

    def test_export_scan_matches_sequential_scan(self):
        model=JpLexer(ModelConfig(hidden=16, dilations=(1, 2))).eval()
        batch=collate_records([self.record()])
        with torch.inference_mode():
            regular=model(batch.inputs)
            exported=model.forward_export(batch.inputs)
        for key in regular:
            self.assertTrue(torch.allclose(regular[key], exported[key], atol=1e-5, rtol=1e-5), key)

    def test_parameter_variants_and_subsystem_accounting(self):
        reports=[JpLexer.from_size(x).parameter_report() for x in ("50k", "100k", "150k", "250k")]
        totals=[x["total_parameters"] for x in reports]
        self.assertEqual(totals, sorted(totals))
        self.assertGreater(totals[0], 30000)
        self.assertLess(totals[-1], 250000)
        for report in reports:
            self.assertEqual(sum(report["by_subsystem"].values()), report["total_parameters"])


class DecoderTests(Phase2Fixture):
    def test_nesting_and_tree(self):
        boundaries=nested_boundaries({"a":[2],"b":[2,4],"bunsetsu":[4],"clause":[4],"sentence":[]}, 4)
        self.assertEqual(boundaries["sentence"], [4])
        self.assertTrue(set(boundaries["clause"]).issubset(boundaries["bunsetsu"]))
        decoded=decode_logits("映画を見た。", {"boundaries": torch.full((6,5), -20.0)})
        self.assertEqual(validate_record({"text":decoded["text"],"boundaries":decoded["boundaries"],
            "spans":decoded["spans"],"b_spans":decoded["b_spans"],"bunsetsu":decoded["bunsetsu"]}), [])
        self.assertEqual(decoded["tree"]["type"], "SENTENCE")


if __name__ == "__main__": unittest.main()
