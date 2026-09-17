import unittest
from data.annotation.gold_ud import convert_sentence,parse_conllu
from data.annotation.gold_kwdlc import convert_sentence as convert_kwdlc,parse_knp
from data.schemas.canonical import validate_record

class UDGoldTests(unittest.TestCase):
 def test_ud_hierarchy(self):
  sample='''# sent_id = x\n# text = 映画を見た。\n1\t映画\t映画\tNOUN\t名詞-普通名詞-一般\t_\t3\tobj\t_\tBunsetuBILabel=B|LUWBILabel=B|SpaceAfter=No\n2\tを\tを\tADP\t助詞-格助詞\t_\t1\tcase\t_\tBunsetuBILabel=I|LUWBILabel=B|SpaceAfter=No\n3\t見\t見る\tVERB\t動詞-一般\t_\t0\troot\t_\tBunsetuBILabel=B|LUWBILabel=B|SpaceAfter=No\n4\tた\tた\tAUX\t助動詞-助動詞-タ\tTense=Past\t3\taux\t_\tBunsetuBILabel=I|LUWBILabel=B|SpaceAfter=No\n5\t。\t。\tPUNCT\t補助記号-句点\t_\t3\tpunct\t_\tBunsetuBILabel=I|LUWBILabel=B\n\n'''
  comments,tokens=next(parse_conllu(sample.splitlines(True)));record=convert_sentence(comments,tokens)
  self.assertEqual(validate_record(record),[]);self.assertEqual(record['spans'][1]['function'],'OBJECT');self.assertEqual(record['spans'][3]['inflections'],['PAST'])

class KWDLCGoldTests(unittest.TestCase):
 def test_knp_hierarchy(self):
  sample='''# S-ID:d-1\n* 1D\n+ 1D\n映画 えいが 映画 名詞 6 普通名詞 1 * 0 * 0 NIL\nを を を 助詞 9 格助詞 1 * 0 * 0 NIL\n* -1D\n+ -1D\n見た みた 見る 動詞 2 * 0 母音動詞 1 タ形 10 NIL\n。 。 。 特殊 1 句点 1 * 0 * 0 NIL\nEOS\n'''
  sid,tokens,bs,buns=next(parse_knp(sample.splitlines(True)));record=convert_kwdlc(sid,tokens,bs,buns,'d')
  self.assertEqual(validate_record(record),[]);self.assertEqual(record['bunsetsu'][0]['role'],'CASED_NOMINAL')

if __name__=='__main__':unittest.main()
