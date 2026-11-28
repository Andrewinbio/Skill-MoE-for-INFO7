import json
from pathlib import Path
import tempfile
import unittest

from collect import annotation_requests, freeze, load_cache, route


class CollectionTests(unittest.TestCase):
    def test_interrupted_tail_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.jsonl'
            first = json.dumps({'id': 'first'}) + '\n'
            path.write_text(first + '{"id": "unfinished')
            with self.assertRaises(ValueError):
                load_cache(path)
            self.assertEqual(set(load_cache(path, repair=True)), {'first'})
            self.assertEqual(path.read_text(), first)
            path.write_text(first + first)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_cache(path, repair=True)

    def test_frozen_inputs_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            freeze(path, {'revision': 'one', 'ids': ['a']})
            freeze(path, {'revision': 'one', 'ids': ['a']})
            with self.assertRaises(ValueError):
                freeze(path, {'revision': 'two', 'ids': ['a']})
            self.assertEqual(json.loads(path.read_text())['revision'], 'one')

    def test_annotation_inputs_do_not_include_gold(self):
        rows = [{'id':'train:0','question':'A medical question','gold_answer':'SENSITIVE_GOLD'}]
        requests = annotation_requests(rows)
        self.assertEqual(len(requests), 5)
        self.assertEqual(len({r['id'] for r in requests}), 5)
        self.assertNotIn('SENSITIVE_GOLD', json.dumps(requests))

    def test_router_probability_and_uniform_fallback(self):
        rows = [{'id':'one'}, {'id':'two'}]
        profiles = {'profiles': {'qwen': {'heart':2}, 'biodecision':{'heart':1}, 'coder':{'heart':-1}}}
        result = route(rows, {'one':['heart'], 'two':[]}, profiles)
        self.assertAlmostEqual(result['one']['probabilities']['qwen'], 32/41)
        self.assertAlmostEqual(result['one']['probabilities']['biodecision'], 9/41)
        self.assertEqual(result['one']['probabilities']['coder'], 0)
        self.assertEqual(len(result['one']['solvers']), 3)
        self.assertEqual(result['two']['fallback'], 'empty_keywords_uniform')
        self.assertTrue(all(abs(p-1/3)<1e-12 for p in result['two']['probabilities'].values()))
        self.assertEqual(result, route(rows, {'one':['heart'], 'two':[]}, profiles))


if __name__ == '__main__':
    unittest.main()
