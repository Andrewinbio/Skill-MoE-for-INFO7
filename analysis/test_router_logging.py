"""Check opt-in logging without importing GPU/embedding dependencies."""
import ast
from pathlib import Path
import random
import unittest

from prepare_medmcqa import features, validate_router


class RouterLoggingTests(unittest.TestCase):
    def test_true_distribution_and_identical_sampling(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / 'recruit_agents.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'recruit_agents')
        rng = random.Random(0)
        namespace = {'random': rng}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'routing', 'exec'), namespace)
        recruit = namespace['recruit_agents']
        scores = [('A', 2), ('B', 1), ('C', -1)]
        strengths = {'A': .6, 'B': .3, 'C': .1}
        original_state = rng.getstate()
        plain = recruit(scores, strengths, k=30)
        next_state = rng.getstate()
        rng.setstate(original_state)
        logged, distribution = recruit(scores, strengths, k=30, return_probabilities=True)
        self.assertEqual(plain, logged)
        self.assertEqual(next_state, rng.getstate())
        self.assertAlmostEqual(distribution['A'], .8)
        self.assertAlmostEqual(distribution['B'], .2)
        self.assertEqual(distribution['C'], 0)
        self.assertEqual(len(logged), 30)
        self.assertLess(len(set(logged)), len(logged))
        negative = recruit([('A', -2), ('B', -1)], {'A': .5, 'B': .5}, return_probabilities=True)[1]
        self.assertAlmostEqual(negative['A'], 1/3)
        self.assertAlmostEqual(negative['B'], 2/3)

    def test_confidence_uses_full_distribution_not_observed_draws(self):
        f = features(['A', 'B', 'A'], ['small'] * 3, {'large': .8, 'small': .2})
        self.assertEqual(f['router_confidence'], .8)
        self.assertAlmostEqual(f['modal_disagreement'], 1/3)
        self.assertIsNone(features(['A'], ['small'])['router_confidence'])

    def test_malformed_routing_rejected(self):
        for value in ({}, {'A': -1, 'B': 2}, {'A': .5}, {'A': float('nan')}, {'A': True}):
            with self.assertRaises(ValueError):
                validate_router(value, ['A'])
        with self.assertRaises(ValueError):
            validate_router({'A': 1, 'B': 0}, ['B'])


if __name__ == '__main__':
    unittest.main()
