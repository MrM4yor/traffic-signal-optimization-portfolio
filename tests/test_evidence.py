"""Check evidence arithmetic and provenance without rerunning any experiment."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo


class EvidenceTests(unittest.TestCase):
    def test_export_hashes_and_main_revision(self):
        record = demo.verify_exports()
        self.assertEqual(len(record['main_commit']), 40)
        self.assertEqual(len(record['exports']), 13)

    def test_percentages_recompute_from_totals(self):
        data = demo.read('evidence/controllers.json')['primary']
        for package in ['mlp', 'neural_ranker']:
            actual = demo.reductions(data['totals'], package)
            expected = data['summary'][f'improvement_{package}_vs_portfolio_pct']
            for key, value in actual.items():
                self.assertTrue(math.isclose(value, expected[key], abs_tol=1e-10))

    def test_tradeoffs_and_rejection_are_visible(self):
        report = demo.build_report()
        self.assertIn('-5.70%', report)
        self.assertIn('-2.68%', report)
        data = demo.read('evidence/reconstruction.json')
        self.assertTrue(data['accepted']['accepted_constraints'])
        self.assertFalse(data['trial']['accepted_constraints'])
        self.assertLess(data['trial']['groups']['internal']['mae_s'], data['accepted']['groups']['internal']['mae_s'])


if __name__ == '__main__':
    unittest.main()
