"""Unit tests for the shortcut metric primitives (pure functions, no data)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shortcut_metrics import ndcg_graded, rouge_l_f1, split_sentences


class TestRougeLF1(unittest.TestCase):
    def test_identical_tokens(self):
        toks = ["the", "cat", "sat"]
        self.assertAlmostEqual(rouge_l_f1(toks, toks), 1.0)

    def test_disjoint_tokens(self):
        self.assertEqual(rouge_l_f1(["a", "b"], ["x", "y", "z"]), 0.0)

    def test_empty(self):
        self.assertEqual(rouge_l_f1([], ["a"]), 0.0)
        self.assertEqual(rouge_l_f1(["a"], []), 0.0)

    def test_partial_subsequence(self):
        # LCS = ["the", "dog"] in both -> F1 = 2*2*2/(2+2)... both len 3 -> 4/6
        s = ["the", "quick", "dog"]
        r = ["the", "dog", "ran"]
        self.assertAlmostEqual(rouge_l_f1(s, r), 4.0 / 6.0)


class TestNdcgGraded(unittest.TestCase):
    def test_perfectly_ranked(self):
        self.assertAlmostEqual(ndcg_graded([1.0, 0.5, 0.0]), 1.0)

    def test_reversed_ranking(self):
        v = ndcg_graded([0.0, 0.5, 1.0])
        self.assertLess(v, 1.0)
        self.assertGreater(v, 0.0)

    def test_all_zero(self):
        self.assertEqual(ndcg_graded([0.0, 0.0]), 0.0)

    def test_order_permutation(self):
        rel = [1.0, 0.0, 0.5]
        best = sorted(range(len(rel)), key=lambda i: -rel[i])
        self.assertAlmostEqual(ndcg_graded(rel, best), 1.0)


class TestSplitSentences(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(split_sentences("One. Two."), ["One.", "Two."])

    def test_no_trailing_period(self):
        self.assertEqual(split_sentences("Just words"), ["Just words"])

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])

    def test_keeps_internal_caps(self):
        out = split_sentences("First sentence. Second with API calls.")
        self.assertEqual(len(out), 2)


class TestVerdictHelper(unittest.TestCase):
    def test_bands(self):
        from src.build_aggregates import VERDICT
        self.assertEqual(VERDICT(0.894), "STRONG")
        self.assertEqual(VERDICT(-0.7), "STRONG")
        self.assertEqual(VERDICT(0.5), "MODERATE")
        self.assertEqual(VERDICT(0.18), "WEAK")
        self.assertEqual(VERDICT(-0.184), "WEAK")


if __name__ == "__main__":
    unittest.main()
