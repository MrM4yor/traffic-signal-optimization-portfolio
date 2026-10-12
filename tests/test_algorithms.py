"""Portable algorithm regressions exported from the reviewed main project."""

import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from blended_predictor import BlendedClassifier
from event_matching import optimal_pairs
import evolution_policy as policy
import itertools


class BlendTests(unittest.TestCase):
    def teacher(self, labels, probabilities):
        return SimpleNamespace(
            classes_=np.array(labels),
            n_features_in_=2,
            predict_proba=lambda x: np.tile(probabilities, (len(x), 1)),
        )

    def test_class_columns_are_aligned(self):
        b = BlendedClassifier(
            self.teacher([0, 1], [0.8, 0.2]), self.teacher([1, 0], [0.6, 0.4]), 0.5
        )
        np.testing.assert_allclose(b.predict_proba([[0, 0]]), [[0.6, 0.4]])

    def test_invalid_blends_rejected(self):
        with self.assertRaises(ValueError):
            BlendedClassifier(
                self.teacher([0, 1], [0.8, 0.2]), self.teacher([0, 2], [0.4, 0.6]), 0.5
            )
        with self.assertRaises(ValueError):
            BlendedClassifier(
                self.teacher([0, 1], [0.8, 0.2]), self.teacher([0, 1], [0.4, 0.6]), 2
            )


class PolicyTests(unittest.TestCase):
    def test_native_noop_and_shapes(self):
        for kind in ("linear", "polynomial", "mlp"):
            self.assertEqual(
                policy.action(policy.native_genome(kind), {}, architecture=kind).kind,
                "noop",
            )
        with self.assertRaises(ValueError):
            policy.forward(np.zeros(6), np.zeros(32), "unknown")
        with self.assertRaises(ValueError):
            policy.decode(np.zeros(5), 10, 10, 0.5)

    def test_alignment_preserves_donor_function(self):
        rng = np.random.default_rng(123)
        reference = rng.normal(size=policy.parameter_count("mlp", 4))
        donor = rng.normal(size=reference.size)
        aligned = policy.align_mlp_genome(reference, donor, 4)
        for _ in range(5):
            obs = rng.normal(size=len(policy.FEATURES))
            np.testing.assert_allclose(
                policy.forward(donor, obs, "mlp", 4),
                policy.forward(aligned, obs, "mlp", 4),
                atol=1e-12,
            )


class MatchingInputTests(unittest.TestCase):
    def test_numpy_events_and_invalid_values(self):
        self.assertEqual(len(optimal_pairs(np.array([1, 1]), np.array([1]), 0)), 1)
        for values, tolerance in [([float("nan")], 1), ([1], -1)]:
            with self.assertRaises(ValueError):
                optimal_pairs(values, [1], tolerance)


class MatchingTests(unittest.TestCase):
    def test_crossed_example(self):
        self.assertEqual(
            sum(abs(e) for i, j, e in optimal_pairs([710, 700], [700, 710])), 0
        )

    def test_duplicates_are_separate_events(self):
        pairs = optimal_pairs([700, 700, 710], [700, 710])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({i for i, j, e in pairs}), 2)
        self.assertEqual(len({j for i, j, e in pairs}), 2)

    def test_optimal_against_brute_force(self):
        for raw in ([0, 2, 5], [2, 2, 4], [0, 10, 20]):
            for sim in ([1, 3, 6], [2, 2], [1, 21, 30]):
                candidates = []
                for n in range(min(len(raw), len(sim)) + 1):
                    for rr in itertools.combinations(range(len(raw)), n):
                        for cc in itertools.permutations(range(len(sim)), n):
                            errs = [abs(raw[i] - sim[j]) for i, j in zip(rr, cc)]
                            if all(e <= 3 for e in errs):
                                candidates.append((-n, sum(errs)))
                pairs = optimal_pairs(raw, sim, 3)
                self.assertEqual(
                    (-len(pairs), sum(abs(e) for i, j, e in pairs)), min(candidates)
                )
