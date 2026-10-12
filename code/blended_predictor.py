#!/usr/bin/env python3
"""Small runtime wrapper for blending two calibrated predictor teachers."""

from __future__ import annotations
import numpy as np


class BlendedClassifier:
    def __init__(self, first, second, alpha):
        """Blend teachers with identical class identities and input dimensions."""
        self.first = first
        self.second = second
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("Blend alpha must be finite and between zero and one.")
        self.classes_ = np.asarray(first.classes_)
        if set(self.classes_) != set(second.classes_):
            raise ValueError("Blended classifiers must predict the same classes.")
        if getattr(first, "n_features_in_", None) != getattr(
            second, "n_features_in_", None
        ):
            raise ValueError("Blended classifiers must use the same feature count.")
        self.n_jobs = 1

    def predict_proba(self, X):
        """Return probabilities aligned by class, including for old saved blends."""
        a = self.first.predict_proba(X)
        b = self.second.predict_proba(X)
        first_classes = np.asarray(self.first.classes_)
        second_classes = np.asarray(self.second.classes_)
        if set(first_classes) != set(second_classes):
            raise ValueError("Blended classifier class identities differ.")
        order = [
            int(np.flatnonzero(second_classes == label)[0]) for label in first_classes
        ]
        b = b[:, order]
        if a.shape != b.shape:
            raise ValueError("Blended classifier probability shapes differ.")
        return (1.0 - self.alpha) * a + self.alpha * b
