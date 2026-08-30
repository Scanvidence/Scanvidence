"""Classical baseline feature selectors for benchmarking against QUBO."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from .base import BaseSelector


class LassoSelector(BaseSelector):
    """Feature selection via L1-regularized Logistic Regression (LASSO)."""

    def __init__(self, budget: int = 50, C: float = 0.1, random_state: int = 42):
        super().__init__(budget)
        self.C = C
        self.random_state = random_state

    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        features = np.asarray(features)
        labels = np.asarray(labels)
        n_features = features.shape[1]
        target_budget = min(self.budget, n_features)

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
        )
        clf.fit(features, labels)
        coef = np.abs(clf.coef_).ravel()
        sorted_idx = np.argsort(coef)[::-1]
        return sorted(sorted_idx[:target_budget].tolist())


class RFESelector(BaseSelector):
    """Feature selection via Recursive Feature Elimination."""

    def __init__(self, budget: int = 50, random_state: int = 42):
        super().__init__(budget)
        self.random_state = random_state

    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        features = np.asarray(features)
        labels = np.asarray(labels)
        n_features = features.shape[1]
        target_budget = min(self.budget, n_features)

        estimator = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            random_state=self.random_state,
            max_iter=500,
        )
        rfe = RFE(estimator=estimator, n_features_to_select=target_budget, step=0.1)
        rfe.fit(features, labels)
        return sorted(np.where(rfe.support_)[0].tolist())


class MutualInfoSelector(BaseSelector):
    """Feature selection via Mutual Information (SelectKBest)."""

    def __init__(self, budget: int = 50, random_state: int = 42):
        super().__init__(budget)
        self.random_state = random_state

    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        features = np.asarray(features)
        labels = np.asarray(labels)
        n_features = features.shape[1]
        target_budget = min(self.budget, n_features)

        mi = mutual_info_classif(features, labels, random_state=self.random_state)
        sorted_idx = np.argsort(mi)[::-1]
        return sorted(sorted_idx[:target_budget].tolist())


class MRMRSelector(BaseSelector):
    """Minimum Redundancy Maximum Relevance (mRMR) greedy feature selector."""

    def __init__(self, budget: int = 50, random_state: int = 42):
        super().__init__(budget)
        self.random_state = random_state

    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        features = np.asarray(features)
        labels = np.asarray(labels)
        n_features = features.shape[1]
        target_budget = min(self.budget, n_features)

        mi = mutual_info_classif(features, labels, random_state=self.random_state)
        selected: list[int] = [int(np.argmax(mi))]

        corr = np.abs(np.corrcoef(features.T))
        corr = np.nan_to_num(corr, nan=0.0)

        for _ in range(target_budget - 1):
            remaining = [i for i in range(n_features) if i not in selected]
            scores = []
            for candidate in remaining:
                rel = mi[candidate]
                red = np.mean([corr[candidate, s] for s in selected])
                score = rel - red
                scores.append(score)
            best_candidate = remaining[int(np.argmax(scores))]
            selected.append(best_candidate)

        return sorted(selected)
