"""Minimal Random Forest classifier built on sklearn.tree.DecisionTreeClassifier.

sklearn.ensemble (RandomForestClassifier) is unusable on this machine: its
package __init__ unconditionally imports the hist-gradient-boosting and svm
extensions, both blocked by a Windows Application Control policy. sklearn.tree
alone imports fine, so this reimplements the standard bagging + per-split
feature-subsampling procedure (the actual "random forest" algorithm) around it.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestLite:
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = 12,
        min_samples_leaf: int = 5,
        max_features="sqrt",
        class_weight="balanced",
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestLite":
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for i in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        proba_sum = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees_:
            tree_proba = np.zeros((len(X), len(self.classes_)))
            p = tree.predict_proba(X)
            tree_classes = tree.classes_
            for j, c in enumerate(tree_classes):
                tree_proba[:, np.searchsorted(self.classes_, c)] = p[:, j]
            proba_sum += tree_proba
        return proba_sum / len(self.trees_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        imps = np.stack([t.feature_importances_ for t in self.trees_])
        return imps.mean(axis=0)
