from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold


class PredictiveModel(Protocol):
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None: ...

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class DeterministicEngine:
    cv_seed: int = 7
    model_seed: int = 42

    def split(self, y: NDArray[np.int64], *, n_splits: int = 5) -> StratifiedKFold:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.cv_seed)


def cross_validate_estimator(
    *,
    model_factory: Callable[[], PredictiveModel],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    engine: DeterministicEngine,
    n_splits: int = 5,
) -> NDArray[np.float64]:
    oof = np.zeros(len(y), dtype=float)
    cv = engine.split(y, n_splits=n_splits)
    for train_idx, valid_idx in cv.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
    return oof
