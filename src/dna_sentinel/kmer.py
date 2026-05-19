from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from dna_sentinel.dataset import LabeledSequence
from dna_sentinel.fasta import canonical_dna, revcomp
from dna_sentinel.metrics import binary_metrics, multiclass_metrics
from dna_sentinel.tokenizer import window_sequence


@dataclass(frozen=True)
class KmerConfig:
    ngram_min: int = 5
    ngram_max: int = 6
    n_features: int = 65536
    alpha: float = 1e-4
    max_iter: int = 2000
    seed: int = 42
    window_size: int = 4096
    stride: int = 2048
    max_windows: int = 32
    rc_consensus: bool = True


class KmerSentinel:
    def __init__(
        self,
        config: KmerConfig,
        vectorizer: HashingVectorizer,
        mobility: Any,
        amr: Any,
        expansion: Any,
        mobility_calibrator: Any | None = None,
        amr_calibrator: Any | None = None,
        expansion_calibrator: Any | None = None,
    ) -> None:
        self.config = config
        self.vectorizer = vectorizer
        self.mobility = mobility
        self.amr = amr
        self.expansion = expansion
        self.mobility_calibrator = mobility_calibrator
        self.amr_calibrator = amr_calibrator
        self.expansion_calibrator = expansion_calibrator

    @classmethod
    def train(cls, records: list[LabeledSequence], config: KmerConfig | None = None) -> "KmerSentinel":
        cfg = config or KmerConfig()
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            n_features=cfg.n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        x = vectorizer.transform([r.dna for r in records])
        mobility = _fit_classifier(x, np.asarray([r.mobility for r in records]), cfg, binary=False)
        amr = _fit_classifier(x, np.asarray([r.amr for r in records]), cfg, binary=True)
        expansion = _fit_classifier(x, np.asarray([r.expansion for r in records]), cfg, binary=True)
        return cls(cfg, vectorizer, mobility, amr, expansion)

    def calibrate(self, records: list[LabeledSequence]) -> "KmerSentinel":
        if not records:
            return self
        mob_y = np.asarray([r.mobility for r in records])
        amr_y = np.asarray([r.amr for r in records])
        exp_y = np.asarray([r.expansion for r in records])
        mob_p = self._raw_predict_mobility([r.dna for r in records])
        amr_p = self._raw_predict_binary(self.amr, [r.dna for r in records])
        exp_p = self._raw_predict_binary(self.expansion, [r.dna for r in records])
        if len(np.unique(mob_y)) > 1:
            self.mobility_calibrator = LogisticRegression(max_iter=1000, random_state=self.config.seed).fit(mob_p, mob_y)
        if len(np.unique(amr_y)) > 1:
            self.amr_calibrator = LogisticRegression(max_iter=1000, random_state=self.config.seed).fit(_logit_feature(amr_p), amr_y)
        if len(np.unique(exp_y)) > 1:
            self.expansion_calibrator = LogisticRegression(max_iter=1000, random_state=self.config.seed).fit(_logit_feature(exp_p), exp_y)
        return self

    def evaluate(self, records: list[LabeledSequence]) -> dict[str, float]:
        mob_y = np.asarray([r.mobility for r in records])
        amr_y = np.asarray([r.amr for r in records])
        exp_y = np.asarray([r.expansion for r in records])
        mob_p = self._predict_mobility([r.dna for r in records])
        amr_p = self._predict_binary(self.amr, [r.dna for r in records])
        exp_p = self._predict_binary(self.expansion, [r.dna for r in records])
        metrics: dict[str, float] = {}
        metrics.update(multiclass_metrics(mob_y, mob_p, "mobility"))
        metrics.update(binary_metrics(amr_y, amr_p, "amr"))
        metrics.update(binary_metrics(exp_y, exp_p, "expansion"))
        return metrics

    def predict_one(self, sequence_id: str, dna: str) -> dict:
        dna = canonical_dna(dna)
        mobility_probs = self._predict_mobility([dna])[0].tolist()
        amr = float(self._predict_binary(self.amr, [dna])[0])
        expansion = float(self._predict_binary(self.expansion, [dna])[0])
        mobile = max(mobility_probs[1], mobility_probs[2]) if len(mobility_probs) >= 3 else 0.0
        risk = float(((mobile**2 + amr**2 + expansion**2) / 3.0)**0.5)
        return {
            "sequence_id": sequence_id,
            "mobility_probs": mobility_probs,
            "amr_probability": amr,
            "expansion_probability": expansion,
            "risk_score": risk,
            "top_windows": self._top_windows(dna),
        }

    def _top_windows(self, dna: str) -> list[dict[str, float]]:
        windows = window_sequence(dna, self.config.window_size, self.config.stride, self.config.max_windows)
        if not windows:
            return []
        x = self.vectorizer.transform(windows)
        amr = self.amr.predict_proba(x)[:, 1]
        exp = self.expansion.predict_proba(x)[:, 1]
        mob = self.mobility.predict_proba(x)[:, 1:].max(axis=1)
        score = np.sqrt((amr**2 + exp**2 + mob**2) / 3.0)
        order = np.argsort(-score)[: min(5, len(windows))]
        return [
            {
                "start": float(i * self.config.stride),
                "end": float(i * self.config.stride + len(windows[i])),
                "weight": float(score[i]),
            }
            for i in order
        ]



    def _predict_binary(self, clf: Any, sequences: list[str]) -> np.ndarray:
        p = self._raw_predict_binary(clf, sequences)
        calibrator = self.amr_calibrator if clf is self.amr else self.expansion_calibrator
        if calibrator is None:
            return p
        return calibrator.predict_proba(_logit_feature(p))[:, 1]

    def _raw_predict_binary(self, clf: Any, sequences: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(sequences)
        p = clf.predict_proba(x)[:, 1]
        if not self.config.rc_consensus:
            return p
        xr = self.vectorizer.transform([revcomp(seq) for seq in sequences])
        return 0.5 * (p + clf.predict_proba(xr)[:, 1])

    def _predict_mobility(self, sequences: list[str]) -> np.ndarray:
        p = self._raw_predict_mobility(sequences)
        if self.mobility_calibrator is None:
            return p
        return self.mobility_calibrator.predict_proba(p)

    def _raw_predict_mobility(self, sequences: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(sequences)
        p = self.mobility.predict_proba(x)
        if not self.config.rc_consensus:
            return p
        xr = self.vectorizer.transform([revcomp(seq) for seq in sequences])
        return 0.5 * (p + self.mobility.predict_proba(xr))

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "config": asdict(self.config),
                "vectorizer": self.vectorizer,
                "mobility": self.mobility,
                "amr": self.amr,
                "expansion": self.expansion,
                "mobility_calibrator": self.mobility_calibrator,
                "amr_calibrator": self.amr_calibrator,
                "expansion_calibrator": self.expansion_calibrator,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "KmerSentinel":
        state = joblib.load(path)
        return cls(
            KmerConfig(**state["config"]),
            state["vectorizer"],
            state["mobility"],
            state["amr"],
            state["expansion"],
            state.get("mobility_calibrator"),
            state.get("amr_calibrator"),
            state.get("expansion_calibrator"),
        )


def _fit_classifier(x, y: np.ndarray, cfg: KmerConfig, binary: bool):
    if len(np.unique(y)) < 2:
        clf = DummyClassifier(strategy="most_frequent")
        return clf.fit(x, y)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=cfg.alpha,
        class_weight="balanced",
        random_state=cfg.seed,
        max_iter=cfg.max_iter,
        tol=1e-4,
    )
    return clf.fit(x, y)


def _logit_feature(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)
