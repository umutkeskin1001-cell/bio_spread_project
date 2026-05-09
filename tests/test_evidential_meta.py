import numpy as np

from bio_spread_project.evidential_meta import EvidentialMetaEstimator


def test_evidential_meta_estimator():
    X = np.random.rand(100, 10)
    y = (np.random.rand(100) > 0.5).astype(float)

    est = EvidentialMetaEstimator(input_dim=10)
    est.fit(X, y)

    probs = est.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1.0)
