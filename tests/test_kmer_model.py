from pathlib import Path

from dna_sentinel.dataset import LabeledSequence
from dna_sentinel.fasta import revcomp
from dna_sentinel.kmer import KmerConfig, KmerSentinel


def test_kmer_sentinel_trains_saves_loads_and_predicts(tmp_path: Path):
    records = [
        LabeledSequence("p1", "ATGCGT" * 20, 2, 1, 1),
        LabeledSequence("p2", "ATGCGT" * 18, 2, 1, 1),
        LabeledSequence("n1", "TTAACC" * 20, 0, 0, 0),
        LabeledSequence("n2", "TTAACC" * 18, 0, 0, 0),
        LabeledSequence("m1", "CCCCGG" * 20, 1, 0, 0),
        LabeledSequence("m2", "GGCCCC" * 20, 1, 0, 0),
    ]

    model = KmerSentinel.train(records, KmerConfig(n_features=1024, max_iter=200))
    metrics = model.evaluate(records)
    path = tmp_path / "kmer.joblib"
    model.save(path)
    loaded = KmerSentinel.load(path)
    pred = loaded.predict_one("q", "ATGCGT" * 10)

    assert path.exists()
    assert 0.0 <= metrics["amr_auroc"] <= 1.0
    assert pred["sequence_id"] == "q"
    assert 0.0 <= pred["risk_score"] <= 1.0
    assert pred["top_windows"]


def test_kmer_sentinel_prediction_is_reverse_complement_consistent():
    records = [
        LabeledSequence("p1", "ATGCGT" * 20, 2, 1, 1),
        LabeledSequence("p2", "ATGCGT" * 18, 2, 1, 1),
        LabeledSequence("n1", "TTAACC" * 20, 0, 0, 0),
        LabeledSequence("n2", "TTAACC" * 18, 0, 0, 0),
        LabeledSequence("m1", "CCCCGG" * 20, 1, 0, 0),
        LabeledSequence("m2", "GGCCCC" * 20, 1, 0, 0),
    ]
    model = KmerSentinel.train(records, KmerConfig(n_features=1024, max_iter=200, rc_consensus=True))
    seq = "ATGCGT" * 10

    a = model.predict_one("a", seq)
    b = model.predict_one("b", revcomp(seq))

    assert abs(a["risk_score"] - b["risk_score"]) < 1e-9
    assert abs(a["amr_probability"] - b["amr_probability"]) < 1e-9
