"""Tests for dataset preparation module."""

from dna_sentinel.prepare import (
    _sampled_kmers,
    _jaccard,
    _DSU,
    _report_similarity,
    SequenceRecord,
    cluster_split,
)


def test_sampled_kmers_short_seq():
    result = _sampled_kmers("ATGC", k=4, n=10)
    assert len(result) == 1
    assert "ATGC" in result


def test_sampled_kmers_long_seq():
    seq = "ATGC" * 1000  # 4000 bp
    result = _sampled_kmers(seq, k=5, n=100)
    assert len(result) <= 100
    assert all(len(kmer) == 5 for kmer in result)


def test_sampled_kmers_exact():
    seq = "ATGC" * 10
    result = _sampled_kmers(seq, k=4, n=100)
    assert len(result) > 0


def test_jaccard_identical():
    s = {"A", "B", "C"}
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"A", "B"}, {"C", "D"}) == 0.0


def test_jaccard_partial():
    sim = _jaccard({"A", "B", "C"}, {"B", "C", "D"})
    assert abs(sim - 0.5) < 1e-6


def test_jaccard_empty():
    assert _jaccard(set(), set()) == 0.0


def test_dsu_find():
    dsu = _DSU(5)
    for i in range(5):
        assert dsu.find(i) == i


def test_dsu_union():
    dsu = _DSU(5)
    dsu.union(0, 1)
    dsu.union(2, 3)
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(0) != dsu.find(2)
    dsu.union(1, 2)
    assert dsu.find(0) == dsu.find(3)


def test_dsu_union_same():
    dsu = _DSU(3)
    dsu.union(0, 1)
    dsu.union(0, 1)
    assert dsu.find(0) == dsu.find(1)


def test_cluster_split_empty():
    result, sketches, clusters = cluster_split([], seed=42)
    assert result["train"] == []
    assert result["val"] == []
    assert result["test"] == []


def test_cluster_split_single():
    records = [SequenceRecord("seq1", "ATGC" * 100, {"mobility": 0, "amr": 0, "expansion": 0})]
    result, sketches, clusters = cluster_split(records, seed=42)
    assert len(result["train"]) == 1
    assert result["val"] == []
    assert result["test"] == []


def test_cluster_split_two():
    records = [
        SequenceRecord("seq1", "ATGC" * 100, {"mobility": 0, "amr": 0, "expansion": 0}),
        SequenceRecord("seq2", "GCAT" * 100, {"mobility": 1, "amr": 1, "expansion": 0}),
    ]
    result, sketches, clusters = cluster_split(records, seed=42)
    assert len(result["train"]) >= 1
    assert "seq1" in result["train"] or "seq1" in result["val"]


def test_report_similarity_empty():
    result = _report_similarity([], {"train": [], "val": [], "test": []})
    assert result["max_train_test_jaccard"] == 0.0
    assert result["top_similar_pairs"] == []


def test_report_similarity_with_data():
    records = [
        SequenceRecord("seq1", "ATGC" * 100, {}),
        SequenceRecord("seq2", "ATGC" * 100, {}),
    ]
    result = _report_similarity(records, {"train": ["seq1"], "val": ["seq2"]})
    assert result["max_train_test_jaccard"] >= 0.0
    assert isinstance(result["top_similar_pairs"], list)
