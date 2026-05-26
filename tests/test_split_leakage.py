from dna_sentinel.prepare import SequenceRecord, _report_similarity, cluster_split


def test_cluster_split_keeps_near_duplicates_in_same_partition():
    records = [SequenceRecord("a", "ACGT" * 30, {}), SequenceRecord("b", "ACGT" * 29 + "ACGA", {}),
               SequenceRecord("c", "TTTT" * 30, {}), SequenceRecord("d", "CCCC" * 30, {})]
    split, _, _ = cluster_split(records, seed=7, threshold=0.80)
    location = {rid: name for name, ids in split.items() for rid in ids}
    assert location["a"] == location["b"] and set(split) == {"train", "val", "test"}


def test_cluster_split_empty():
    split, sk, _ = cluster_split([])
    assert split == {"train": [], "val": [], "test": []} and sk == []


def test_cluster_split_single():
    split, _, _ = cluster_split([SequenceRecord("a", "ACGT" * 30, {})])
    assert sum(len(v) for v in split.values()) == 1


def test_report_similarity():
    records = [SequenceRecord("a", "ACGT" * 30, {}), SequenceRecord("b", "ACGT" * 29 + "ACGA", {}),
               SequenceRecord("c", "TTTT" * 30, {})]
    split = {"train": ["c"], "val": ["a"], "test": ["b"]}
    report = _report_similarity(records, split)
    assert "max_train_test_jaccard" in report
    assert report["max_train_test_jaccard"] >= 0
