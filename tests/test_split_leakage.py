from dna_sentinel.prepare import SequenceRecord, cluster_split


def test_cluster_split_keeps_near_duplicates_in_same_partition():
    records = [
        SequenceRecord("a", "ACGT" * 30, {"amr": 1}),
        SequenceRecord("b", "ACGT" * 29 + "ACGA", {"amr": 1}),
        SequenceRecord("c", "TTTT" * 30, {"amr": 0}),
        SequenceRecord("d", "CCCC" * 30, {"amr": 0}),
    ]
    split = cluster_split(records, seed=7, threshold=0.80)
    location = {rid: name for name, ids in split.items() for rid in ids}
    assert location["a"] == location["b"]
    assert set(split) == {"train", "val", "test"}
