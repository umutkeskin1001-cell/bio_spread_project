from dna_sentinel.prepare import SequenceRecord, cluster_split


def kmer_jaccard(seq_a: str, seq_b: str, k: int = 5) -> float:
    set_a = {seq_a[i : i + k] for i in range(len(seq_a) - k + 1)}
    set_b = {seq_b[i : i + k] for i in range(len(seq_b) - k + 1)}
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def test_kmer_jaccard_detects_near_identical_sequences():
    a = "ACGT" * 20
    b = "ACGT" * 19 + "ACGA"
    c = "TTTT" * 20

    assert kmer_jaccard(a, b, k=5) >= 0.80
    assert kmer_jaccard(a, c, k=5) < 0.20


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
