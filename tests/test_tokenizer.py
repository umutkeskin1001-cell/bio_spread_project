import torch

from dna_sentinel.fasta import revcomp
from dna_sentinel.tokenizer import DnaTokenizer, window_sequence


def test_tokenizer_encodes_raw_bases_with_padding():
    tok = DnaTokenizer()
    ids, mask = tok.encode("ACGTN", max_len=8)

    assert ids.tolist() == [1, 2, 3, 4, 0, 0, 0, 0]
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_window_sequence_covers_short_and_long_sequences():
    short = window_sequence("ACGT", window=8, stride=4, max_windows=4)
    long = window_sequence("A" * 20, window=8, stride=4, max_windows=3)

    assert short == ["ACGT"]
    assert len(long) == 3
    assert all(len(w) == 8 for w in long)


def test_reverse_complement_token_ids_are_deterministic():
    tok = DnaTokenizer()
    seq = "AACCGGTTN"
    ids_a, _ = tok.encode(revcomp(seq), max_len=16)
    ids_b, _ = tok.encode("NAACCGGTT", max_len=16)

    assert torch.equal(ids_a, ids_b)
