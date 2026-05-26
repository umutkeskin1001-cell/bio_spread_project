from pathlib import Path

from dna_sentinel.utils import canonical_dna, circular_shift, read_fasta, revcomp, write_fasta


def test_canonical_dna_uppercases_and_masks():
    assert canonical_dna("acgturyk-nx") == "ACGTNNNNNN"


def test_revcomp_preserves_unknowns():
    assert revcomp("AACGTN") == "NACGTT"


def test_circular_shift_wraps():
    assert circular_shift("AACCGG", 2) == "CCGGAA"
    assert circular_shift("AACCGG", 8) == "CCGGAA"
    assert circular_shift("", 3) == ""


def test_read_and_write_fasta_roundtrip(tmp_path: Path):
    records = [("seq1", "ACGTACGT"), ("seq2", "NNNAAA")]
    path = tmp_path / "sample.fa"
    write_fasta(records, path)
    assert list(read_fasta(path)) == records


def test_canonical_dna_empty():
    assert canonical_dna("") == ""


def test_canonical_dna_skips_whitespace():
    assert canonical_dna("A C G T") == "ACGT"


def test_write_fasta_skips_empty(tmp_path):
    write_fasta([("empty", "")], tmp_path / "empty.fa")
    assert list(read_fasta(tmp_path / "empty.fa")) == []
