from pathlib import Path

from dna_sentinel.fasta import canonical_dna, read_fasta, revcomp, write_fasta


def test_canonical_dna_uppercases_and_masks_ambiguous_bases():
    assert canonical_dna("acgturyk-nx") == "ACGTNNNNNN"


def test_revcomp_preserves_unknowns_and_reverses_orientation():
    assert revcomp("AACGTN") == "NACGTT"


def test_read_and_write_fasta_roundtrip(tmp_path: Path):
    records = [("seq1", "ACGTACGT"), ("seq2", "NNNAAA")]
    path = tmp_path / "sample.fa"

    write_fasta(records, path)

    assert list(read_fasta(path)) == records
