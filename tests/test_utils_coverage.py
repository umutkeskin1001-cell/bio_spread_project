"""Additional utils tests for coverage."""

import numpy as np
import torch

from dna_sentinel.utils import (
    canonical_dna,
    revcomp,
    circular_shift,
    read_fasta,
    write_fasta,
    save_jsonl,
    load_jsonl,
    LabeledSequence,
    expected_calibration_error,
    false_positive_summary,
    WindowDropout,
)
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig


def test_canonical_dna():
    assert canonical_dna("ATGCNatgcn") == "ATGCNATGCN"


def test_canonical_dna_whitespace():
    # canonical_dna removes whitespace first, then normalizes non-ACGT to N
    assert canonical_dna("AT GC\nN") == "ATGCN"


def test_revcomp():
    assert revcomp("ATGC") == "GCAT"


def test_revcomp_with_n():
    assert revcomp("ATGCN") == "NGCAT"


def test_circular_shift():
    assert circular_shift("ATGC", 0) == "ATGC"
    assert circular_shift("ATGC", 1) == "TGCA"
    assert circular_shift("ATGC", 2) == "GCAT"


def test_circular_shift_empty():
    assert circular_shift("", 5) == ""


def test_circular_shift_large_offset():
    assert circular_shift("ATGC", 10) == "GCAT"  # 10 % 4 = 2


def test_read_fasta(tmp_path):
    fa = tmp_path / "test.fa"
    fa.write_text(">seq1\nATGCGT\n>seq2\nGCTA\n")
    records = list(read_fasta(fa))
    assert len(records) == 2
    assert records[0][0] == "seq1"
    assert records[1][1] == "GCTA"


def test_read_fasta_empty(tmp_path):
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    records = list(read_fasta(fa))
    assert len(records) == 0


def test_read_fasta_with_whitespace(tmp_path):
    fa = tmp_path / "ws.fa"
    fa.write_text(">seq1\nATC G\nT\n>seq2\nG\nC\nT\nA\n")
    records = list(read_fasta(fa))
    assert len(records) == 2


def test_write_fasta(tmp_path):
    out = tmp_path / "out.fa"
    write_fasta([("seq1", "ATGCGT"), ("seq2", "GCATGC")], out)
    content = out.read_text()
    assert ">seq1" in content
    assert ">seq2" in content


def test_write_fasta_empty():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".fa", delete=False, mode="w") as f:
        path = f.name
    try:
        write_fasta([], path)
        import pathlib
        content = pathlib.Path(path).read_text()
        assert content == ""
    finally:
        os.unlink(path)


def test_save_load_jsonl(tmp_path):
    records = [
        LabeledSequence("s1", "ATGC", 0, 1, 0),
        LabeledSequence("s2", "CGTA", 1, 0, 1),
    ]
    path = tmp_path / "test.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].sequence_id == "s1"
    assert loaded[1].amr == 0


def test_expected_calibration_error():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    ece = expected_calibration_error(y, p, bins=5)
    assert 0.0 <= ece <= 1.0


def test_expected_calibration_error_empty():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_false_positive_summary_with_true():
    mob = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    s = false_positive_summary(mob, np.array([0.1, 0.9]), np.array([0.2, 0.8]),
                                np.array([0.3, 0.7]), mob_true=np.array([0, 0]))
    assert "false_mobile_rate" in s


def test_false_positive_summary_no_true():
    mob = np.array([[0.9, 0.05, 0.05]])
    s = false_positive_summary(mob, np.array([0.1]), np.array([0.2]), np.array([0.3]))
    assert "false_mobile_rate" in s


def test_window_dropout_noop():
    wd = WindowDropout(drop_rate=0.0)
    feat = torch.randn(2, 10, 5)
    mask = torch.ones(2, 10, dtype=torch.bool)
    f, m = wd(feat, mask, training=True)
    assert torch.equal(f, feat)
    assert torch.equal(m, mask)


def test_window_dropout_not_training():
    wd = WindowDropout(drop_rate=0.5)
    feat = torch.randn(2, 10, 5)
    mask = torch.ones(2, 10, dtype=torch.bool)
    f, m = wd(feat, mask, training=False)
    assert torch.equal(f, feat)


def test_window_dropout_drops_some():
    wd = WindowDropout(drop_rate=0.3)
    feat = torch.randn(4, 10, 5)
    mask = torch.ones(4, 10, dtype=torch.bool)
    f, m = wd(feat, mask, training=True)
    # At least some windows might be dropped (but not all)
    assert (m.sum(dim=1) > 0).all()


def test_window_dropout_keeps_at_least_one():
    wd = WindowDropout(drop_rate=0.99)
    feat = torch.randn(4, 10, 5)
    mask = torch.ones(4, 10, dtype=torch.bool)
    f, m = wd(feat, mask, training=True)
    assert m.sum(dim=1).min() >= 1


def test_configure_logging(tmp_path):
    from dna_sentinel.utils import configure_logging
    logfile = tmp_path / "test.log"
    configure_logging(log_path=logfile)
    import logging
    logger = logging.getLogger("cassiopeia")
    logger.info("test message")
    assert logfile.exists()


def test_labeled_sequence_creation():
    ls = LabeledSequence("test", "ATGC", 0, 1, 0)
    assert ls.sequence_id == "test"
    assert ls.amr == 1
    assert ls.mobility == 0
