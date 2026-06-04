"""CLI-focused coverage tests using synthetic checkpoints."""

import json
import tempfile
from pathlib import Path
import torch

# ── Test benchmark with dummy checkpoint ──────────────────────────

def _make_dummy_checkpoint(tmp_path):
    """Create a minimal valid checkpoint for testing."""
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    path = tmp_path / "dummy.pt"
    model.save(path)
    return path


def test_benchmark_with_dummy_model(tmp_path):
    """Test benchmark runs with a real (dummy) model and synthetic data."""
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    # Match model's n_structural_features to cache
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56,
                                         n_structural_features=0))
    ckpt = tmp_path / "dummy.pt"
    model.save(ckpt)
    feat = {
        "features": torch.randn(4, 56, 2728),
        "masks": torch.ones(4, 56, dtype=torch.bool),
        "struct_features": torch.zeros(4, 56, 0),
        "_schema_version": "v9.0",
        "_n_structural_features": 0,
    }
    for name in ("val", "test", "heldout_test"):
        torch.save(feat, tmp_path / f"{name}_features.pt")
        lab = {
            "mobility": torch.randint(0, 3, (4,)),
            "amr": torch.randint(0, 2, (4,)).float(),
            "expansion": torch.randint(0, 2, (4,)).float(),
        }
        torch.save(lab, tmp_path / f"{name}_labels.pt")
    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--checkpoint", str(ckpt),
                            "--data-dir", str(tmp_path),
                            "--out", str(tmp_path / "report.json")])
    assert r.exit_code == 0, f"Failed: {r.output[:200]}"
    report = json.loads((tmp_path / "report.json").read_text())
    assert "splits" in report
    assert "parameters" in report


def test_benchmark_with_missing_split(tmp_path):
    """Test benchmark with only one split available."""
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56,
                                         n_structural_features=0))
    ckpt = tmp_path / "dummy.pt"
    model.save(ckpt)
    feat = {
        "features": torch.randn(2, 56, 2728),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "struct_features": torch.zeros(2, 56, 0),
        "_schema_version": "v9.0",
        "_n_structural_features": 0,
    }
    torch.save(feat, tmp_path / "val_features.pt")
    lab = {"mobility": torch.randint(0, 3, (2,)), "amr": torch.randint(0, 2, (2,)).float(),
           "expansion": torch.randint(0, 2, (2,)).float()}
    torch.save(lab, tmp_path / "val_labels.pt")
    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--checkpoint", str(ckpt),
                            "--data-dir", str(tmp_path)])
    assert r.exit_code == 0


def test_predict_with_dummy_model(tmp_path):
    """Test predict with a real dummy model."""
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    ckpt = _make_dummy_checkpoint(tmp_path)
    fa = tmp_path / "test.fa"
    fa.write_text(">seq1\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(cli, ["predict", "--checkpoint", str(ckpt),
                            "--fasta", str(fa), "--json"])
    assert r.exit_code == 0
    output = json.loads(r.output)
    assert len(output) == 1
    assert output[0]["sequence_id"] == "seq1"


def test_predict_with_interpret(tmp_path):
    """Test predict with --interpret flag."""
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    ckpt = _make_dummy_checkpoint(tmp_path)
    fa = tmp_path / "test.fa"
    fa.write_text(">seq1\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(cli, ["predict", "--checkpoint", str(ckpt),
                            "--fasta", str(fa), "--json", "--interpret"])
    assert r.exit_code == 0
    output = json.loads(r.output)
    assert "interpretation" in output[0]
    assert "disclaimer" in output[0]["interpretation"]


def test_dna_predict_with_dummy(tmp_path):
    """Test dna predict alias works."""
    from click.testing import CliRunner
    from dna_sentinel.cli import dna
    ckpt = _make_dummy_checkpoint(tmp_path)
    fa = tmp_path / "test.fa"
    fa.write_text(">seq1\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(dna, ["predict", "-m", str(ckpt), "-f", str(fa)])
    assert r.exit_code == 0


def test_dna_interpret_with_dummy(tmp_path):
    """Test dna interpret command."""
    from click.testing import CliRunner
    from dna_sentinel.cli import dna
    ckpt = _make_dummy_checkpoint(tmp_path)
    fa = tmp_path / "test.fa"
    fa.write_text(">seq1\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(dna, ["interpret", "-m", str(ckpt), "-f", str(fa)])
    assert r.exit_code == 0
    result = json.loads(r.output)
    assert "interpretation" in result


def test_dna_bench_with_dummy(tmp_path):
    """Test dna bench alias works."""
    from click.testing import CliRunner
    from dna_sentinel.cli import dna
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56,
                                         n_structural_features=0))
    ckpt = tmp_path / "dummy.pt"
    model.save(ckpt)
    feat = {
        "features": torch.randn(2, 56, 2728),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "struct_features": torch.zeros(2, 56, 0),
        "_schema_version": "v9.0",
        "_n_structural_features": 0,
    }
    torch.save(feat, tmp_path / "val_features.pt")
    lab = {"mobility": torch.randint(0, 3, (2,)), "amr": torch.randint(0, 2, (2,)).float(),
           "expansion": torch.randint(0, 2, (2,)).float()}
    torch.save(lab, tmp_path / "val_labels.pt")
    runner = CliRunner()
    r = runner.invoke(dna, ["bench", "-m", str(ckpt), "-d", str(tmp_path)])
    assert r.exit_code == 0


# ── Train edge cases ──────────────────────────────────────────────

def test_epoch_indices_no_balanced():
    from dna_sentinel.train import _epoch_indices
    gen = torch.Generator().manual_seed(0)
    data = {"mobility": torch.randint(0, 3, (10,)), "amr": torch.randint(0, 2, (10,)).float(),
            "expansion": torch.randint(0, 2, (10,)).float()}
    idx = _epoch_indices(5, data, {"balanced_sampling": False}, gen)
    assert idx.shape == (5,)


def test_epoch_indices_balanced_cached():
    from dna_sentinel.train import _epoch_indices
    gen = torch.Generator().manual_seed(0)
    data = {"mobility": torch.randint(0, 3, (10,)), "amr": torch.randint(0, 2, (10,)).float(),
            "expansion": torch.randint(0, 2, (10,)).float()}
    idx = _epoch_indices(20, data, {"balanced_sampling": True}, gen, cached_weights=torch.ones(10))
    assert idx.shape == (20,)


def test_build_optimizer_no_group():
    from dna_sentinel.train import _build_optimizer
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    opt = _build_optimizer(model, {"lr": 1e-3, "weight_decay": 0.05})
    assert len(opt.param_groups) >= 1


# ── prepare.py: build_labels with edge cases ─────────────────────

def test_build_labels_edge_cases(tmp_path):
    from dna_sentinel.prepare import build_labels
    bb = tmp_path / "bb.tsv"
    bb.write_text("sequence_accession\tpredicted_mobility\tbackbone_id\tcountry\tresolved_year\n"
                  "s1\tunknown\tb1\tTR\t0\ns2\tmobilizable\tb2\tUS;UK\t2020\n")
    amr = tmp_path / "amr.tsv"
    amr.write_text("sequence_accession\tamr_any\ns2\t1\n")
    labels = build_labels(str(bb), str(amr))
    assert "s1" in labels
    assert labels["s1"]["mobility"] == 0  # unknown -> 0
    assert labels["s2"]["amr"] == 1


def test_build_labels_multiple_countries(tmp_path):
    from dna_sentinel.prepare import build_labels
    bb = tmp_path / "bb.tsv"
    bb.write_text("sequence_accession\tpredicted_mobility\tbackbone_id\tcountry\tresolved_year\n"
                  "s1\tconjugative\tb1\tTR\t2020\n"
                  "s2\tconjugative\tb1\tUS\t2020\n"
                  "s3\tconjugative\tb1\tDE\t2020\n")
    amr = tmp_path / "amr.tsv"
    amr.write_text("sequence_accession\tamr_any\ns1\t1\ns2\t0\ns3\t1\n")
    labels = build_labels(str(bb), str(amr), expansion_country_threshold=3)
    # backbone b1 has 3 countries
    assert labels["s1"]["expansion"] == 1
    assert labels["s1"]["amr"] == 1


def test_cluster_split_three_groups():
    from dna_sentinel.prepare import cluster_split, SequenceRecord
    records = [
        SequenceRecord(f"s{i}", "ATCG" * 100, {"mobility": 0, "amr": 0, "expansion": 0})
        for i in range(5)
    ]
    result, sketches, clusters = cluster_split(records, seed=0)
    total = sum(len(v) for v in result.values())
    assert total == 5


def test_cluster_split_with_group():
    from dna_sentinel.prepare import cluster_split, SequenceRecord
    records = [
        SequenceRecord("s1", "ATCG" * 100, {"group": "g1"}),
        SequenceRecord("s2", "ATCG" * 100, {"group": "g1"}),
    ]
    result, sketches, clusters = cluster_split(records, seed=0)
    total = sum(len(v) for v in result.values())
    assert total == 2


# ── evaluate with expansion multiclass ───────────────────────────

def test_evaluate_expansion_multiclass():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    from dna_sentinel.train import evaluate
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=3)
    model = Cassiopeia(cfg)
    model.eval()
    data = {
        "features": torch.randn(4, 56, 2728),
        "masks": torch.ones(4, 56, dtype=torch.bool),
        "mobility": torch.randint(0, 3, (4,)),
        "amr": torch.randint(0, 2, (4,)).float(),
        "expansion": torch.randint(0, 3, (4,)),
    }
    metrics = evaluate(model, data, "cpu")
    assert "expansion_auroc" in metrics
