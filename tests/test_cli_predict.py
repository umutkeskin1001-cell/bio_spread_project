import subprocess
import sys
from pathlib import Path

import torch
from click.testing import CliRunner

from dna_sentinel.cli import _load_data, cli
from dna_sentinel.model import Cassiopeia


def test_cli_predict_returns_json_for_fasta(tmp_path: Path):
    model = Cassiopeia()
    ckpt = tmp_path / "cassiopeia.pt"
    model.save(ckpt)
    fasta = tmp_path / "query.fa"
    fasta.write_text(">query\nATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGT\n")
    result = CliRunner().invoke(cli, ["predict", "--checkpoint", str(ckpt), "--fasta", str(fasta), "--json"])
    assert result.exit_code == 0 and '"sequence_id": "query"' in result.output and '"risk_score"' in result.output


def test_train_command_exposed_in_help():
    result = CliRunner().invoke(cli, ["train", "--help"])
    assert result.exit_code == 0 and "--config" in result.output


def test_inference_utils_import_without_sklearn():
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise ImportError('sklearn intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import dna_sentinel.utils
print('ok')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_load_data_includes_consistency_cache_when_present(tmp_path: Path):
    base = {"features": torch.randn(2, 3, 4), "masks": torch.ones(2, 3, dtype=torch.bool), "scale_ids": torch.zeros(2, 3, dtype=torch.long)}
    labels = {"mobility": torch.tensor([0, 1], dtype=torch.long), "amr": torch.tensor([0.0, 1.0]), "expansion": torch.tensor([0.0, 1.0])}
    cons = {"features": torch.randn(2, 3, 4), "masks": torch.ones(2, 3, dtype=torch.bool), "struct_features": torch.randn(2, 3, 19), "scale_ids": torch.zeros(2, 3, dtype=torch.long)}
    for name, d in [("train_features.pt", base), ("train_labels.pt", labels), ("train_consistency_features.pt", cons)]:
        torch.save(d, tmp_path / name)
    data = _load_data(tmp_path, "train", 19)
    assert "consistency_features" in data and data["consistency_features"].shape == cons["features"].shape


def test_cli_prepare_features_validates_config(tmp_path: Path):
    cfg = tmp_path / "bad_cfg.yaml"
    cfg.write_text("model:\n  max_windows: 28\nfeatures:\n  max_windows: [10, 10, 10]\n")
    result = CliRunner().invoke(cli, ["prepare-features", "--config", str(cfg)])
    assert result.exit_code != 0
