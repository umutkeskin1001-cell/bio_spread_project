from pathlib import Path

from click.testing import CliRunner

from dna_sentinel.cli import cli
from dna_sentinel.model import Cassiopeia


def test_cli_predict_returns_json_for_fasta(tmp_path: Path):
    model = Cassiopeia()
    ckpt = tmp_path / "cassiopeia.pt"
    model.save(ckpt)
    fasta = tmp_path / "query.fa"
    fasta.write_text(">query\nATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGT\n")
    result = CliRunner().invoke(cli, ["predict", "--checkpoint", str(ckpt), "--fasta", str(fasta), "--json"])
    assert result.exit_code == 0, result.output
    assert '"sequence_id": "query"' in result.output
    assert '"risk_score"' in result.output


def test_train_command_exposed_in_help():
    result = CliRunner().invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
