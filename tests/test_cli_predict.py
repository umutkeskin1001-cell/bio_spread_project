from pathlib import Path

from click.testing import CliRunner

from dna_sentinel.cli import cli
from dna_sentinel.dataset import DnaDataset, LabeledSequence
from dna_sentinel.model import DnaSentinel, DnaSentinelConfig
from dna_sentinel.train import TrainConfig, train_model


def test_cli_predict_returns_json_for_fasta(tmp_path: Path):
    records = [
        LabeledSequence("p1", "ATGCGT" * 20, 2, 1, 1),
        LabeledSequence("n1", "TTAACC" * 20, 0, 0, 0),
        LabeledSequence("m1", "CCCCGG" * 20, 1, 0, 0),
    ]
    ds = DnaDataset(records, window_size=48, stride=24, max_windows=4)
    model = DnaSentinel(DnaSentinelConfig(channels=16, layers=2, window_size=48, stride=24, max_windows=4))
    ckpt, _ = train_model(model, ds, ds, TrainConfig(epochs=1, batch_size=3, artifact_dir=tmp_path, seed=5))
    fasta = tmp_path / "query.fa"
    fasta.write_text(">query\nATGCGTATGCGTATGCGT\n")

    result = CliRunner().invoke(cli, ["predict", "--checkpoint", str(ckpt), "--fasta", str(fasta), "--json"])

    assert result.exit_code == 0, result.output
    assert '"sequence_id": "query"' in result.output
    assert '"risk_score"' in result.output
