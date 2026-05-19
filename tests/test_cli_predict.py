from pathlib import Path

from click.testing import CliRunner

from dna_sentinel.cli import cli
from dna_sentinel.model import KmerTransformer, KmerTransformerConfig


def test_cli_predict_returns_json_for_fasta(tmp_path: Path):
    cfg = KmerTransformerConfig(hidden_dim=16, n_heads=2, n_layers=1, n_kmer_features=128)
    model = KmerTransformer(cfg)
    ckpt = tmp_path / "kmer_transformer.pt"
    model.save(ckpt)

    fasta = tmp_path / "query.fa"
    fasta.write_text(">query\nATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGTATGCGT\n")

    result = CliRunner().invoke(cli, ["predict", "--checkpoint", str(ckpt), "--fasta", str(fasta), "--json"])

    assert result.exit_code == 0, result.output
    assert '"sequence_id": "query"' in result.output
    assert '"risk_score"' in result.output
