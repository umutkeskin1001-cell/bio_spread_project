"""Cassiopeia CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import torch
import yaml

from dna_sentinel.features import extract_features
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.prepare import prepare_dataset
from dna_sentinel.train import evaluate, train_cassiopeia
from dna_sentinel.utils import predict_one, read_fasta


def _load_data(data_dir: Path, name: str, n_struct: int = 0) -> dict:
    feat = torch.load(data_dir / f"{name}_features.pt", weights_only=True)
    lab = torch.load(data_dir / f"{name}_labels.pt", weights_only=True)
    if "struct_features" not in feat and n_struct > 0:
        B, W = feat["features"].shape[:2]
        feat["struct_features"] = feat["features"].new_zeros(B, W, n_struct)
    return {**feat, **lab}


def _make_model(cfg: dict):
    return Cassiopeia(CassiopeiaConfig.from_yaml(cfg))


@click.group()
def cli() -> None:
    """Cassiopeia: DNA-only plasmid risk modeling."""


@cli.command()
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def prepare(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    click.echo(json.dumps(prepare_dataset(**cfg.get("data", cfg)), indent=2))


@cli.command("prepare-features")
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def prepare_features(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    data_cfg = cfg.get("data", cfg)
    kt = cfg.get("features", {})
    extract_features(data_cfg["out_dir"], {**kt, **cfg.get("model", {})})


@cli.command("train")
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def train_cmd(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    data_dir = Path(cfg["data"]["out_dir"])
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    model = _make_model(cfg)
    ns = model_cfg.get("n_structural_features", 19)
    train_data = _load_data(data_dir, "train", ns)
    val_data = _load_data(data_dir, "val", ns)

    ckpt, history = train_cassiopeia(
        model, train_data, val_data,
        {**train_cfg, "artifact_dir": train_cfg.get("artifact_dir", "artifacts/default")},
    )
    click.echo(json.dumps({"checkpoint": str(ckpt), "last": history[-1] if history else {}}, indent=2))


@cli.command("evaluate")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default=None)
def evaluate_cmd(checkpoint: str, data_dir: str | None) -> None:
    model = Cassiopeia.load(checkpoint)
    if data_dir is None:
        data_dir = str(Path(checkpoint).parent.parent / "data" / "dna_sentinel")
    data = _load_data(Path(data_dir), "test", model.config.n_structural_features)
    click.echo(json.dumps(evaluate(model, data), indent=2))


@cli.command("predict")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fasta", "fasta_path", required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def predict_cmd(checkpoint: str, fasta_path: str, as_json: bool) -> None:
    from dataclasses import asdict
    model = Cassiopeia.load(checkpoint)
    rows = [asdict(predict_one(model, sid, seq)) for sid, seq in read_fasta(fasta_path)]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for r in rows:
            click.echo(f"{r['sequence_id']}\trisk={r['risk_score']:.4f}\tamr={r['amr_probability']:.4f}")


@cli.command("serve")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--port", default=8000, type=int)
@click.option("--host", default="0.0.0.0")
def serve(checkpoint: str, port: int, host: str) -> None:
    import uvicorn

    from dna_sentinel.api import app
    os.environ["CASSIOPEIA_CHECKPOINT"] = checkpoint
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
