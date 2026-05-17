from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click
import yaml

from dna_sentinel.dataset import DnaDataset, load_jsonl
from dna_sentinel.fasta import read_fasta
from dna_sentinel.kmer import KmerConfig, KmerSentinel
from dna_sentinel.model import DnaSentinel, DnaSentinelConfig
from dna_sentinel.predict import predict_one
from dna_sentinel.prepare import prepare_dataset
from dna_sentinel.train import TrainConfig, evaluate, load_checkpoint, train_model


@click.group()
def cli() -> None:
    """DNA-only mobile genetic element risk modeling."""


@cli.command()
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def prepare(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    stats = prepare_dataset(**cfg["data"])
    click.echo(json.dumps(stats, indent=2))


@cli.command()
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def train(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["out_dir"])
    model_cfg = DnaSentinelConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["training"])
    train_ds = DnaDataset(load_jsonl(data_dir / "train.jsonl"), model_cfg.window_size, model_cfg.stride, model_cfg.max_windows)
    val_ds = DnaDataset(load_jsonl(data_dir / "val.jsonl"), model_cfg.window_size, model_cfg.stride, model_cfg.max_windows)
    model = DnaSentinel(model_cfg)
    ckpt, history = train_model(model, train_ds, val_ds, train_cfg)
    click.echo(json.dumps({"checkpoint": str(ckpt), "last": history[-1] if history else {}}, indent=2))


@cli.command("train-kmer")
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def train_kmer(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["out_dir"])
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "val.jsonl")
    kmer_cfg = KmerConfig(**cfg.get("kmer", {}))
    model = KmerSentinel.train(train_records, kmer_cfg)
    model.calibrate(val_records)
    artifact_dir = Path(cfg["training"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "kmer.joblib"
    model.save(path)
    metrics = model.evaluate(val_records)
    (artifact_dir / "kmer_val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    click.echo(json.dumps({"checkpoint": str(path), "validation": metrics}, indent=2))


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default="data/dna_sentinel")
def evaluate_cmd(checkpoint: str, data_dir: str) -> None:
    model = load_checkpoint(checkpoint)
    cfg = model.cfg
    ds = DnaDataset(load_jsonl(Path(data_dir) / "test.jsonl"), cfg.window_size, cfg.stride, cfg.max_windows)
    click.echo(json.dumps(evaluate(model, ds), indent=2))


cli.add_command(evaluate_cmd, "evaluate")


@cli.command("evaluate-kmer")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default="data/dna_sentinel")
def evaluate_kmer(checkpoint: str, data_dir: str) -> None:
    model = KmerSentinel.load(checkpoint)
    records = load_jsonl(Path(data_dir) / "test.jsonl")
    click.echo(json.dumps(model.evaluate(records), indent=2))


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fasta", "fasta_path", required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def predict(checkpoint: str, fasta_path: str, as_json: bool) -> None:
    model = load_checkpoint(checkpoint)
    rows = [asdict(predict_one(model, sid, seq)) for sid, seq in read_fasta(fasta_path)]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for row in rows:
            click.echo(f"{row['sequence_id']}\trisk={row['risk_score']:.4f}\tamr={row['amr_probability']:.4f}")


@cli.command("predict-kmer")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fasta", "fasta_path", required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def predict_kmer(checkpoint: str, fasta_path: str, as_json: bool) -> None:
    model = KmerSentinel.load(checkpoint)
    rows = [model.predict_one(sid, seq) for sid, seq in read_fasta(fasta_path)]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for row in rows:
            click.echo(f"{row['sequence_id']}\trisk={row['risk_score']:.4f}\tamr={row['amr_probability']:.4f}")


if __name__ == "__main__":
    cli()
