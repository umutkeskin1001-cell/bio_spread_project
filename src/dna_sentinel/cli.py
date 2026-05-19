from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click
import torch
import yaml

from dna_sentinel.features import MultiScaleKmerConfig, preprocess_all_features
from dna_sentinel.model import KmerTransformer, KmerTransformerConfig
from dna_sentinel.prepare import prepare_dataset
from dna_sentinel.train import evaluate_kmer_transformer, train_kmer_transformer
from dna_sentinel.utils import load_jsonl, predict_one, read_fasta


@click.group()
def cli() -> None:
    """DNA-only mobile genetic element risk modeling (Multi-Scale K-mer Transformer)."""


@cli.command()
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def prepare(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    stats = prepare_dataset(**cfg["data"])
    click.echo(json.dumps(stats, indent=2))


@cli.command("prepare-kmer-transformer")
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def prepare_kmer_transformer(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["out_dir"])
    kt_cfg = cfg.get("kmer_transformer", {})
    kmer_cfg = MultiScaleKmerConfig(
        ngram_min=kt_cfg.get("ngram_min", 4),
        ngram_max=kt_cfg.get("ngram_max", 6),
        n_features=kt_cfg.get("n_kmer_features", 4096),
        rc_consensus=kt_cfg.get("rc_consensus", True),
    )
    train_records = load_jsonl(data_dir / "train.jsonl")
    if not kmer_cfg.rc_consensus:
        from dna_sentinel.utils import rc_augment
        train_records = rc_augment(train_records)
    val_records = load_jsonl(data_dir / "val.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")
    preprocess_all_features(train_records, kmer_cfg, data_dir / "train_features.pt")
    preprocess_all_features(val_records, kmer_cfg, data_dir / "val_features.pt")
    preprocess_all_features(test_records, kmer_cfg, data_dir / "test_features.pt")
    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        torch.save({
            "mobility": torch.tensor([r.mobility for r in records], dtype=torch.long),
            "amr": torch.tensor([float(r.amr) for r in records]),
            "expansion": torch.tensor([float(r.expansion) for r in records]),
        }, data_dir / f"{name}_labels.pt")
    click.echo(json.dumps({"train_n": len(train_records), "val_n": len(val_records), "test_n": len(test_records)}, indent=2))


@cli.command("train-kmer-transformer")
@click.option("--config", default="config/dna_sentinel.yaml", type=click.Path(exists=True))
def train_kmer_transformer_cmd(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data"]["out_dir"])
    kt_cfg = cfg.get("kmer_transformer", {})
    model_cfg = KmerTransformerConfig(
        n_kmer_features=kt_cfg.get("n_kmer_features", 4096),
        hidden_dim=kt_cfg.get("hidden_dim", 64),
        n_heads=kt_cfg.get("n_heads", 4),
        n_layers=kt_cfg.get("n_layers", 2),
        ffn_ratio=kt_cfg.get("ffn_ratio", 4),
        dropout=kt_cfg.get("dropout", 0.25),
        max_windows=sum(kt_cfg.get("max_windows", [16, 8, 4])),
        n_scales=len(kt_cfg.get("window_sizes", [512, 2048, 8192])),
    )
    model = KmerTransformer(model_cfg)

    def load_split(name):
        feat = torch.load(data_dir / f"{name}_features.pt", weights_only=True)
        lab = torch.load(data_dir / f"{name}_labels.pt", weights_only=True)
        return {**feat, **lab}
    ckpt, history = train_kmer_transformer(model, load_split("train"), load_split("val"), {
        "epochs": kt_cfg.get("epochs", 15),
        "batch_size": kt_cfg.get("batch_size", 32),
        "lr": kt_cfg.get("lr", 1e-3),
        "backbone_lr": kt_cfg.get("lr", 1e-3),
        "head_lr": kt_cfg.get("lr", 1e-3),
        "weight_decay": kt_cfg.get("weight_decay", 0.05),
        "patience": kt_cfg.get("patience", 5),
        "window_dropout": kt_cfg.get("window_dropout", 0.25),
        "artifact_dir": cfg["training"]["artifact_dir"],
    })
    click.echo(json.dumps({"checkpoint": str(ckpt), "last": history[-1] if history else {}}, indent=2))


@cli.command("evaluate-kmer-transformer")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default="data/dna_sentinel")
def evaluate_kmer_transformer_cmd(checkpoint: str, data_dir: str) -> None:
    model = KmerTransformer.load(checkpoint)
    data_path = Path(data_dir)
    feat = torch.load(data_path / "test_features.pt", weights_only=True)
    lab = torch.load(data_path / "test_labels.pt", weights_only=True)
    click.echo(json.dumps(evaluate_kmer_transformer(model, {**feat, **lab}), indent=2))


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fasta", "fasta_path", required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def predict(checkpoint: str, fasta_path: str, as_json: bool) -> None:
    model = KmerTransformer.load(checkpoint)
    rows = [asdict(predict_one(model, sid, seq)) for sid, seq in read_fasta(fasta_path)]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for row in rows:
            click.echo(f"{row['sequence_id']}\trisk={row['risk_score']:.4f}\tamr={row['amr_probability']:.4f}")


# Compatibility Aliases
cli.add_command(train_kmer_transformer_cmd, "train")
cli.add_command(evaluate_kmer_transformer_cmd, "evaluate")


if __name__ == "__main__":
    cli()
