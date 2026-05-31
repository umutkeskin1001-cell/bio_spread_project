"""Cassiopeia CLI."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from dna_sentinel.features import FEATURE_SCHEMA_VERSION, extract_features
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.prepare import prepare_dataset
from dna_sentinel.train import cross_validate, evaluate, train_cassiopeia
from dna_sentinel.utils import (
    CassiopeiaExperiment,
    ConfigError,
    bootstrap_ci,
    logger,
    predict_one,
    read_fasta,
    task_score,
)


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _validate_config(cfg: dict) -> None:
    m = cfg.get("model", {})
    f = cfg.get("features", {})
    if m.get("max_windows") and f.get("max_windows"):
        mw = m["max_windows"]
        fw = sum(f["max_windows"])
        if mw != fw:
            raise ConfigError(f"feature max_windows sum ({fw}) != model max_windows ({mw})")
    if f.get("window_sizes") and f.get("strides"):
        if len(f["window_sizes"]) != len(f["strides"]):
            raise ConfigError(f"window_sizes ({len(f['window_sizes'])}) != strides ({len(f['strides'])})")


def _load_data(data_dir: Path, name: str, n_struct: int = 0) -> dict:
    feat = torch.load(data_dir / f"{name}_features.pt", weights_only=True)
    lab = torch.load(data_dir / f"{name}_labels.pt", weights_only=True)
    cached_schema = feat.get("_schema_version")
    cached_ns = feat.get("_n_structural_features")
    if cached_schema and cached_schema != FEATURE_SCHEMA_VERSION:
        raise ConfigError(
            f"Feature cache schema mismatch: cached={cached_schema!r}, expected={FEATURE_SCHEMA_VERSION!r}. "
            f"Re-run `prepare-features` to regenerate the cache."
        )
    if cached_ns is not None and n_struct and cached_ns != n_struct:
        raise ConfigError(
            f"Feature cache n_structural_features mismatch: cached={cached_ns}, model expects={n_struct}. "
            f"Re-run `prepare-features` to regenerate the cache."
        )
    if "struct_features" not in feat and n_struct:
        feat["struct_features"] = feat["features"].new_zeros(*feat["features"].shape[:2], n_struct)
    cons = data_dir / f"{name}_consistency_features.pt"
    if cons.exists():
        cached = torch.load(cons, weights_only=True)
        cs_schema = cached.get("_schema_version")
        cs_ns = cached.get("_n_structural_features")
        if cs_schema and cs_schema != FEATURE_SCHEMA_VERSION:
            logger.warning("Consistency cache schema mismatch: cached=%s, expected=%s. Re-run prepare-features.",
                           cs_schema, FEATURE_SCHEMA_VERSION)
        if cs_ns is not None and n_struct and cs_ns != n_struct:
            logger.warning("Consistency cache n_structural_features mismatch: cached=%d, expected=%d.",
                           cs_ns, n_struct)
        for k in ("features", "masks", "scale_ids"):
            if k in cached:
                feat[f"consistency_{k}"] = cached[k]
        if "struct_features" in cached:
            feat["consistency_struct_features"] = cached["struct_features"]
        elif n_struct:
            feat["consistency_struct_features"] = feat["features"].new_zeros(*cached["features"].shape[:2], n_struct)
    return {**feat, **lab}


def _make_model(cfg: dict):
    return Cassiopeia(CassiopeiaConfig.from_yaml(cfg))


@click.group()
def cli():
    """Cassiopeia: DNA-only plasmid risk modeling."""


@cli.command()
@click.option("--config", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True))
def prepare(config):
    cfg = _load_config(config)
    click.echo(json.dumps(prepare_dataset(**cfg.get("data", cfg)), indent=2))


@cli.command("prepare-features")
@click.option("--config", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True))
def prepare_features(config):
    cfg = _load_config(config)
    _validate_config(cfg)
    extract_features(cfg["data"]["out_dir"], {**cfg.get("features", {}), **cfg.get("model", {})})


@cli.command("train")
@click.option("--config", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True))
@click.option("--experiment", default=None, help="Experiment name for tracking")
def train_cmd(config, experiment):
    cfg = _load_config(config)
    _validate_config(cfg)
    data_dir = Path(cfg["data"]["out_dir"])
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})
    model = _make_model(cfg)
    ns = mc.get("n_structural_features", 49)
    train_data = _load_data(data_dir, "train", ns)
    val_data = _load_data(data_dir, "val", ns)
    exp = None
    if experiment:
        exp = CassiopeiaExperiment(experiment, config=cfg)
    ckpt, history = train_cassiopeia(
        model,
        train_data,
        val_data,
        {**tc, "artifact_dir": tc.get("artifact_dir", "artifacts/default"), "experiment": exp},
    )
    click.echo(json.dumps({"checkpoint": str(ckpt), "last": history[-1] if history else {}}, indent=2))


@cli.command("cross-validate")
@click.option("--config", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True))
@click.option("--folds", default=5, type=int)
def cross_validate_cmd(config, folds):
    cfg = _load_config(config)
    _validate_config(cfg)
    data_dir = Path(cfg["data"]["out_dir"])
    all_data = _load_data(data_dir, "train", cfg.get("model", {}).get("n_structural_features", 49))
    val_data = _load_data(data_dir, "val", cfg.get("model", {}).get("n_structural_features", 49))
    for k, v in val_data.items():
        if isinstance(v, torch.Tensor) and k in all_data:
            all_data[k] = torch.cat([all_data[k], v])
    split_path = data_dir / "split.json"
    train_ids = None
    if split_path.exists():
        split_info = json.loads(split_path.read_text())
        all_ids = split_info.get("train", []) + split_info.get("val", [])
        cluster_map = split_info.get("_cluster_map", {})
        if cluster_map and all_ids:
            train_ids = [cluster_map.get(sid, i) for i, sid in enumerate(all_ids)]
    result, fold_models = cross_validate(
        cfg.get("model", {}), all_data, cfg.get("training", {}),
        n_folds=folds, group_ids=train_ids, save_folds=True,
    )
    out_path = Path(cfg.get("training", {}).get("artifact_dir", "artifacts/default")) / "cv_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    if fold_models:
        cv_dir = Path(cfg.get("training", {}).get("artifact_dir", "artifacts/default")) / "cv_folds"
        cv_dir.mkdir(parents=True, exist_ok=True)
        fold_info = {"folds": [str(m) for m in fold_models]}
        (cv_dir / "fold_models.json").write_text(json.dumps(fold_info, indent=2))
        logger.info(f"Saved {len(fold_models)} fold checkpoints for ensemble")
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command("benchmark")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default="data/dna_sentinel")
@click.option("--out", default="artifacts/cassiopeia_prime/report.json")
def benchmark_cmd(checkpoint, data_dir, out):
    model = Cassiopeia.load(checkpoint)
    root = Path(data_dir)
    SN = {"val": "validation", "heldout_test": "heldout"}
    report = {
        "checkpoint": checkpoint,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_mb": Path(checkpoint).stat().st_size / 1_000_000,
        "config": model.config.to_dict(),
        "splits": {},
    }
    fasta_query = "ATGCGT" * 1000
    start = time.perf_counter()
    predict_one(model, "bench_6kb", fasta_query)
    report["fasta_latency_ms_6kb"] = 1000.0 * (time.perf_counter() - start)

    for split in ("val", "test", "heldout_test"):
        p = root / f"{split}_features.pt"
        if not p.exists():
            continue
        data = _load_data(root, split, model.config.n_structural_features)
        metrics, mob, amr, exp = evaluate(model, data, return_probs=True)
        metrics["task_score"] = task_score(metrics)
        y_mob = data["mobility"].cpu().numpy()
        y_amr = data["amr"].cpu().numpy().ravel()
        y_exp = data["expansion"].cpu().numpy().ravel()

        def _mob_fn(t, p):
            return balanced_accuracy_score(t, p.argmax(-1))

        def _task_fn(t, p):
            return (balanced_accuracy_score(t, p.argmax(-1)) + roc_auc_score(y_amr, amr) + roc_auc_score(y_exp, exp)) / 3

        ci_specs = {
            "mobility_balanced_accuracy": (y_mob, mob, _mob_fn),
            "amr_auroc": (y_amr, amr, roc_auc_score),
            "expansion_auroc": (y_exp, exp, roc_auc_score),
            "task_score": (y_mob, mob, _task_fn),
        }
        ci_metrics = {}
        for k, (yt, yp, fn) in ci_specs.items():
            try:
                pt, lo, hi = bootstrap_ci(yt, yp, fn, n_resamples=500)
                ci_metrics[k] = {"point": float(pt), "ci_95": [float(lo), float(hi)]}
            except Exception:
                ci_metrics[k] = {"point": float(metrics.get(k, 0.0)), "ci_95": None}
        metrics["_bootstrap_ci"] = ci_metrics
        report["splits"][SN.get(split, split)] = metrics
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    click.echo(json.dumps(report, indent=2, default=str))


@cli.command("predict")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--fasta", "fasta_path", required=True, type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True)
def predict_cmd(checkpoint, fasta_path, as_json):
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
def serve(checkpoint, port, host):
    import uvicorn

    from dna_sentinel.api import app

    os.environ["CASSIOPEIA_CHECKPOINT"] = checkpoint
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command("experiment-list")
def experiment_list():
    base = Path("experiments")
    if not base.exists():
        click.echo("No experiments directory found.")
        return
    dirs = sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in dirs[:20]:
        click.echo(d)


if __name__ == "__main__":
    cli()
