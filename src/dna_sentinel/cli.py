"""Cassiopeia CLI — full command interface with short `dna` alias."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click
import numpy as np
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
    binary_metrics,
    bootstrap_ci,
    interpret_prediction,
    load_jsonl,
    logger,
    multiclass_metrics,
    predict_batch,
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


def _common_config_options(f):
    f = click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
                     help="Path to YAML config file")(f)
    return f


# ── main group ─────────────────────────────────────────────────────────

@click.group()
def cli():
    """Cassiopeia: DNA-only plasmid risk modeling.

    Predicts mobility class, AMR probability, and geographic expansion
    risk directly from raw FASTA sequence without BLAST or metadata.
    """


# ── dna shorthand group ────────────────────────────────────────────────

@click.group()
def dna():
    """Cassiopeia short alias. Usage: dna <command> [options].

    Commands: train, predict, bench, prep, features, cv, serve, list, interpret.
    """


# ── prepare ────────────────────────────────────────────────────────────

@cli.command()
@_common_config_options
def prepare(config):
    """Build dataset from FASTA + labels.

    Filters PLSDB-derived sequences, applies train/val/test/holdout
    split with 21-mer Jaccard-aware clustering (threshold >= 0.85).
    """
    cfg = _load_config(config)
    click.echo(json.dumps(prepare_dataset(**cfg.get("data", cfg)), indent=2))


@dna.command("prep")
@_common_config_options
def dna_prepare(config):
    """Alias: build dataset from FASTA + labels."""
    cfg = _load_config(config)
    click.echo(json.dumps(prepare_dataset(**cfg.get("data", cfg)), indent=2))


# ── prepare-features ───────────────────────────────────────────────────

@cli.command("prepare-features")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
def prepare_features(config):
    """Extract canonical k-mer + structural features.

    Generates multi-scale window features (k-mer counts, GC content,
    dinucleotide frequency, entropy) for all splits.
    """
    cfg = _load_config(config)
    _validate_config(cfg)
    extract_features(cfg["data"]["out_dir"], {**cfg.get("features", {}), **cfg.get("model", {})})


@dna.command("features")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
def dna_features(config):
    """Alias: extract canonical k-mer + structural features."""
    cfg = _load_config(config)
    _validate_config(cfg)
    extract_features(cfg["data"]["out_dir"], {**cfg.get("features", {}), **cfg.get("model", {})})


# ── train ──────────────────────────────────────────────────────────────

@cli.command("train")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--experiment", "-e", default=None, help="Experiment name for tracking")
def train_cmd(config, experiment):
    """Train the Cassiopeia model.

    Runs the full training loop with balanced sampling, SWA, consistency
    regularization, and L-BFGS calibration.
    """
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


@dna.command("train")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--experiment", "-e", default=None, help="Experiment name for tracking")
def dna_train(config, experiment):
    """Alias: train the Cassiopeia model."""
    train_cmd.callback(config=config, experiment=experiment)


# ── cross-validate ─────────────────────────────────────────────────────

@cli.command("cross-validate")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--folds", "-k", default=5, type=int, help="Number of CV folds")
def cross_validate_cmd(config, folds):
    """Run k-fold cross-validation.

    Reports mean ± std for mobility BA, AMR AUROC, expansion AUROC,
    and task score across folds. Optionally saves fold checkpoints.
    """
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


@dna.command("cv")
@click.option("--config", "-c", default="config/cassiopeia_prime.yaml", type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--folds", "-k", default=5, type=int, help="Number of CV folds")
def dna_cross_validate(config, folds):
    """Alias: run k-fold cross-validation."""
    cross_validate_cmd.callback(config=config, folds=folds)


# ── benchmark ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--checkpoint", "-m", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint (.pt)")
@click.option("--data-dir", "-d", default="data/dna_sentinel",
              help="Data directory with feature/label files")
@click.option("--out", "-o", default="artifacts/cassiopeia_prime_v14/report.json",
              help="Output path for benchmark report JSON")
@click.option("--rc-average/--no-rc-average", default=False,
              help="Use reverse-complement-averaged inference (matches "
                   "production dna predict). Slower (re-extracts features) "
                   "and requires .jsonl split files. Default uses cached "
                   "features and skips RC averaging for fast reproducible "
                   "benchmarks.")
@click.option("--ensemble-checkpoint", "-e", default=None, type=click.Path(exists=True),
              help="Optional second checkpoint for model ensemble")
@click.option("--ensemble-weight", type=float, default=0.5,
              help="Weight for the ensemble model (default: 0.5)")
def benchmark(checkpoint, data_dir, out, rc_average, ensemble_checkpoint=None, ensemble_weight=0.5):
    """Benchmark model on all splits.

    Evaluates mobility BA, AMR AUROC, expansion AUROC, and task score
    on validation, test, and held-out sets with bootstrap confidence
    intervals (95% CI, 500 resamples).
    """
    from dna_sentinel.utils import evaluate_records

    model = Cassiopeia.load(checkpoint)
    ensemble = None
    if ensemble_checkpoint:
        em = Cassiopeia.load(ensemble_checkpoint)
        ensemble = [(em, ensemble_weight)]
    root = Path(data_dir)
    SN = {"val": "validation", "heldout_test": "heldout"}
    report = {
        "checkpoint": checkpoint,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_mb": Path(checkpoint).stat().st_size / 1_000_000,
        "config": model.config.to_dict(),
        "inference_mode": "rc_averaged" if rc_average else "cached_features",
        "splits": {},
    }
    if ensemble:
        report["ensemble"] = {
            "checkpoint": ensemble_checkpoint,
            "weight": ensemble_weight,
        }
    fasta_query = "ATGCGT" * 1000
    start = time.perf_counter()
    predict_one(model, "bench_6kb", fasta_query, rc_average=False)
    report["fasta_latency_ms_6kb"] = 1000.0 * (time.perf_counter() - start)

    for split in ("val", "test", "heldout_test"):
        if rc_average:
            jsonl_path = root / f"{split}.jsonl"
            if not jsonl_path.exists():
                continue
            records = load_jsonl(jsonl_path)
            preds = predict_batch(
                model, [(r.sequence_id, r.dna) for r in records], device="cpu",
                rc_average=True, ensemble=ensemble,
            )
            mob = np.array([p.mobility_probs for p in preds])
            amr = np.array([p.amr_probability for p in preds])
            exp = np.array([p.expansion_probability for p in preds])
            y_mob = np.array([r.mobility for r in records])
            y_amr = np.array([float(r.amr) for r in records])
            y_exp = np.array([float(r.expansion) for r in records])
            metrics = evaluate_records(model, records, device="cpu")
        else:
            p = root / f"{split}_features.pt"
            if not p.exists():
                continue
            data = _load_data(root, split, model.config.n_structural_features)
            w1 = 1.0
            w2 = ensemble_weight if ensemble else 0.0
            total_w = w1 + w2
            _, mob, amr, exp = evaluate(model, data, return_probs=True)
            if ensemble:
                em = ensemble[0][0]
                _, mob2, amr2, exp2 = evaluate(em, data, return_probs=True)
                mob = (mob * w1 + mob2 * w2) / total_w
                amr = (amr * w1 + amr2 * w2) / total_w
                exp = (exp * w1 + exp2 * w2) / total_w
            y_mob = data["mobility"].cpu().numpy()
            y_amr = data["amr"].cpu().numpy().ravel()
            y_exp = data["expansion"].cpu().numpy().ravel()
            metrics = {}
            metrics.update(multiclass_metrics(y_mob, mob, "mobility"))
            metrics.update(binary_metrics(y_amr, amr, "amr"))
            metrics.update(binary_metrics(y_exp, exp, "expansion"))
        metrics["task_score"] = task_score(metrics)

        def _mob_fn(t, p):
            return balanced_accuracy_score(t, p.argmax(-1))

        def _task_fn(t, p):
            return (balanced_accuracy_score(t, p.argmax(-1))
                    + roc_auc_score(y_amr, amr)
                    + roc_auc_score(y_exp, exp)) / 3

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


@dna.command("bench")
@click.option("--checkpoint", "-m", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint (.pt)")
@click.option("--data-dir", "-d", default="data/dna_sentinel",
              help="Data directory with feature/label files")
@click.option("--out", "-o", default="artifacts/cassiopeia_prime_v14/report.json",
              help="Output path for benchmark report JSON")
@click.option("--rc-average/--no-rc-average", default=False,
              help="Use reverse-complement-averaged inference (matches "
                   "production dna predict). Default uses cached features.")
@click.option("--ensemble-checkpoint", "-e", default=None, type=click.Path(exists=True),
              help="Optional second checkpoint for model ensemble")
@click.option("--ensemble-weight", type=float, default=0.5,
              help="Weight for the ensemble model (default: 0.5)")
def dna_benchmark(checkpoint, data_dir, out, rc_average, ensemble_checkpoint, ensemble_weight):
    """Alias: benchmark model on all splits."""
    benchmark.callback(checkpoint=checkpoint, data_dir=data_dir, out=out, rc_average=rc_average,
                       ensemble_checkpoint=ensemble_checkpoint, ensemble_weight=ensemble_weight)


# ── champion paths ──────────────────────────────────────────────────────

_CHAMPION = "artifacts/cassiopeia_prime_v15/cassiopeia_best.pt"
_ENSEMBLE_PARTNER = "artifacts/cassiopeia_prime_v14/cassiopeia_best.pt"
_ENSEMBLE_WEIGHT = 0.53


def _load_model(checkpoint: str) -> Cassiopeia:
    return Cassiopeia.load(checkpoint)


def _load_ensemble(ensemble_ckpt: str | None) -> list[tuple[Cassiopeia, float]] | None:
    if ensemble_ckpt and Path(ensemble_ckpt).exists():
        try:
            em = _load_model(ensemble_ckpt)
            return [(em, _ENSEMBLE_WEIGHT)]
        except Exception:
            return None
    return None


def _auto_ensemble(checkpoint: str) -> str | None:
    """Auto-detect ensemble partner when primary model is the champion."""
    ckpt = str(Path(checkpoint).resolve())
    if ckpt.endswith("cassiopeia_prime_v15/cassiopeia_best.pt") and Path(_ENSEMBLE_PARTNER).exists():
        return _ENSEMBLE_PARTNER
    if ckpt.endswith("cassiopeia_prime_v14/cassiopeia_best.pt") and Path(_CHAMPION).exists():
        return _CHAMPION
    return None


# ── predict ────────────────────────────────────────────────────────────

@cli.command("predict")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--fasta", "-f", "fasta_path", required=True, type=click.Path(exists=True),
              help="Input FASTA file path")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON")
@click.option("--interpret", "-i", "with_interpret", is_flag=True,
              help="Include biological interpretation with confidence labels")
def predict_cmd(checkpoint, ensemble_checkpoint, fasta_path, as_json, with_interpret):
    """Predict on FASTA sequences.

    By default auto-detects the champion ensemble (v15+v14) when the primary
    checkpoint matches the champion path. Use --ensemble-checkpoint to specify
    a custom ensemble partner, or --no-ensemble-checkpoint to disable.
    """
    from dataclasses import asdict

    model = _load_model(checkpoint)
    if ensemble_checkpoint is None:
        ensemble_checkpoint = _auto_ensemble(checkpoint)
    ensemble = _load_ensemble(ensemble_checkpoint)
    rows = []
    for sid, seq in read_fasta(fasta_path):
        pred = asdict(predict_one(model, sid, seq, ensemble=ensemble))
        if with_interpret:
            pred["interpretation"] = interpret_prediction(pred, model.config.risk_weights)
        rows.append(pred)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    else:
        for r in rows:
            parts = [f"{r['sequence_id']}\trisk={r['risk_score']:.4f}",
                     f"amr={r['amr_probability']:.4f}",
                     f"exp={r['expansion_probability']:.4f}"]
            if with_interpret and "interpretation" in r:
                i = r["interpretation"]
                parts.append(f"mob={i['mobility']['label']}")
                parts.append(f"amr={i['amr']['confidence']}")
                parts.append(f"exp={i['expansion']['confidence']}")
            click.echo("\t".join(parts))


@dna.command("predict")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--fasta", "-f", "fasta_path", required=True, type=click.Path(exists=True),
              help="Input FASTA file path")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON")
@click.option("--interpret", "-i", "with_interpret", is_flag=True,
              help="Include biological interpretation with confidence labels")
def dna_predict(checkpoint, ensemble_checkpoint, fasta_path, as_json, with_interpret):
    """Alias: predict on FASTA sequences."""
    predict_cmd.callback(checkpoint=checkpoint, ensemble_checkpoint=ensemble_checkpoint,
                         fasta_path=fasta_path, as_json=as_json, with_interpret=with_interpret)


# ── serve ──────────────────────────────────────────────────────────────

@cli.command("serve")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--ensemble-weight", type=float, default=_ENSEMBLE_WEIGHT,
              help="Weight for ensemble model (default: 0.53)")
@click.option("--port", "-p", default=8080, type=int, help="HTTP port")
@click.option("--host", default="0.0.0.0", help="Bind address")
def serve_cmd(checkpoint, ensemble_checkpoint, ensemble_weight, port, host):
    """Start FastAPI inference server + web UI.

    Serves the static web interface at / and exposes /predict, /predict-batch,
    and /health endpoints. Ensemble auto-detected for champion checkpoint.
    """
    import uvicorn

    from dna_sentinel.api import app
    if ensemble_checkpoint is None:
        ensemble_checkpoint = _auto_ensemble(checkpoint)
    os.environ["CASSIOPEIA_CHECKPOINT"] = checkpoint
    os.environ["CASSIOPEIA_ENSEMBLE_CHECKPOINT"] = ensemble_checkpoint or ""
    os.environ["CASSIOPEIA_ENSEMBLE_WEIGHT"] = str(ensemble_weight)
    uvicorn.run(app, host=host, port=port, log_level="info")


@dna.command("serve")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--ensemble-weight", type=float, default=_ENSEMBLE_WEIGHT,
              help="Weight for ensemble model (default: 0.53)")
@click.option("--port", "-p", default=8080, type=int, help="HTTP port")
@click.option("--host", default="0.0.0.0", help="Bind address")
def dna_serve(checkpoint, ensemble_checkpoint, ensemble_weight, port, host):
    """Alias: start FastAPI inference server + web UI."""
    serve_cmd.callback(checkpoint=checkpoint, ensemble_checkpoint=ensemble_checkpoint,
                       ensemble_weight=ensemble_weight, port=port, host=host)


# ── experiment-list ────────────────────────────────────────────────────

@cli.command("experiment-list")
def experiment_list():
    """List recent experiment runs.

    Shows up to 20 most recent timestamped experiment directories.
    """
    base = Path("experiments")
    if not base.exists():
        click.echo("No experiments directory found.")
        return
    dirs = sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in dirs[:20]:
        click.echo(d)


@dna.command("list")
def dna_experiment_list():
    """Alias: list recent experiment runs."""
    experiment_list.callback()


# ── interpret ──────────────────────────────────────────────────────────

@cli.command("interpret")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--fasta", "-f", "fasta_path", required=True, type=click.Path(exists=True),
              help="Input FASTA file path")
def interpret_cmd(checkpoint, ensemble_checkpoint, fasta_path):
    """Predict with biological interpretation.

    Adds confidence labels (HIGH >= 0.80, MEDIUM 0.60-0.79, LOW < 0.60),
    CARD family matching for AMR, and mobility + AMR co-occurrence
    inference for expansion. Includes safety disclaimer.
    Auto-detects champion ensemble for maximum accuracy.
    """
    from dataclasses import asdict
    model = _load_model(checkpoint)
    if ensemble_checkpoint is None:
        ensemble_checkpoint = _auto_ensemble(checkpoint)
    ensemble = _load_ensemble(ensemble_checkpoint)
    for sid, seq in read_fasta(fasta_path):
        pred = asdict(predict_one(model, sid, seq, ensemble=ensemble))
        interp = interpret_prediction(pred, model.config.risk_weights)
        click.echo(json.dumps({"sequence_id": sid, **pred, "interpretation": interp}, indent=2, default=str))


@dna.command("interpret")
@click.option("--checkpoint", "-m", default=_CHAMPION, type=click.Path(exists=True),
              show_default=True, help="Path to model checkpoint (.pt)")
@click.option("--ensemble-checkpoint", "-e", default=None,
              type=click.Path(), help="Ensemble partner checkpoint (auto-detected for champion)")
@click.option("--fasta", "-f", "fasta_path", required=True, type=click.Path(exists=True),
              help="Input FASTA file path")
def dna_interpret(checkpoint, ensemble_checkpoint, fasta_path):
    """Alias: predict with biological interpretation."""
    interpret_cmd.callback(checkpoint=checkpoint, ensemble_checkpoint=ensemble_checkpoint,
                           fasta_path=fasta_path)


# ── entrypoints ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
