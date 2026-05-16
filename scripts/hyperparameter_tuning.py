import json
import logging
from pathlib import Path

from bio_spread.data.loader import get_device, load_training_data, make_data_loader, make_sequence_dataset
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)

def objective(trial, cfg, train_loader, val_loader, n_static, n_snapshot, device, tax_vocab):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    lambda_count = trial.suggest_float("lambda_count", 0.0, 0.5)
    lambda_rank = trial.suggest_float("lambda_rank", 0.0, 0.5)
    lambda_cold = trial.suggest_float("lambda_cold", 0.0, 0.5)
    lambda_all = trial.suggest_float("lambda_all", 0.0, 2.0)
    lambda_gate = trial.suggest_float("lambda_gate", 0.0, 0.2)
    temporal_masking_prob = trial.suggest_float("temporal_masking_prob", 0.0, 0.5)
    gaussian_noise_std = trial.suggest_float("gaussian_noise_std", 0.0, 0.1)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    gru_hidden = trial.suggest_categorical("gru_hidden", [128, 192, 256])

    trial_model_cfg = cfg.model.model_copy(update={"dropout": dropout, "gru_hidden": gru_hidden})
    set_seed(cfg.training.seed)
    model = create_model(n_static, n_snapshot, trial_model_cfg, taxonomy_vocab=tax_vocab)
    trainer = BioSpreadTrainer(model, device=device, lr=lr, weight_decay=weight_decay, epochs=cfg.training.epochs, patience=cfg.training.patience, warmup_epochs=cfg.training.warmup_epochs, grad_clip=cfg.training.grad_clip, lambda_count=lambda_count, lambda_rank=lambda_rank, lambda_cold=lambda_cold, lambda_all=lambda_all, lambda_gate=lambda_gate, temporal_masking_prob=temporal_masking_prob, gaussian_noise_std=gaussian_noise_std, calibrate=False)
    trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(val_loader)
    return metrics.get("roc_auc", 0.0)

def tune_hyperparameters(config_path="config/default.yaml", feature_dir="data/features", n_trials=30, study_name="bio_spread_tuning"):
    cfg = load_config(config_path)
    data = load_training_data(cfg, feature_dir)
    train_ds = make_sequence_dataset(data["train_df"], data["split"]["train"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    val_ds = make_sequence_dataset(data["sequences_df"], data["split"]["val"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    train_loader = make_data_loader(train_ds, cfg.training.batch_size, cfg.model.max_seq_len, shuffle=True)
    val_loader = make_data_loader(val_ds, cfg.training.batch_size, cfg.model.max_seq_len)
    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)
    device = get_device()
    tax_vocab = data.get("tax_vocab") if data["use_taxonomy"] else None

    import optuna
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=None, load_if_exists=False)
    study.optimize(lambda trial: objective(trial, cfg, train_loader, val_loader, n_static, n_snapshot, device, tax_vocab), n_trials=n_trials)

    output_path = Path("artifacts/hyperparameter_tuning.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"best_params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials)}
    output_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 50 + "\n  HYPERPARAMETER TUNING RESULTS\n" + "=" * 50 + f"\n  Best AUC: {study.best_value:.4f}\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:25s} = {v}")
    print("=" * 50)
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="bio_spread_tuning")
    args = parser.parse_args()
    tune_hyperparameters(args.config, args.feature_dir, args.n_trials, args.study_name)
