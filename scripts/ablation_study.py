import json
import logging
from pathlib import Path

from bio_spread.data.loader import get_device, load_training_data, make_data_loader, make_sequence_dataset
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)

def run_ablation(config_path="config/default.yaml", feature_dir="data/features"):
    cfg = load_config(config_path)
    data = load_training_data(cfg, feature_dir)
    train_ds = make_sequence_dataset(data["train_df"], data["split"]["train"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    val_ds = make_sequence_dataset(data["sequences_df"], data["split"]["val"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    train_loader = make_data_loader(train_ds, cfg.training.batch_size, cfg.model.max_seq_len, shuffle=True)
    val_loader = make_data_loader(val_ds, cfg.training.batch_size, cfg.model.max_seq_len)
    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)
    device = get_device()
    t = cfg.training

    ablations = {
        "full": {},
        "no_per_timestep": {"lambda_all": 0.0},
        "no_cold_start": {"lambda_cold": 0.0},
        "no_ranking": {"lambda_rank": 0.0},
        "no_count": {"lambda_count": 0.0},
        "no_gate_entropy": {"lambda_gate": 0.0},
        "no_temporal_masking": {"temporal_masking_prob": 0.0},
        "no_gaussian_noise": {"gaussian_noise_std": 0.0},
    }

    results = {}
    for name, overrides in ablations.items():
        set_seed(t.seed)
        logger.info("=" * 40)
        logger.info("Ablation: %s", name)
        logger.info("=" * 40)
        model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab=data.get("tax_vocab") if data["use_taxonomy"] else None)
        kwargs = dict(lr=t.lr, weight_decay=t.weight_decay, epochs=20, patience=5, warmup_epochs=t.warmup_epochs, grad_clip=t.grad_clip, lambda_count=t.lambda_count, lambda_rank=t.lambda_rank, lambda_cold=t.lambda_cold, lambda_all=t.lambda_all, lambda_gate=t.lambda_gate, temporal_masking_prob=t.temporal_masking_prob, gaussian_noise_std=t.gaussian_noise_std, calibrate=False)
        kwargs.update(overrides)
        trainer = BioSpreadTrainer(model, device=device, **kwargs)
        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(val_loader)
        results[name] = {k: metrics.get(k, 0.0) for k in ("roc_auc", "pr_auc", "f1", "ece", "n")}
        logger.info("%s: AUC=%.4f F1=%.4f ECE=%.4f", name, results[name]["roc_auc"], results[name]["f1"], results[name]["ece"])

    output_path = Path("artifacts/ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60 + "\n  ABLATION STUDY RESULTS\n" + "=" * 60)
    for name, m in results.items():
        delta = m["roc_auc"] - results["full"]["roc_auc"]
        print(f"  {name:25s} AUC={m['roc_auc']:.4f} (Δ={delta:+.4f})  F1={m['f1']:.4f}  ECE={m['ece']:.4f}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ablation()
