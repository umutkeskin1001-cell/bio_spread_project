import json
import logging
from pathlib import Path

import numpy as np
import torch

from bio_spread.data.loader import get_device, load_training_data, make_data_loader, make_sequence_dataset
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config, set_seed

logger = logging.getLogger(__name__)

def train_ensemble(config_path="config/default.yaml", feature_dir="data/features", n_members=5, ensemble_dir="artifacts/ensemble"):
    cfg = load_config(config_path)
    ensemble_dir = Path(ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    data = load_training_data(cfg, feature_dir)
    train_ds = make_sequence_dataset(data["train_df"], data["split"]["train"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    val_ds = make_sequence_dataset(data["sequences_df"], data["split"]["val"], cfg, data["seq_means"], data["seq_stds"], data["static_means"], data["static_stds"], data["use_taxonomy"])
    train_loader = make_data_loader(train_ds, cfg.training.batch_size, cfg.model.max_seq_len, shuffle=True)
    val_loader = make_data_loader(val_ds, cfg.training.batch_size, cfg.model.max_seq_len)
    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)
    device = get_device()
    t = cfg.training

    member_results = []
    for member_id in range(n_members):
        seed = t.seed + member_id
        set_seed(seed)
        logger.info("=" * 40)
        logger.info("Training ensemble member %d/%d (seed=%d)", member_id + 1, n_members, seed)
        logger.info("=" * 40)
        model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab=data.get("tax_vocab") if data["use_taxonomy"] else None)
        trainer = BioSpreadTrainer(model, device=device, lr=t.lr, weight_decay=t.weight_decay, epochs=t.epochs, patience=t.patience, warmup_epochs=t.warmup_epochs, grad_clip=t.grad_clip, lambda_count=t.lambda_count, lambda_rank=t.lambda_rank, lambda_cold=t.lambda_cold, lambda_all=t.lambda_all, lambda_gate=t.lambda_gate, temporal_masking_prob=t.temporal_masking_prob, gaussian_noise_std=t.gaussian_noise_std, calibrate=t.calibrate)
        artifact_dir = trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(val_loader)
        best_model_src = artifact_dir / "best_model.pt"
        if best_model_src.exists():
            torch.save(model.state_dict(), ensemble_dir / f"member_{member_id:02d}.pt")
        member_results.append({"member_id": member_id, "seed": seed, **{k: metrics.get(k, 0.0) for k in ("roc_auc", "pr_auc", "f1", "ece")}})
        logger.info("Member %d: AUC=%.4f F1=%.4f", member_id, metrics.get("roc_auc", 0.0), metrics.get("f1", 0.0))

    results = {"n_members": n_members, "members": member_results, "mean_auc": float(np.mean([m["roc_auc"] for m in member_results])), "std_auc": float(np.std([m["roc_auc"] for m in member_results]))}
    (ensemble_dir / "ensemble_results.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 50 + "\n  ENSEMBLE TRAINING RESULTS\n" + "=" * 50)
    for m in member_results:
        print(f"  Member {m['member_id']:2d}: AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  ECE={m['ece']:.4f}")
    print(f"  Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}\n  Models saved to: {ensemble_dir}\n" + "=" * 50)
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--ensemble-dir", default="artifacts/ensemble")
    args = parser.parse_args()
    train_ensemble(args.config, args.feature_dir, args.n_members, args.ensemble_dir)
