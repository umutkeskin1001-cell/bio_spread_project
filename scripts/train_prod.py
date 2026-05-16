"""
BioSpread Sovereign-X Ultra: Production Training Entrypoint.
Implements the Unified 4-Loss Formulation and Deterministic Cold-Start Path.
"""
import click
import torch
import logging
from pathlib import Path

from bio_spread.utils.config import load_config, set_seed
from bio_spread.data.dataset import SequenceDataset, sequence_collate
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from torch.utils.data import DataLoader
import polars as pl

logger = logging.getLogger(__name__)

@click.command()
@click.option("--config", default="config/prod.yaml")
@click.option("--output-dir", default="artifacts/prod_v4")
def main(config, output_dir):
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    set_seed(cfg.training.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    feature_dir = Path(cfg.data.feature_dir)
    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    
    # Simple split for demonstration
    backbones = seq_df["backbone_id"].unique().to_list()
    train_ids = backbones[:int(0.8*len(backbones))]
    val_ids = backbones[int(0.8*len(backbones)):]
    
    train_ds = SequenceDataset(seq_df, train_ids, cfg.model.max_seq_len)
    val_ds = SequenceDataset(seq_df, val_ids, cfg.model.max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, 
                              collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len))
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, 
                            collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len))

    # 2. Initialize Model
    # Get dims from dataset
    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)
    
    model = create_model(n_static, n_snapshot, cfg.model)
    model.to(device)
    
    logger.info(f"Initialized Sovereign-X Ultra with {n_static} static and {n_snapshot} temporal features.")

    # 3. Initialize Trainer with Production Objectives
    trainer = BioSpreadTrainer(
        model,
        device=device,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        use_adaptive_loss=cfg.training.use_adaptive_loss,
        use_curriculum=cfg.training.use_curriculum,
        calibrate=True,
        calibrate_cold=True
    )

    # 4. Fit
    logger.info("Starting Production Training...")
    trainer.fit(train_loader, val_loader)
    
    # 5. Final Calibration and Export
    torch.save(model.state_dict(), out_path / "best_model.pt")
    logger.info(f"Training complete. Model saved to {out_path}")

if __name__ == "__main__":
    main()
