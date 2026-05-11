import click
import torch
import json
import logging
from pathlib import Path
import polars as pl
from torch.utils.data import DataLoader
import numpy as np

from bio_spread_reborn.data.loader import DataPipeline
from bio_spread_reborn.data.tokenizer import GeneTokenizer
from bio_spread_reborn.data.dataset import BioDataset
from bio_spread_reborn.data.snapshot import TemporalSnapshotBuilder
from bio_spread_reborn.models.evidential import SovereignNet
from bio_spread_reborn.models.trainer import EvidentialTrainer
from bio_spread_reborn.utils.config import load_config
from bio_spread_reborn.utils.metrics import compute_metrics
from bio_spread_reborn.utils.logging import setup_logging

logger = logging.getLogger("BioSpreadCLI")

@click.group()
def cli():
    """BioSpread Reborn: Elite-grade plasmid spread surveillance."""
    setup_logging()

@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
def snapshot(config):
    """Build temporal snapshots to eliminate future leakage."""
    cfg = load_config(config)
    logger.info(f"Loading raw data for snapshot building...")
    
    # Load raw records
    raw_records = pl.read_csv(cfg.data.records_path, separator='\t')
    
    # Rename resolved_year if it exists
    if 'resolved_year' in raw_records.columns and 'year' not in raw_records.columns:
        raw_records = raw_records.rename({'resolved_year': 'year'})
    
    # If year is missing, join it
    if 'year' not in raw_records.columns:
        backbones = pl.read_csv(cfg.data.backbones_path, separator='\t', columns=['backbone_id', 'resolved_year'])
        backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
        raw_records = raw_records.join(backbones, on='backbone_id', how='left').rename({'resolved_year': 'year'})
    
    # Load metadata and AMR for builder (if needed for labeling logic)
    backbone_meta = pl.read_csv(cfg.data.backbones_path, separator='\t')
    amr = pl.read_csv(cfg.data.amr_path, separator='\t')
    
    builder = TemporalSnapshotBuilder(raw_records, backbone_meta, amr)
    
    cache_dir = Path(cfg.data.snapshot_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "snapshot_records.tsv"
    
    builder.build_snapshot(output_path)
    click.echo(f"Snapshot created at {output_path}")

@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
def train(config):
    """Train the SovereignNet model using the evidential learning pipeline."""
    cfg = load_config(config)
    logger.info("Initializing Data Pipeline...")
    
    pipeline = DataPipeline(cfg)
    
    # 1. Load and Tokenize Genetic Map
    genetic_df = pipeline.load_genetic_map()
    tokenizer = GeneTokenizer(max_len=cfg.model.max_genes)
    tokenizer.fit(genetic_df['gene_list'])
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Pre-tokenize entire genetic map once (Efficient O(N))
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    # 2. Use snapshots if available, otherwise fallback to records_path
    snapshot_path = Path(cfg.data.snapshot_cache_dir) / "snapshot_records.tsv"
    records_to_load = str(snapshot_path) if snapshot_path.exists() else cfg.data.records_path
    
    if not snapshot_path.exists():
        logger.warning("Snapshot not found! Using raw records (LEAKAGE RISK). Run 'snapshot' first.")
    
    train_df, val_df = pipeline.prepare_dataset(records_path=records_to_load)
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 3. Create Loaders
    train_ds = BioDataset(train_df, genetic_map, max_genes=cfg.model.max_genes)
    val_ds = BioDataset(val_df, genetic_map, max_genes=cfg.model.max_genes)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)
    
    # 4. Initialize Model
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        time_freqs=cfg.model.time_freqs
    )
    
    # 5. Initialize Trainer
    trainer = EvidentialTrainer(model, cfg.model_dump())
    
    # 6. Save Vocab in artifact dir BEFORE training for safety
    tokenizer.save(trainer.artifact_dir / 'tokenizer.json')
    
    # 7. Fit
    artifact_dir = trainer.fit(train_loader, val_loader)
    logger.info(f"Training complete. Artifacts saved in {artifact_dir}")

@cli.command()
@click.option('--model-path', required=True, help='Path to model weights (.pt)')
@click.option('--tokenizer-path', required=True, help='Path to tokenizer (.json)')
@click.option('--input-path', required=True, help='Path to TSV with backbone records')
@click.option('--output-path', default='predictions.json', help='Path to save results')
@click.option('--config', default='config/default.yaml', help='Path to config file')
def predict(model_path, tokenizer_path, input_path, output_path, config):
    """Run inference on new backbone records."""
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Tokenizer and Model
    tokenizer = GeneTokenizer.load(tokenizer_path)
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        time_freqs=cfg.model.time_freqs
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Load Data
    pipeline = DataPipeline(cfg)
    genetic_df = pipeline.load_genetic_map()
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    input_df = pl.read_csv(input_path, separator='\t')
    
    # Ensure year exists
    if 'year' not in input_df.columns:
        backbones = pl.read_csv(cfg.data.backbones_path, separator='\t', columns=['backbone_id', 'resolved_year'])
        backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
        input_df = input_df.join(backbones, on='backbone_id', how='left').rename({'resolved_year': 'year'})
    
    dataset = BioDataset(input_df, genetic_map, max_genes=cfg.model.max_genes)
    loader = DataLoader(dataset, batch_size=32)
    
    # 3. Predict
    results = []
    # Fetch backbone IDs once for efficient mapping
    all_bids = input_df['backbone_id'].to_list()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, t = batch[0].to(device), batch[1].to(device)
            prob, unc, _ = model(x, t)
            
            # Efficiently map back to IDs
            start_idx = i * 32
            batch_ids = all_bids[start_idx : start_idx + len(x)]
            
            results.extend([
                {"backbone_id": bid, "spread_probability": float(p), "uncertainty": float(u)}
                for bid, p, u in zip(batch_ids, prob.cpu().tolist(), unc.cpu().tolist())
            ])
            
    # 4. Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    click.echo(f"Predictions saved to {output_path}")

@cli.command()
@click.option('--model-path', required=True)
@click.option('--tokenizer-path', required=True)
@click.option('--input-path', required=True)
@click.option('--config', default='config/default.yaml')
def evaluate(model_path, tokenizer_path, input_path, config):
    """Evaluate model performance on a labeled dataset."""
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GeneTokenizer.load(tokenizer_path)
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        time_freqs=cfg.model.time_freqs
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    pipeline = DataPipeline(cfg)
    genetic_df = pipeline.load_genetic_map()
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    input_df = pl.read_csv(input_path, separator='\t')
    if 'year' not in input_df.columns:
        backbones = pl.read_csv(cfg.data.backbones_path, separator='\t', columns=['backbone_id', 'resolved_year'])
        backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
        input_df = input_df.join(backbones, on='backbone_id', how='left').rename({'resolved_year': 'year'})
    
    input_df = input_df.filter(pl.col('spread_label').is_not_null())
    dataset = BioDataset(input_df, genetic_map, max_genes=cfg.model.max_genes)
    loader = DataLoader(dataset, batch_size=32)
    
    all_probs, all_uncs, all_targets = [], [], []
    with torch.no_grad():
        for x, t, y in loader:
            x, t = x.to(device), t.to(device)
            prob, unc, _ = model(x, t)
            all_probs.extend(prob.cpu().tolist())
            all_uncs.extend(unc.cpu().tolist())
            all_targets.extend(y.tolist())
                
    m = compute_metrics(np.array(all_targets), np.array(all_probs), np.array(all_uncs))
    
    click.echo("\n" + "="*40)
    click.echo("       BIO-SPREAD EVALUATION REPORT")
    click.echo("="*40)
    click.echo(f"ROC AUC:          {m['auc']:.4f}")
    click.echo(f"Brier Score:      {m['brier']:.4f}")
    click.echo(f"Precision:        {m['precision']:.4f}")
    click.echo(f"Recall:           {m['recall']:.4f}")
    click.echo(f"F1 Score:         {m['f1']:.4f}")
    click.echo("-"*40)
    click.echo(f"TP/FP/TN/FN:      {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    click.echo(f"Uncertainty AUC:  {m['uncertainty_auc']:.4f}")
    click.echo(f"Avg Uncertainty:  {m['avg_uncertainty']:.4f}")
    click.echo("="*40 + "\n")

if __name__ == '__main__':
    cli()
