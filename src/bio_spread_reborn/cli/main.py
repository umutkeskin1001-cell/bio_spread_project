import click
import logging
import polars as pl
import torch
import json
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial

from bio_spread_reborn.config.schema import Config
from bio_spread_reborn.utils.config import load_config
from bio_spread_reborn.data.loader import DataPipeline
from bio_spread_reborn.data.snapshot import TemporalSnapshotBuilder
from bio_spread_reborn.data.dataset import BioDataset, bio_collate_fn
from bio_spread_reborn.data.tokenizer import GeneTokenizer
from bio_spread_reborn.data.functional_tokenizer import FunctionalTokenizer
from bio_spread_reborn.models.evidential import FusionNet
from bio_spread_reborn.models.trainer import EvidentialTrainer
from bio_spread_reborn.utils.metrics import print_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioSpreadCLI")

@click.group()
def cli():
    """BioSpread Reborn: Elite-grade plasmid spread surveillance."""
    pass

@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
@click.option('--output-path', default='data/snapshots/snapshot_records.tsv')
def snapshot(config, output_path):
    """Build temporal snapshots with backcast features."""
    cfg = load_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading raw data for snapshot building...")
    raw_records = pl.read_csv(cfg.data.records_path, separator='\t')
    if 'resolved_year' in raw_records.columns and 'year' not in raw_records.columns:
        raw_records = raw_records.rename({'resolved_year': 'year'})
        
    backbone_meta = pl.read_csv(cfg.data.backbones_path, separator='\t')
    amr = pl.read_csv(cfg.data.amr_path, separator='\t')
    
    builder = TemporalSnapshotBuilder(raw_records, backbone_meta, amr)
    builder.build_snapshot(output_path)
    click.echo(f"Enriched snapshot created at {output_path}")

@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
def train(config):
    """Train the FusionNet model with backcast features and functional encoding."""
    cfg = load_config(config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load Enriched Snapshot
    snapshot_path = Path(cfg.data.snapshot_cache_dir) / "snapshot_records.tsv"
    if not snapshot_path.exists():
        raise FileNotFoundError("Snapshot missing. Run 'bio-spread snapshot' first.")
    
    df = pl.read_csv(snapshot_path, separator='\t')
    
    # 2. Data Pipeline
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    train_df = df.filter(pl.col("year") < cfg.data.split_year)
    val_df = df.filter(pl.col("year") >= cfg.data.split_year)
    
    # 3. Tokenizers
    tokenizer = GeneTokenizer(max_len=cfg.model.max_genes)
    tokenizer.fit(genetic_map_df["gene_list"])
    func_tokenizer = FunctionalTokenizer()
    
    # 4. Create Loaders
    train_ds = BioDataset(train_df, genetic_map, max_genes=cfg.model.max_genes)
    val_ds = BioDataset(val_df, genetic_map, max_genes=cfg.model.max_genes)
    
    # Balanced Sampler
    labels = train_df["spread_label"].to_list()
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0/class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, collate_fn=collate)
    
    # 5. Initialize FusionNet
    model = FusionNet(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(train_ds.history_cols),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=1, # Light Transformer for Phase 2
        time_freqs=cfg.model.time_freqs
    )
    
    # 6. Initialize Trainer
    train_cfg = cfg.model_dump()
    train_cfg["run_id"] = run_id
    trainer = EvidentialTrainer(model, train_cfg)
    
    # Save artifacts before fit
    tokenizer.save(trainer.artifact_dir / 'tokenizer.json')
    
    # 7. Fit
    artifact_dir = trainer.fit(train_loader, val_loader)
    logger.info(f"Training complete. Artifacts saved in {artifact_dir}")

@cli.command()
@click.option('--model-path', required=True)
@click.option('--tokenizer-path', required=True)
@click.option('--input-path', required=True)
@click.option('--output-path', default='predictions.json')
@click.option('--config', default='config/default.yaml')
def predict(model_path, tokenizer_path, input_path, output_path, config):
    """Run inference using FusionNet."""
    cfg = load_config(config)
    tokenizer = GeneTokenizer.load(Path(tokenizer_path))
    func_tokenizer = FunctionalTokenizer()
    
    # Build a mini-dataset for prediction to get the correct dimensions
    input_df = pl.read_csv(input_path, separator='\t')
    
    # If columns are missing, add defaults to avoid crash
    required = ["n_countries_so_far", "n_host_genera_so_far", "years_since_first_obs", "delta_countries_last_2y", "n_records_so_far"]
    for col in required:
        if col not in input_df.columns:
            input_df = input_df.with_columns(pl.lit(0.0).alias(col))
    
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    ds = BioDataset(input_df, genetic_map, max_genes=cfg.model.max_genes)
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate)
    
    model = FusionNet(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(ds.history_cols),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=1,
        time_freqs=cfg.model.time_freqs
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_gene, x_func, x_history, t, _ = batch
            prob, unc, _ = model(x_gene, x_func, x_history, t)
            results.append({
                "backbone_id": input_df["backbone_id"][i],
                "spread_probability": float(prob.item()),
                "uncertainty": float(unc.item())
            })
            
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"Predictions saved to {output_path}")

@cli.command()
@click.option('--model-path', required=True)
@click.option('--tokenizer-path', required=True)
@click.option('--input-path', required=True)
@click.option('--config', default='config/default.yaml')
def evaluate(model_path, tokenizer_path, input_path, config):
    """Evaluate FusionNet performance."""
    cfg = load_config(config)
    tokenizer = GeneTokenizer.load(Path(tokenizer_path))
    func_tokenizer = FunctionalTokenizer()
    
    df = pl.read_csv(input_path, separator='\t')
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    ds = BioDataset(df, genetic_map, max_genes=cfg.model.max_genes)
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=collate)
    
    model = FusionNet(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(ds.history_cols),
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=1,
        time_freqs=cfg.model.time_freqs
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    trainer = EvidentialTrainer(model, cfg.model_dump())
    metrics = trainer.evaluate(loader)
    print_report(metrics)

if __name__ == "__main__":
    cli()
