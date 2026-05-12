import click
import logging
import polars as pl
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial
from typing import Dict, List, Optional

from bio_spread_reborn.config.schema import Config
from bio_spread_reborn.utils.config import load_config
from bio_spread_reborn.data.loader import DataPipeline
from bio_spread_reborn.data.snapshot import TemporalSnapshotBuilder
from bio_spread_reborn.data.dataset import BioDataset, bio_collate_fn
from bio_spread_reborn.data.tokenizer import GeneTokenizer
from bio_spread_reborn.data.functional_tokenizer import FunctionalTokenizer
from bio_spread_reborn.models.evidential import FusionNetV3
from bio_spread_reborn.models.trainer import EvidentialTrainer
from bio_spread_reborn.utils.metrics import print_report
from bio_spread_reborn.features.pfp_extractor import PFPExtractor
from bio_spread_reborn.features.epistasis_matrix import EBVExtractor
from bio_spread_reborn.features.geo_aura import CAVExtractor
from bio_spread_reborn.data.data_fetcher import DataFetcher

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
@click.option('--config', default='config/default.yaml')
def piu_prepare(config):
    """Pillar Feature Extraction (PIU Phase 3)."""
    cfg = load_config(config)
    fetcher = DataFetcher()
    fetcher.fetch_geo_adjacency()
    fetcher.fetch_simulated_flights()
    
    # 1. Load Data
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    snapshot_path = Path(cfg.data.snapshot_cache_dir) / "snapshot_records.tsv"
    df = pl.read_csv(snapshot_path, separator='\t')
    backbone_meta = pl.read_csv(cfg.data.backbones_path, separator='\t')
    
    # 2. Pillar 1: PFP
    pfp_ext = PFPExtractor()
    # Mocking sequences for demonstration if faa files are missing
    bid_to_seqs = {bid: [("p1", "MKKVLLLSVLLV")] for bid in df["backbone_id"].unique()}
    pfp_feats = pfp_ext.compute_pfp(bid_to_seqs, backbone_meta)
    
    # 3. Pillar 2: EBV (Leakage-free: only use backbones seen before split_year)
    raw_records = pl.read_csv(cfg.data.records_path, separator='\t')
    if 'resolved_year' in raw_records.columns and 'year' not in raw_records.columns:
        raw_records = raw_records.rename({'resolved_year': 'year'})
    
    past_backbones = set(raw_records.filter(pl.col("year") < cfg.data.split_year)["backbone_id"].unique().to_list())
    past_genetic_map = {bid: genes for bid, genes in genetic_map.items() if bid in past_backbones}
    
    ebv_ext = EBVExtractor()
    ebv_ext.build_matrix(past_genetic_map)
    ebv_feats = ebv_ext.compute_all_ebv(genetic_map) # Can compute for all, but matrix is built on past
    
    # 4. Pillar 3: CAV (Leakage-free: dynamic per snapshot)
    cav_ext = CAVExtractor()
    G = cav_ext.build_world_graph()
    cav_ext.train_node2vec(G)
    
    # 5. Merge and Save (Snapshot aware: bid_year)
    extra_features = {}
    
    # Pre-group raw_records by backbone for fast historical lookups
    bid_records = raw_records.select(["backbone_id", "year", "country"]).sort("year")
    
    for row in df.to_dicts():
        bid = row["backbone_id"]
        year = row["year"]
        key = f"{bid}_{year}"
        
        # PFP (Static)
        pfp = pfp_feats.get(bid, np.zeros(16))
        # EBV (Static, but matrix built without leakage)
        ebv = ebv_feats.get(bid, np.zeros(6))
        
        # CAV (Dynamic: only countries seen <= year)
        past_countries = bid_records.filter(
            (pl.col("backbone_id") == bid) & (pl.col("year") <= year)
        )["country"].unique().to_list()
        
        cav = cav_ext.compute_aura(past_countries)
        
        extra_features[key] = np.concatenate([pfp, ebv, cav]).tolist()
    
    with open("data/piu_extra_features.json", "w") as f:
        json.dump(extra_features, f)
    
    logger.info("PIU feature extraction complete. Saved to data/piu_extra_features.json")


@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
@click.option('--resume-from', default=None, help='Path to best_model.pt to resume training from')
def train(config, resume_from):
    """Train the FusionNetV3 model with PIU features."""
    cfg = load_config(config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load Data
    snapshot_path = Path(cfg.data.snapshot_cache_dir) / "snapshot_records.tsv"
    df = pl.read_csv(snapshot_path, separator='\t')
    
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    # Load Extra Features
    extra_path = Path("data/piu_extra_features.json")
    extra_features = None
    if extra_path.exists():
        with open(extra_path, "r") as f:
            extra_features = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
            logger.info(f"Loaded extra features for {len(extra_features)} backbones")

    train_df = df.filter(pl.col("year") < cfg.data.split_year)
    val_df = df.filter(pl.col("year") >= cfg.data.split_year)
    
    # 3. Tokenizers
    tokenizer = GeneTokenizer(max_len=cfg.model.max_genes)
    tokenizer.fit(genetic_map_df["gene_list"])
    func_tokenizer = FunctionalTokenizer()
    
    # 4. Create Loaders
    train_ds = BioDataset(train_df, genetic_map, extra_features=extra_features, max_genes=cfg.model.max_genes, return_pairs=True)
    val_ds = BioDataset(val_df, genetic_map, extra_features=extra_features, max_genes=cfg.model.max_genes, return_pairs=False)
    
    labels = train_df["spread_label"].to_list()
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0/max(1, class_counts[l]) for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, collate_fn=collate)
    
    # 5. Initialize FusionNetV3
    model = FusionNetV3(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(train_ds.history_cols),
        extra_dim=30,
        emb_dim=cfg.model.emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=1,
        time_freqs=cfg.model.time_freqs
    )
    
    # 6. Initialize Trainer
    train_cfg = cfg.model_dump()
    train_cfg["run_id"] = run_id
    trainer = EvidentialTrainer(model, train_cfg)
    
    if resume_from:
        logger.info(f"Resuming model weights from {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location="cpu"))
    
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
    """Run inference using FusionNetV3."""
    cfg = load_config(config)
    tokenizer = GeneTokenizer.load(Path(tokenizer_path))
    func_tokenizer = FunctionalTokenizer()
    
    input_df = pl.read_csv(input_path, separator='\t')
    
    # Add dummy extra features for prediction if not provided
    extra_features = {}
    for bid in input_df["backbone_id"].unique():
        extra_features[bid] = np.zeros(30, dtype=np.float32)
        
    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    ds = BioDataset(input_df, genetic_map, extra_features=extra_features, max_genes=cfg.model.max_genes)
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate)
    
    model = FusionNetV3(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(ds.history_cols),
        extra_dim=30,
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
            x_gene, x_func, x_history, x_extra, t, _ = batch
            prob, unc, _ = model(x_gene, x_func, x_history, x_extra, t)
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
    """Evaluate FusionNetV3 performance."""
    cfg = load_config(config)
    tokenizer = GeneTokenizer.load(Path(tokenizer_path))
    func_tokenizer = FunctionalTokenizer()
    
    df = pl.read_csv(input_path, separator='\t')
    
    extra_path = Path("data/piu_extra_features.json")
    extra_features = None
    if extra_path.exists():
        with open(extra_path, "r") as f:
            extra_features = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

    pipeline = DataPipeline(cfg)
    genetic_map_df = pipeline.load_genetic_map()
    genetic_map = dict(zip(genetic_map_df["backbone_id"], genetic_map_df["gene_list"]))
    
    ds = BioDataset(df, genetic_map, extra_features=extra_features, max_genes=cfg.model.max_genes)
    collate = partial(bio_collate_fn, tokenizer=tokenizer, max_genes=cfg.model.max_genes)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, collate_fn=collate)
    
    model = FusionNetV3(
        vocab_size=tokenizer.get_vocab_size(),
        func_dim=func_tokenizer.get_dim(),
        history_dim=len(ds.history_cols),
        extra_dim=30,
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
