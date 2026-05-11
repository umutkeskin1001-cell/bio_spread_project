import click
import torch
import logging
import json
from pathlib import Path
from torch.utils.data import DataLoader

from bio_spread_reborn.data.loader import DataPipeline
from bio_spread_reborn.data.tokenizer import GeneTokenizer
from bio_spread_reborn.data.dataset import BioDataset
from bio_spread_reborn.models.evidential import SovereignNet
from bio_spread_reborn.models.trainer import EvidentialTrainer
from bio_spread_reborn.utils.config import load_config
from bio_spread_reborn.utils.metrics import compute_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """BioSpread Reborn: Elite-grade plasmid spread surveillance."""
    pass

@cli.command()
@click.option('--config', default='config/default.yaml', help='Path to config file')
def train(config):
    """Train the SovereignNet model using the evidential learning pipeline."""
    cfg = load_config(config)
    logger.info("Initializing Data Pipeline...")
    
    pipeline = DataPipeline(cfg)
    
    # 1. Load genetic map and fit tokenizer
    genetic_df = pipeline.load_genetic_map()
    tokenizer = GeneTokenizer(max_len=cfg['model']['max_genes'])
    tokenizer.fit(genetic_df['gene_list'])
    
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # 2. Encode genetic map
    # We store the encoded lists in a dict for the dataset
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    # 3. Prepare train/val splits
    train_df, val_df = pipeline.prepare_dataset()
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # 4. Create Datasets and Loaders
    train_ds = BioDataset(train_df, genetic_map)
    val_ds = BioDataset(val_df, genetic_map)
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'])
    
    # 5. Initialize Model and Trainer
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg['model']['emb_dim'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    
    trainer = EvidentialTrainer(model, cfg)
    
    # 6. Fit
    trainer.fit(train_loader, val_loader)
    
    # 7. Save artifacts
    torch.save(tokenizer.vocab, 'vocab.pt')
    logger.info("Training complete. Model and vocab saved.")

@cli.command()
@click.option('--model-path', default='best_model.pt', help='Path to model weights')
@click.option('--vocab-path', default='vocab.pt', help='Path to vocabulary')
@click.option('--input-path', required=True, help='Path to TSV with backbone records')
@click.option('--output-path', default='predictions.json', help='Path to save results')
@click.option('--config', default='config/default.yaml', help='Path to config file')
def predict(model_path, vocab_path, input_path, output_path, config):
    """Run inference on new backbone records."""
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Vocab and Tokenizer
    vocab = torch.load(vocab_path)
    tokenizer = GeneTokenizer(max_len=cfg['model']['max_genes'])
    tokenizer.vocab = vocab
    
    # 2. Load Model
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg['model']['emb_dim'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Load Data
    pipeline = DataPipeline(cfg)
    genetic_df = pipeline.load_genetic_map()
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    import polars as pl
    input_df = pl.read_csv(input_path, separator='\t')
    dataset = BioDataset(input_df, genetic_map)
    loader = DataLoader(dataset, batch_size=32)
    
    # 4. Predict
    results = []
    with torch.no_grad():
        for i, (x, t, *y) in enumerate(loader):
            x, t = x.to(device), t.to(device)
            prob, unc, alpha = model(x, t)
            
            # Map back to IDs
            start_idx = i * 32
            end_idx = start_idx + len(x)
            batch_ids = input_df['backbone_id'][start_idx:end_idx].to_list()
            
            for bid, p, u in zip(batch_ids, prob.cpu().numpy(), unc.cpu().numpy()):
                results.append({
                    'backbone_id': bid,
                    'spread_probability': float(p),
                    'uncertainty': float(u)
                })
                
@cli.command()
@click.option('--model-path', default='best_model.pt', help='Path to model weights')
@click.option('--vocab-path', default='vocab.pt', help='Path to vocabulary')
@click.option('--input-path', required=True, help='Path to TSV with backbone records')
@click.option('--config', default='config/default.yaml', help='Path to config file')
def evaluate(model_path, vocab_path, input_path, config):
    """Evaluate model performance on a labeled dataset."""
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Vocab and Tokenizer
    vocab = torch.load(vocab_path)
    tokenizer = GeneTokenizer(max_len=cfg['model']['max_genes'])
    tokenizer.vocab = vocab
    
    # 2. Load Model
    model = SovereignNet(
        vocab_size=tokenizer.get_vocab_size(),
        emb_dim=cfg['model']['emb_dim'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Load Data
    pipeline = DataPipeline(cfg)
    genetic_df = pipeline.load_genetic_map()
    genetic_map = {
        row['backbone_id']: tokenizer.encode(row['gene_list']) 
        for row in genetic_df.to_dicts()
    }
    
    import polars as pl
    input_df = pl.read_csv(input_path, separator='\t')
    
    # Ensure 'year' exists (similar to prepare_dataset)
    if 'year' not in input_df.columns:
        backbones = pl.read_csv(cfg['data']['backbones_path'], separator='\t', columns=['backbone_id', 'resolved_year'])
        backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
        input_df = input_df.join(backbones, on='backbone_id', how='left')
        input_df = input_df.rename({'resolved_year': 'year'})
    
    # Filter for rows with a label
    input_df = input_df.filter(pl.col('spread_label').is_not_null())
    
    dataset = BioDataset(input_df, genetic_map)
    loader = DataLoader(dataset, batch_size=32)
    
    # 4. Predict
    all_probs = []
    all_uncs = []
    all_targets = []
    
    with torch.no_grad():
        for x, t, y in loader:
            x, t = x.to(device), t.to(device)
            prob, unc, alpha = model(x, t)
            all_probs.extend(prob.cpu().numpy())
            all_uncs.extend(unc.cpu().numpy())
            all_targets.extend(y.numpy())
                
    # 5. Compute Metrics
    import numpy as np
    m = compute_metrics(np.array(all_targets), np.array(all_probs), np.array(all_uncs))
    
    # 6. Print Report
    click.echo("\n" + "="*40)
    click.echo("       BIO-SPREAD EVALUATION REPORT")
    click.echo("="*40)
    click.echo(f"ROC AUC:          {m['auc']:.4f}")
    click.echo(f"Brier Score:      {m['brier']:.4f}")
    click.echo(f"Precision:        {m['precision']:.4f}")
    click.echo(f"Recall:           {m['recall']:.4f}")
    click.echo(f"F1 Score:         {m['f1']:.4f}")
    click.echo("-"*40)
    click.echo(f"True Positives:   {m['tp']}")
    click.echo(f"False Positives:  {m['fp']}")
    click.echo(f"True Negatives:   {m['tn']}")
    click.echo(f"False Negatives:  {m['fn']}")
    click.echo("-"*40)
    click.echo(f"Uncertainty AUC:  {m['uncertainty_auc']:.4f}")
    click.echo(f"Avg Uncertainty:  {m['avg_uncertainty']:.4f}")
    click.echo("="*40 + "\n")

if __name__ == '__main__':
    cli()
