import polars as pl
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Infiller")

def infill_labels(input_path: str, output_path: str):
    logger.info(f"Loading scored records from {input_path}")
    df = pl.read_csv(input_path, separator='\t')
    
    # Analyze the rule used for existing labels
    labeled = df.filter(pl.col('spread_label').is_not_null())
    total_unlabeled = df.filter(pl.col('spread_label').is_null()).shape[0]
    
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Currently labeled: {len(labeled)}")
    logger.info(f"Unlabeled: {total_unlabeled}")
    
    # We apply a strict rule based on our discovery:
    # If n_new_countries > 0, it's highly likely a spread event.
    # However, spread_label in the original was even more conservative.
    # We will use spread_severity_bin and n_new_countries to fill labels.
    
    # Logic: 
    # 1. If spread_label is missing AND (n_new_countries > 0 OR spread_severity_bin > 0) -> 1
    # 2. If spread_label is missing AND n_new_countries == 0 -> 0
    
    logger.info("Applying weak supervision rule to fill labels...")
    
    new_df = df.with_columns(
        pl.when(pl.col('spread_label').is_not_null())
        .then(pl.col('spread_label'))
        .when((pl.col('n_new_countries') > 0) | (pl.col('spread_severity_bin') > 0))
        .then(1.0)
        .otherwise(0.0)
        .alias('spread_label_filled')
    )
    
    # Check stats
    filled_stats = new_df['spread_label_filled'].value_counts()
    logger.info(f"Filled Label Distribution:\n{filled_stats}")
    
    # Replace original spread_label for training
    final_df = new_df.drop('spread_label').rename({'spread_label_filled': 'spread_label'})
    
    final_df.write_csv(output_path, separator='\t')
    logger.info(f"Saved infilled dataset to {output_path}")

if __name__ == "__main__":
    infill_labels(
        'data/project_inputs/scores/backbone_scored.tsv',
        'data/project_inputs/scores/backbone_scored_infilled.tsv'
    )
