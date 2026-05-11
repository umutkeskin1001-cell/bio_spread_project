import polars as pl
from pathlib import Path
from typing import Tuple, Dict, Any

class DataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_genetic_map(self) -> pl.DataFrame:
        """
        Return DataFrame with columns: backbone_id, gene_list (list of strings).
        """
        backbones_path = Path(self.config['data']['backbones_path'])
        amr_path = Path(self.config['data']['amr_path'])

        if not backbones_path.exists() or not amr_path.exists():
            raise FileNotFoundError(f"Data files not found: {backbones_path} or {amr_path}")

        # Load backbones and AMR
        backbones = pl.scan_csv(str(backbones_path), separator='\t')
        amr = pl.scan_csv(str(amr_path), separator='\t')
        
        # Ensure AMR has sequence_accession
        if "NUCCORE_ACC" in amr.columns:
            amr = amr.rename({"NUCCORE_ACC": "sequence_accession"})
        
        # Group AMR genes by sequence_accession
        amr_genes = amr.group_by('sequence_accession').agg(pl.col('gene_symbol').alias('amr_genes'))
        
        # Join backbones and AMR
        genetic = backbones.join(amr_genes, on='sequence_accession', how='left')
        
        # Fill null AMR genes with empty lists
        # Also split replicon_types if it's a string
        genetic = genetic.with_columns([
            pl.col('amr_genes').fill_null([]),
            pl.col('replicon_types').str.split(',').fill_null([])
        ])

        # Concatenate replicon_types and amr_genes
        genetic = genetic.with_columns(
            pl.concat_list(
                pl.col('replicon_types'), 
                pl.col('amr_genes')
            ).alias('gene_list')
        ).select(['backbone_id', 'gene_list'])
        
        # Multiple records might have same backbone_id, keep unique signatures
        genetic = genetic.unique(subset=['backbone_id'])
        
        return genetic.collect()
    
    def prepare_dataset(self, records_path: str = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split by year: train < split_year, validation >= split_year.
        Joins with backbones to get the year if missing.
        """
        path = records_path or self.config['data']['records_path']
        backbones_path = self.config['data']['backbones_path']
        split_year = self.config['data']['split_year']
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Records file not found: {path}")
            
        # Load scored records
        scored = pl.read_csv(path, separator='\t')
        
        # If 'year' is missing, join with backbones to get 'resolved_year'
        if 'year' not in scored.columns:
            backbones = pl.read_csv(backbones_path, separator='\t', columns=['backbone_id', 'resolved_year'])
            # Keep only unique backbone years (latest resolution)
            backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
            scored = scored.join(backbones, on='backbone_id', how='left')
            scored = scored.rename({'resolved_year': 'year'})
            
        # Filter for rows with a label
        scored = scored.filter(pl.col('spread_label').is_not_null())
        
        # Cast spread_label to int
        scored = scored.with_columns(pl.col('spread_label').cast(pl.Int64))
            
        train = scored.filter(pl.col('year') < split_year)
        valid = scored.filter(pl.col('year') >= split_year)
        
        return train, valid
