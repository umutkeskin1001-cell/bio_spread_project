import polars as pl
from pathlib import Path
from typing import Tuple, Optional
from bio_spread_reborn.config.schema import Config

class DataPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._genetic_map_df: Optional[pl.DataFrame] = None

    def load_genetic_map(self) -> pl.DataFrame:
        """
        Return DataFrame with columns: backbone_id, gene_list (list of strings).
        Caches the result to avoid redundant I/O.
        """
        if self._genetic_map_df is not None:
            return self._genetic_map_df

        backbones_path = Path(self.config.data.backbones_path)
        amr_path = Path(self.config.data.amr_path)

        if not backbones_path.exists() or not amr_path.exists():
            raise FileNotFoundError(f"Data files not found: {backbones_path} or {amr_path}")

        # Use lazy loading for efficiency
        backbones = pl.scan_csv(str(backbones_path), separator='\t')
        amr = pl.scan_csv(str(amr_path), separator='\t')
        
        if "NUCCORE_ACC" in amr.columns:
            amr = amr.rename({"NUCCORE_ACC": "sequence_accession"})
        
        amr_genes = amr.group_by('sequence_accession').agg(pl.col('gene_symbol').alias('amr_genes'))
        
        genetic = backbones.join(amr_genes, on='sequence_accession', how='left')
        
        genetic = genetic.with_columns([
            pl.col('amr_genes').fill_null([]),
            pl.col('replicon_types').str.split(',').fill_null([])
        ])

        genetic = genetic.with_columns(
            pl.concat_list(
                pl.col('replicon_types'), 
                pl.col('amr_genes')
            ).alias('gene_list')
        ).select(['backbone_id', 'gene_list'])
        
        self._genetic_map_df = genetic.unique(subset=['backbone_id']).collect()
        return self._genetic_map_df
    
    def prepare_dataset(self, records_path: Optional[str] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split by year: train < split_year, validation >= split_year.
        """
        path = records_path or self.config.data.records_path
        split_year = self.config.data.split_year
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Records file not found: {path}")
            
        scored = pl.read_csv(path, separator='\t')
        
        if 'year' not in scored.columns:
            # Re-use cached genetic map if possible, but we need resolved_year which is NOT in genetic_map
            # So we load only necessary columns from backbones
            backbones = pl.read_csv(self.config.data.backbones_path, separator='\t', columns=['backbone_id', 'resolved_year'])
            backbones = backbones.group_by('backbone_id').agg(pl.col('resolved_year').max())
            scored = scored.join(backbones, on='backbone_id', how='left')
            scored = scored.rename({'resolved_year': 'year'})
            
        # Ensure label exists and is correct type
        scored = scored.filter(pl.col('spread_label').is_not_null())
        scored = scored.with_columns(pl.col('spread_label').cast(pl.Int64))
            
        train = scored.filter(pl.col('year') < split_year)
        valid = scored.filter(pl.col('year') >= split_year)
        
        return train, valid
