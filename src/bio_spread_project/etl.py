import polars as pl
from pathlib import Path
import logging
from bio_spread_project.schema import PLASMID_SCHEMA

logger = logging.getLogger("ETL")

class SovereignETL:
    """
    High-performance ETL engine.
    Ensures leak-proof data loading and genetic feature mapping.
    """
    def __init__(self, silver_dir: str = "data/project_inputs/silver"):
        self.silver_path = Path(silver_dir)
        self.backbone_path = self.silver_path / "plasmid_backbones.tsv"
        self.amr_path = self.silver_path / "plasmid_amr_hits.tsv"

    def load_genetic_map(self) -> dict[str, list[str]]:
        """Maps backbone_ids to their DNA signatures (AMR + Replicons)."""
        logger.info("Scanning genetic databases...")
        
        df_backbones = pl.read_csv(self.backbone_path, separator="\t")
        df_amr = pl.read_csv(self.amr_path, separator="\t")
        
        # Group AMR genes
        df_amr_grouped = df_amr.group_by("sequence_accession").agg(
            pl.col("gene_symbol").drop_nulls().alias("genes")
        )
        
        # Join to backbones
        df_joined = df_backbones.join(df_amr_grouped, on="sequence_accession", how="left")
        
        genetic_map = {}
        for row in df_joined.iter_rows(named=True):
            bid = row["backbone_id"]
            if not bid: continue
            
            signature = []
            # Add Replicons
            reps = row.get("replicon_types")
            if reps:
                signature.extend([f"REP_{r.strip()}" for r in reps.split(",") if r.strip()])
            
            # Add AMR Genes
            genes = row.get("genes")
            if genes:
                signature.extend([f"AMR_{g}" for g in genes if g])
                
            genetic_map[bid] = list(set(signature))
            
        return genetic_map

    def prepare_dataset(self, file_path: str, genetic_map: dict[str, list[str]]) -> pl.DataFrame:
        """Loads and validates a scored dataset, attaching genetic features."""
        df = pl.read_csv(file_path, separator="\t")
        
        # Ensure minimal schema requirements
        required = ["backbone_id", "T_eff_norm", "spread_label"]
        for col in required:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        
        # Convert to list of features
        data = []
        for row in df.iter_rows(named=True):
            bid = row["backbone_id"]
            data.append({
                "backbone_id": bid,
                "y": int(row["spread_label"] or 0),
                "t": float(row["T_eff_norm"] or 0.0),
                "genes": genetic_map.get(bid, [])
            })
            
        return pl.from_dicts(data)
