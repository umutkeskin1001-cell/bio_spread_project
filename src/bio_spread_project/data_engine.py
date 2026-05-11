import polars as pl
from pathlib import Path
import logging

logger = logging.getLogger("DataEngine")

class DataEngine:
    """
    Sovereign Data Engine v1.0.
    Unified, lazy-loading, and high-performance data ingestion for the Oracle pipeline.
    Replaces legacy data.py and data_io.py.
    """
    def __init__(self, silver_dir: str = "data/project_inputs/silver"):
        self.silver_path = Path(silver_dir)
        self.backbone_path = self.silver_path / "plasmid_backbones.tsv"
        self.amr_path = self.silver_path / "plasmid_amr_hits.tsv"
        
    def scan_genetic_makeup(self) -> dict[str, set[str]]:
        """
        Scans silver databases to map backbone_ids to their complete genetic signatures (AMR + Replicons).
        """
        logger.info("Scanning genetic databases...")
        
        # Scan backbones for replicons
        df_backbones = pl.read_csv(self.backbone_path, separator="\t", null_values=[""])
        
        # Scan AMR hits
        df_amr = pl.read_csv(self.amr_path, separator="\t", null_values=[""])
        df_amr_grouped = df_amr.group_by("sequence_accession").agg(pl.col("gene_symbol").drop_nulls())
        
        # Join to link AMR genes to backbones
        df_joined = df_backbones.join(df_amr_grouped, on="sequence_accession", how="left")
        
        backbone_dict = {}
        for row in df_joined.iter_rows(named=True):
            bid = row["backbone_id"]
            if not bid: continue
            
            if bid not in backbone_dict:
                backbone_dict[bid] = set()
                
            # Add AMR genes
            genes = row.get("gene_symbol")
            if genes:
                for g in genes:
                    if g: backbone_dict[bid].add(f"AMR_{g}")
                    
            # Add Replicons
            reps = row.get("replicon_types")
            if reps and isinstance(reps, str):
                for r in reps.split(","):
                    if r.strip(): backbone_dict[bid].add(f"REP_{r.strip()}")
                    
        return backbone_dict

    def load_scored_dataset(self, file_path: str, backbone_dict: dict) -> list[dict]:
        """
        Loads a scored dataset (train or test) and attaches genetic features.
        """
        df = pl.read_csv(file_path, separator="\t")
        data = []
        for row in df.iter_rows(named=True):
            bid = row.get("backbone_id")
            if not bid: continue
            
            y = row.get("spread_label")
            if y is None: y = 0
            
            t = row.get("T_eff_norm")
            if t is None: t = 0.0
            
            genes = list(backbone_dict.get(bid, []))
            
            data.append({
                "backbone_id": bid,
                "genes": genes,
                "t": float(t),
                "y": int(y)
            })
        return data
