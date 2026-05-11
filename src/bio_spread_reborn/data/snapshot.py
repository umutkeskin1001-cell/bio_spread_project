import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TemporalSnapshotBuilder:
    def __init__(self, raw_records: pl.DataFrame, backbone_meta: pl.DataFrame, amr: pl.DataFrame):
        """
        raw_records must contain backbone_id, year, country, etc.
        backbone_meta must contain backbone_id and gene information.
        amr must contain backbone_id and amr hits.
        """
        self.raw = raw_records
        self.backbone_meta = backbone_meta
        self.amr = amr
        
        # Pre-filter for performance
        self.backbone_ids = set(self.raw["backbone_id"].unique().to_list())

    def get_label(self, backbone_id: str, cutoff_year: int) -> int:
        """
        Return 1 if this backbone spreads to a NEW country after cutoff_year, 0 otherwise.
        A 'spread' is defined as appearing in a country where it wasn't seen before cutoff_year.
        """
        # Records of this backbone BEFORE or AT cutoff
        past_countries = set(
            self.raw.filter(
                (pl.col("backbone_id") == backbone_id) & 
                (pl.col("year") <= cutoff_year)
            )["country"].unique().to_list()
        )
        
        if not past_countries:
            return 0  # Should not happen if we are building snapshot for a seen record
            
        # Records of this backbone AFTER cutoff
        future_countries = set(
            self.raw.filter(
                (pl.col("backbone_id") == backbone_id) & 
                (pl.col("year") > cutoff_year)
            )["country"].unique().to_list()
        )
        
        # Spread = any future country not in past countries
        has_spread = any(c not in past_countries for c in future_countries)
        return int(has_spread)

    def build_snapshot(self, output_path: Path):
        """
        Build a dataset where each row is an observation at year T, 
        with label computed based on T+1...Infinity.
        """
        logger.info("Building temporal snapshots for all records...")
        
        results = []
        # We iterate over every unique (backbone_id, year) pair in raw records
        observations = self.raw.select(["backbone_id", "year"]).unique().sort("year")
        
        for row in observations.to_dicts():
            bid = row["backbone_id"]
            year = row["year"]
            
            label = self.get_label(bid, year)
            
            results.append({
                "backbone_id": bid,
                "year": year,
                "spread_label": label
            })
            
        snapshot_df = pl.DataFrame(results)
        
        # Join with backbone metadata and AMR (only if needed for features)
        # For now, we just need the labels and years
        snapshot_df.write_csv(output_path, separator="\t")
        logger.info(f"Snapshot dataset saved to {output_path} with {len(snapshot_df)} records.")
        return snapshot_df
