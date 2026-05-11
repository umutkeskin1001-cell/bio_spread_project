import polars as pl
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TemporalSnapshotBuilder:
    def __init__(self, raw_records: pl.DataFrame, backbone_meta: pl.DataFrame, amr: pl.DataFrame):
        """
        raw_records must contain backbone_id, year, country, host_genus, etc.
        """
        self.raw = raw_records
        self.backbone_meta = backbone_meta
        self.amr = amr
        
        # Ensure year column is sorted
        self.raw = self.raw.sort("year")

    def get_label(self, backbone_id: str, cutoff_year: int) -> int:
        """
        Return 1 if this backbone spreads to a NEW country after cutoff_year, 0 otherwise.
        """
        past_countries = set(
            self.raw.filter(
                (pl.col("backbone_id") == backbone_id) & 
                (pl.col("year") <= cutoff_year)
            )["country"].unique().to_list()
        )
        
        if not past_countries:
            return 0
            
        future_countries = set(
            self.raw.filter(
                (pl.col("backbone_id") == backbone_id) & 
                (pl.col("year") > cutoff_year)
            )["country"].unique().to_list()
        )
        
        has_spread = any(c not in past_countries for c in future_countries)
        return int(has_spread)

    def get_backcast_features(self, backbone_id: str, cutoff_year: int) -> Dict[str, float]:
        """
        Compute epidemiological features using ONLY data <= cutoff_year.
        """
        history = self.raw.filter(
            (pl.col("backbone_id") == backbone_id) & (pl.col("year") <= cutoff_year)
        )
        
        if history.is_empty():
            return {
                "n_countries_so_far": 0.0,
                "n_host_genera_so_far": 0.0,
                "years_since_first_obs": 0.0,
                "delta_countries_last_2y": 0.0,
                "n_records_so_far": 0.0
            }
            
        first_year = history["year"].min()
        n_countries = history["country"].n_unique()
        n_hosts = history["genus"].n_unique()
        n_records = len(history)
        
        # Velocity: new countries in the last 2 years of the window
        last_2y = history.filter(pl.col("year") >= (cutoff_year - 2))
        older = history.filter(pl.col("year") < (cutoff_year - 2))
        
        countries_in_last_2y = set(last_2y["country"].unique().to_list())
        countries_before = set(older["country"].unique().to_list())
        new_recent = len([c for c in countries_in_last_2y if c not in countries_before])
        
        return {
            "n_countries_so_far": float(n_countries),
            "n_host_genera_so_far": float(n_hosts),
            "years_since_first_obs": float(cutoff_year - first_year),
            "delta_countries_last_2y": float(new_recent),
            "n_records_so_far": float(n_records)
        }

    def build_snapshot(self, output_path: Path):
        """
        Build a feature-rich snapshot dataset.
        """
        logger.info("Building enriched temporal snapshots...")
        
        # We iterate over unique observations
        observations = self.raw.select(["backbone_id", "year"]).unique().sort(["year", "backbone_id"])
        
        results = []
        for row in observations.to_dicts():
            bid = row["backbone_id"]
            year = row["year"]
            
            label = self.get_label(bid, year)
            features = self.get_backcast_features(bid, year)
            
            # Combine
            record = {
                "backbone_id": bid,
                "year": year,
                "spread_label": label,
                **features
            }
            results.append(record)
            
        snapshot_df = pl.DataFrame(results)
        
        # Add static traits from backbone_meta if possible
        meta_cols = ["backbone_id", "size", "gc", "n_replicon_types", "n_relaxase_types"]
        meta_cols = [c for c in meta_cols if c in self.backbone_meta.columns]
        
        if len(meta_cols) > 1:
            meta_subset = self.backbone_meta.select(meta_cols).unique(subset=["backbone_id"])
            snapshot_df = snapshot_df.join(meta_subset, on="backbone_id", how="left")
            
            # Fill nulls
            for col in meta_cols:
                if col != "backbone_id":
                    snapshot_df = snapshot_df.with_columns(pl.col(col).fill_null(0))

        snapshot_df.write_csv(output_path, separator="\t")
        logger.info(f"Enriched snapshot saved to {output_path}")
        return snapshot_df
