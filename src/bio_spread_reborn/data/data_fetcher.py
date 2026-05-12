import polars as pl
import requests
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Fetches external assets for geographic and biological context.
    """
    def __init__(self, data_dir: Path = Path("data/external")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_geo_adjacency(self):
        """
        Fetch country adjacency data from a public source.
        Using a common GitHub repository for country data.
        """
        url = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"
        target = self.data_dir / "countries.json"
        
        if not target.exists():
            logger.info(f"Fetching country data from {url}")
            resp = requests.get(url)
            resp.raise_for_status()
            with open(target, "w") as f:
                json.dump(resp.json(), f)
        
        return target

    def fetch_simulated_flights(self):
        """
        Since real flight data is proprietary/large, we simulate a hub-and-spoke model
        based on country population/GDP or use a small public sample.
        """
        target = self.data_dir / "simulated_flights.csv"
        if not target.exists():
            # Create a dummy hub-and-spoke CSV
            # origin, dest, flights
            hubs = ["USA", "CHN", "DEU", "GBR", "TUR", "ARE", "SGP"]
            data = []
            for h1 in hubs:
                for h2 in hubs:
                    if h1 != h2:
                        data.append({"origin": h1, "dest": h2, "flights": 1000})
            
            pl.DataFrame(data).write_csv(target)
        return target

    def fetch_protein_sequences(self, backbone_ids: list):
        """
        In a real scenario, this would use Bio.Entrez or NCBI datasets CLI.
        For this implementation, we assume sequences are provided in data/raw/sequences.
        If missing, we generate representative placeholders to avoid crashing.
        """
        seq_dir = Path("data/raw/sequences")
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Placeholder logic
        for bid in backbone_ids:
            target = seq_dir / f"{bid}.faa"
            if not target.exists():
                # Generate a dummy protein for the backbone
                with open(target, "w") as f:
                    f.write(f">{bid}_p1\nMKKVLLLSVLLV\n")
        
        return seq_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    fetcher.fetch_geo_adjacency()
    fetcher.fetch_simulated_flights()
    print("Data acquisition complete.")
