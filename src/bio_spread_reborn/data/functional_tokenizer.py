import json
from pathlib import Path
from typing import List, Dict

class FunctionalTokenizer:
    """
    Maps genes to functional biological categories instead of raw symbols.
    Categories: AMR, Mobility, Maintenance, Phage, Other.
    """
    CATEGORIES = ["AMR", "MOBILITY", "MAINTENANCE", "PHAGE", "OTHER"]
    
    def __init__(self):
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.CATEGORIES)}
        # A small curated map of common gene prefixes/keywords to categories
        self.keywords = {
            "bla": "AMR", "mcr": "AMR", "tet": "AMR", "aac": "AMR", "aph": "AMR", "ant": "AMR",
            "erm": "AMR", "sul": "AMR", "dfr": "AMR", "qnr": "AMR", "cat": "AMR", "flo": "AMR",
            "tra": "MOBILITY", "trb": "MOBILITY", "trw": "MOBILITY", "vir": "MOBILITY", "mob": "MOBILITY",
            "par": "MAINTENANCE", "rep": "MAINTENANCE", "ccdB": "MAINTENANCE", "ccdA": "MAINTENANCE",
            "hok": "MAINTENANCE", "sok": "MAINTENANCE", "vag": "MAINTENANCE", "pem": "MAINTENANCE",
            "int": "PHAGE", "xer": "PHAGE", "hp": "OTHER"
        }

    def map_gene(self, gene: str) -> str:
        gene = gene.lower()
        for kw, cat in self.keywords.items():
            if kw in gene:
                return cat
        return "OTHER"

    def encode(self, gene_list: List[str]) -> List[float]:
        """
        Returns a multi-hot vector (bag-of-categories).
        """
        vector = [0.0] * len(self.CATEGORIES)
        if gene_list is None or len(gene_list) == 0:
            return vector
            
        for gene in gene_list:
            cat = self.map_gene(gene)
            idx = self.cat_to_idx[cat]
            vector[idx] += 1.0 # Count-based or multi-hot
            
        # Optional: Normalize
        total = sum(vector)
        if total > 0:
            vector = [v / total for v in vector]
            
        return vector

    def get_dim(self) -> int:
        return len(self.CATEGORIES)
