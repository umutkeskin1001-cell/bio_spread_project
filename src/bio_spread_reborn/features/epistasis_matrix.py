import numpy as np
from scipy.stats import fisher_exact
from collections import defaultdict
from itertools import combinations
import polars as pl
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class EBVExtractor:
    """
    Pillar 2: Epistatic Pressure Vector (EBV).
    Extracts gene interaction synergy/antagonism based on Fisher's Exact Test.
    """
    def __init__(self, min_count: int = 5, p_threshold: float = 0.01, log_or_threshold: float = 1.5):
        self.min_count = min_count
        self.p_threshold = p_threshold
        self.log_or_threshold = log_or_threshold
        self.epistasis_matrix = {} # (gene1, gene2) -> odds_ratio

    def build_matrix(self, gene_lists: Dict[str, List[str]]):
        """
        Build the epistasis matrix from a population of plasmid gene lists.
        """
        logger.info(f"Building epistasis matrix from {len(gene_lists)} plasmids...")
        
        # 1. Gene frequencies
        gene_freq = defaultdict(int)
        total_plasmids = len(gene_lists)
        for genes in gene_lists.values():
            for g in set(genes):
                gene_freq[g] += 1
        
        # 2. Co-occurrence counts
        pair_counts = defaultdict(int)
        for genes in gene_lists.values():
            unique_genes = sorted(list(set(genes)))
            for g1, g2 in combinations(unique_genes, 2):
                pair_counts[(g1, g2)] += 1
        
        # 3. Fisher Exact Test
        epistasis = {}
        for (g1, g2), co_count in pair_counts.items():
            if co_count < self.min_count:
                continue
            
            # 2x2 contingency table
            #         g2+   g2-
            # g1+ [ [a,    b],
            # g1-   [c,    d] ]
            a = co_count
            b = gene_freq[g1] - a
            c = gene_freq[g2] - a
            d = total_plasmids - (a + b + c)
            
            if d < 0: continue # Should not happen with correct logic
            
            table = [[a, b], [c, d]]
            odds_ratio, p_value = fisher_exact(table)
            
            # Cap OR to avoid Infinity
            odds_ratio = min(odds_ratio, 1000.0)
            
            # Apply thresholds
            if p_value < self.p_threshold:
                log_or = np.log2(odds_ratio + 1e-10)
                if abs(log_or) > self.log_or_threshold:
                    epistasis[(g1, g2)] = odds_ratio
        
        self.epistasis_matrix = epistasis
        logger.info(f"Matrix built with {len(epistasis)} significant interactions.")

    def compute_ebv(self, genes: List[str]) -> np.ndarray:
        """
        Compute the 6-dimensional Epistatic Pressure Vector for a plasmid.
        """
        unique_genes = sorted(list(set(genes)))
        pos_ors = []
        neg_ors = []
        
        for g1, g2 in combinations(unique_genes, 2):
            pair = (g1, g2)
            if pair in self.epistasis_matrix:
                or_val = self.epistasis_matrix[pair]
                if or_val > 1:
                    pos_ors.append(or_val)
                else:
                    neg_ors.append(or_val)
        
        all_ors = pos_ors + neg_ors
        n_interact = len(all_ors)
        n_possible = max(1, len(unique_genes) * (len(unique_genes) - 1) / 2)
        
        v1 = np.mean(pos_ors) if pos_ors else 1.0
        v2 = np.mean(neg_ors) if neg_ors else 1.0
        v3 = n_interact / n_possible
        v4 = (len(pos_ors) - len(neg_ors)) / (n_interact + 1e-6)
        v5 = np.var(all_ors) if all_ors else 0.0
        v6 = np.max(all_ors) if all_ors else 1.0
        
        vec = np.array([v1, v2, v3, v4, v5, v6], dtype=np.float32)
        # Final safety check
        vec = np.nan_to_num(vec, nan=0.0, posinf=1000.0, neginf=0.0)
        return vec

    def compute_all_ebv(self, gene_lists: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        return {bid: self.compute_ebv(genes) for bid, genes in gene_lists.items()}
