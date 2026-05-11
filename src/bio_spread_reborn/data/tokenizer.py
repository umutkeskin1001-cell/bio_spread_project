import polars as pl
from typing import List, Dict

class GeneTokenizer:
    def __init__(self, max_len: int = 300):
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        
    def fit(self, gene_series: pl.Series):
        """
        Build vocabulary from a series of gene lists.
        """
        # Explode the list of lists into a single series of genes and get unique values
        unique_genes = gene_series.explode().unique().drop_nulls().to_list()
        
        # Sort for reproducibility
        for i, gene in enumerate(sorted(unique_genes), start=len(self.vocab)):
            self.vocab[gene] = i
            
    def encode(self, gene_list: List[str]) -> List[int]:
        """
        Tokenize a list of genes with truncation and OOV handling.
        """
        if gene_list is None:
            return [0] * self.max_len
            
        encoded = [self.vocab.get(g, 1) for g in gene_list[:self.max_len]]
        # Pad with zero
        padding = [0] * (self.max_len - len(encoded))
        return encoded + padding

    def get_vocab_size(self) -> int:
        return len(self.vocab)
