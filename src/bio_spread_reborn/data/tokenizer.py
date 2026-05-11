import polars as pl
import json
from pathlib import Path
from typing import List, Dict, Union

class GeneTokenizer:
    def __init__(self, max_len: int = 300):
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.inv_vocab = {0: '<PAD>', 1: '<UNK>'}
        
    def fit(self, gene_series: pl.Series):
        """
        Build vocabulary from a series of gene lists.
        """
        # Explode the list of lists into a single series of genes and get unique values
        unique_genes = gene_series.explode().unique().drop_nulls().to_list()
        
        # Sort for reproducibility
        for i, gene in enumerate(sorted(unique_genes), start=len(self.vocab)):
            self.vocab[gene] = i
            self.inv_vocab[i] = gene
            
    def encode(self, gene_list: List[str]) -> List[int]:
        """
        Tokenize a list of genes with truncation and OOV handling.
        """
        if gene_list is None:
            return [0] * self.max_len
            
        # Use a list comprehension for speed
        encoded = [self.vocab.get(g, 1) for g in gene_list[:self.max_len]]
        
        # Pad with zero (already optimized list concatenation)
        if len(encoded) < self.max_len:
            encoded.extend([0] * (self.max_len - len(encoded)))
        return encoded

    def save(self, path: Union[str, Path]):
        """Save vocabulary to a JSON file."""
        with open(path, 'w') as f:
            json.dump({
                "max_len": self.max_len,
                "vocab": self.vocab
            }, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GeneTokenizer':
        """Load vocabulary from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(max_len=data["max_len"])
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        return tokenizer

    def get_vocab_size(self) -> int:
        return len(self.vocab)
