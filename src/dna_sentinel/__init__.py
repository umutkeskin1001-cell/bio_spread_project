"""Cassiopeia Prime: DNA-only multi-task plasmid risk modeling.

Predicts mobility class, AMR probability, and geographic expansion risk
directly from raw FASTA sequence without BLAST, gene calling, or
metadata. Champion checkpoint: v14, 568,437 trainable parameters.
"""

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig, compress_checkpoint, load_compressed

__all__ = ["Cassiopeia", "CassiopeiaConfig", "compress_checkpoint", "load_compressed"]

