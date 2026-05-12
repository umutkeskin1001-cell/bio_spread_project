import torch
import esm
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import polars as pl
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class PFPExtractor:
    """
    Pillar 1: Evolutionary Fitness Fingerprint (PFP).
    Extracts metabolic load and fitness features using ESM-2 (8M) and PCA.
    """
    def __init__(self, n_components: int = 16, device: str = "cpu"):
        self.n_components = n_components
        self.device = device
        # Use the lightest ESM-2 model (8M parameters)
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def extract_plasmid_embedding(self, protein_sequences: List[Tuple[str, str]]) -> np.ndarray:
        """
        Extract mean-pooled embedding for a plasmid (average of its proteins).
        """
        embeddings = []
        for header, seq in protein_sequences:
            # Prepare data
            data = [(header, seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
            
            # Representation from the last layer (index 6 for t6 model)
            token_representations = results["representations"][6]
            
            # Mean pooling over sequence (ignoring BOS/EOS tokens)
            # tokens: [B, L] -> representations: [B, L, D]
            # seq_len = len(seq)
            # We take 1:seq_len+1 to exclude BOS and EOS
            seq_emb = token_representations[0, 1:len(seq)+1].mean(dim=0).cpu().numpy()
            embeddings.append(seq_emb)
        
        if not embeddings:
            return np.zeros(320) # ESM-2 8M embedding dimension
            
        return np.mean(embeddings, axis=0)

    def compute_pfp(self, backbone_id_to_seqs: Dict[str, List[Tuple[str, str]]], backbone_meta: pl.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute PFP features for all plasmids.
        """
        logger.info(f"Computing PFP embeddings for {len(backbone_id_to_seqs)} backbones...")
        pfp_raw = {}
        
        for bid, seqs in backbone_id_to_seqs.items():
            plasmid_emb = self.extract_plasmid_embedding(seqs)
            
            # Add intrinsic traits from meta
            meta = backbone_meta.filter(pl.col("backbone_id") == bid)
            if not meta.is_empty():
                # plasmid_size, gc_content, replicon_count, toxin_count, conjugation_score
                extra_cols = ["size", "gc", "n_replicon_types", "n_relaxase_types"]
                extra_feats = [meta[c][0] if c in meta.columns else 0 for c in extra_cols]
                # Add one more for conjugation_score (dummy if missing)
                extra_feats.append(meta["conjugation_score"][0] if "conjugation_score" in meta.columns else 0)
                extra_feats = np.array(extra_feats, dtype=np.float32)
            else:
                extra_feats = np.zeros(5, dtype=np.float32)
            
            pfp_raw[bid] = np.concatenate([plasmid_emb, extra_feats])
            
        all_vecs = np.stack(list(pfp_raw.values()))
        
        # PCA to 16 dimensions
        logger.info(f"Reducing PFP to {self.n_components} dimensions using PCA...")
        if not self.is_fitted:
            self.pca.fit(all_vecs)
            self.is_fitted = True
            
        reduced_vecs = self.pca.transform(all_vecs)
        
        pfp_features = {bid: reduced_vecs[i] for i, bid in enumerate(pfp_raw.keys())}
        return pfp_features

    def save(self, path: Path):
        import joblib
        joblib.dump({"pca": self.pca, "is_fitted": self.is_fitted}, path)

    def load(self, path: Path):
        import joblib
        data = joblib.load(path)
        self.pca = data["pca"]
        self.is_fitted = data["is_fitted"]
