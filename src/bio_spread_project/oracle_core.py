import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GeneVectorTable(nn.Module):
    """
    Learned h_dim=1024 vectors per gene family.
    Includes sigmoid gene-importance weights and 1/sqrt(count) scaling.
    """
    def __init__(self, vocab_size: int, h_dim: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.h_dim = h_dim
        
        # The core embedding table for gene families
        self.embedding = nn.Embedding(vocab_size, h_dim, padding_idx=0)
        
        # Sigmoid gene-importance weights (one per gene family)
        self.importance = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        """
        x_gene: [B, G] tensor of gene indices (padded with 0s)
        Returns: z_gene: [B, h_dim] aggregated vector
        """
        # [B, G] -> [B, G, h_dim]
        embeds = self.embedding(x_gene)
        
        # Get importance weights and apply sigmoid
        # [B, G]
        weights = torch.sigmoid(self.importance[x_gene])
        
        # Mask out padding (index 0)
        mask = (x_gene != 0).float()
        
        # Apply mask to weights
        # [B, G]
        weights = weights * mask
        
        # Count non-zero elements for scaling (clamped to 1 to avoid div by zero)
        # [B, 1]
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        # 1/sqrt(count) scaling
        scale = 1.0 / torch.sqrt(counts)
        
        # Weighted sum of embeddings
        # [B, G, 1] * [B, G, h_dim] -> [B, G, h_dim]
        weighted_embeds = weights.unsqueeze(-1) * embeds
        
        # Aggregate across genes: [B, h_dim]
        z_gene = weighted_embeds.sum(dim=1)
        
        # Apply scaling
        z_gene = z_gene * scale
        
        return z_gene

class TemporalGate(nn.Module):
    """
    TemporalGate(z, t) -- element-wise sigmoid modulation
    (no-op when t is None)
    """
    def __init__(self, d_dim: int = 64):
        super().__init__()
        # We project the time feature `t` to match the dimension of `z`
        self.time_proj = nn.Linear(1, d_dim)
        
    def forward(self, z: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: [B, D=64]
        t: [B, 1] optional time feature (e.g., years since reference)
        """
        if t is None:
            return z
            
        # [B, D]
        t_mod = self.time_proj(t)
        
        # Element-wise sigmoid modulation
        modulation = torch.sigmoid(t_mod)
        
        return z * modulation

class SovereignOracleNet(nn.Module):
    """
    The Sovereign Bio 'Oracle' v15 Architecture.
    A hardened, DNA-centric evidential deep learning model for geographic spread.
    """
    def __init__(self, vocab_size: int, h_dim: int = 1024, d_dim: int = 64):
        super().__init__()
        
        # Gene Vector Table
        self.gene_table = GeneVectorTable(vocab_size=vocab_size, h_dim=h_dim)
        
        # spectral_norm(Linear(1024 -> 64))
        self.proj = nn.utils.spectral_norm(nn.Linear(h_dim, d_dim))
        
        # Temporal Gate
        self.temporal_gate = TemporalGate(d_dim=d_dim)
        
        # layer_norm(z)
        self.layer_norm = nn.LayerNorm(d_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Linear(64 -> 2)
        self.evidential_head = nn.Linear(d_dim, 2)
        
    def forward(self, x_gene: torch.Tensor, t: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Oracle architecture.
        
        Returns:
            prob: [B] The probability of the target class (index 1)
            uncertainty: [B] The epistemic uncertainty score
            alpha: [B, 2] The raw Dirichlet parameters
        """
        # 1. Gene Vector Aggregation
        # z_gene: [B, 1024]
        z_gene = self.gene_table(x_gene)
        
        # 2. Spectral Norm Projection
        # z: [B, 64]
        z = self.proj(z_gene)
        
        # 3. Temporal Modulation
        z = self.temporal_gate(z, t)
        
        # 4. Layer Normalization
        z = self.layer_norm(z)
        
        # 4.5 Dropout
        z = self.dropout(z)
        
        # 5. Evidential Head
        # evidence: [B, 2] -> using softplus
        evidence = F.softplus(self.evidential_head(z))
        
        # alpha = evidence + 1
        alpha = evidence + 1.0
        
        # S = sum(alpha)
        S = alpha.sum(dim=1, keepdim=True)
        
        # prob = alpha_1 / S (Assuming class 1 is the positive 'spread' class)
        prob = alpha[:, 1] / S.squeeze()
        
        # uncertainty = 2 / S (Since K=2 classes)
        uncertainty = 2.0 / S.squeeze()
        
        return prob, uncertainty, alpha

def evidential_loss(alpha: torch.Tensor, y: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
    """
    Dirichlet Evidential Loss for binary classification.
    y: [B] containing 0 or 1 labels.
    alpha: [B, 2]
    """
    y_one_hot = F.one_hot(y.long(), num_classes=2).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Negative Log-Likelihood of Dirichlet
    loss_err = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    
    # KL Divergence Regularization to shrink evidence of incorrect class
    alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    
    kl_reg = torch.lgamma(S_tilde.squeeze()) - torch.lgamma(torch.tensor(2.0, device=alpha.device))
    kl_reg -= torch.sum(torch.lgamma(alpha_tilde), dim=1)
    kl_reg += torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1)
    
    loss = loss_err + lambda_reg * kl_reg
    return loss.mean()
