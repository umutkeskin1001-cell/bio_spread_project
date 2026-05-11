import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import logging

class EvidentialTrainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0
        
    def _kl_dirichlet(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between two Dirichlet distributions.
        beta is usually torch.ones_like(alpha) for uniform prior.
        """
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_beta = beta.sum(dim=1, keepdim=True)
        
        lnB = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(S_alpha)
        lnB_prior = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)
        
        kl = (alpha - beta) * (dg_alpha - dg_S_alpha)
        kl = kl.sum(dim=1, keepdim=True) + lnB_prior - lnB
        return kl.squeeze(-1)

    def evidential_loss(self, alpha: torch.Tensor, y: torch.Tensor, epoch: int):
        """
        Dirichlet Evidential Loss with KL annealing.
        """
        num_classes = 2
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected log-likelihood loss
        # ll = sum(y_i * (digamma(alpha_i) - digamma(S)))
        ll = (y_onehot * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
        
        # KL Divergence regularization (annealed)
        # We only regularize the non-target evidence towards 0 (alpha towards 1)
        # alpha_reg = y_onehot + (1 - y_onehot) * alpha
        # Actually, standard approach: KL(Dir(alpha_tilde) || Dir(1,1))
        # where alpha_tilde = y + (1-y)*alpha
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        kl = self._kl_dirichlet(alpha_tilde, torch.ones_like(alpha_tilde))
        
        # Annealing coefficient
        total_epochs = self.config['training']['epochs']
        lambda_reg = self.config['training']['kl_annealing'] * min(1.0, epoch / (total_epochs * 0.5))
        
        loss = -ll.mean() + lambda_reg * kl.mean()
        return loss
    
    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        for x, t, y in loader:
            x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            prob, unc, alpha = self.model(x, t)
            loss = self.evidential_loss(alpha, y, epoch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        all_probs = []
        all_targets = []
        for x, t, y in loader:
            x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
            prob, unc, alpha = self.model(x, t)
            all_probs.extend(prob.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        if len(np.unique(all_targets)) < 2:
            return 0.5 # Default AUC if only one class present
        return roc_auc_score(all_targets, all_probs)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']
        
        logging.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_auc = self.evaluate(val_loader)
            self.scheduler.step(val_auc)
            
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.pt')
                self.epochs_no_improve = 0
                logging.info(f"New best model saved with AUC: {val_auc:.4f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best weights
        if Path('best_model.pt').exists():
            self.model.load_state_dict(torch.load('best_model.pt'))
            logging.info("Training complete. Best model loaded.")
