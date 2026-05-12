import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
from itertools import combinations
from bio_spread_reborn.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class EvidentialTrainer:
    def __init__(self, model, config, device="cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.lr = config.get("training", {}).get("learning_rate", 1e-3)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.epochs = config.get("training", {}).get("epochs", 50)
        self.patience = config.get("training", {}).get("patience", 5)
        self.artifact_dir = Path("artifacts") / config.get("run_id", "default")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Focal Loss factor
        self.gamma = 2.0

    def evidential_loss(self, alpha, y, epoch):
        """
        Dirichlet Evidential Loss with Focal Factor.
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        y_one_hot = torch.zeros_like(alpha).scatter_(1, y.unsqueeze(1), 1)
        
        # Likelihood loss
        A = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # Focal factor: (1 - p_target)^gamma
        p = alpha / S
        p_target = torch.sum(y_one_hot * p, dim=1, keepdim=True)
        focal_factor = (1 - p_target) ** self.gamma
        
        # KL Divergence Regularization
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(2.0)) - \
             torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) + \
             torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        # Annealing coefficient
        lambda_t = min(1.0, epoch / 10.0)
        
        loss = focal_factor * (A + lambda_t * kl)
        return torch.mean(loss)

    def ranknet_loss(self, prob_a, prob_b, true_order):
        """
        RankNet pairwise loss. true_order: 1 if a > b, 0 if b > a.
        """
        diff = prob_a - prob_b
        o = torch.sigmoid(diff)
        loss = -true_order * torch.log(o + 1e-8) - (1 - true_order) * torch.log(1 - o + 1e-8)
        return loss.mean()

    def fit(self, train_loader, val_loader, ranking_pairs=None):
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                # Batch is now a dict
                x_gene = batch["x_gene"].to(self.device)
                x_func = batch["x_func"].to(self.device)
                x_hist = batch["x_hist"].to(self.device)
                x_extra = batch["x_extra"].to(self.device)
                t = batch["t"].to(self.device)
                y = batch["y"].to(self.device)
                bids = batch["bids"]
                
                self.optimizer.zero_grad()
                prob, unc, alpha = self.model(x_gene, x_func, x_hist, x_extra, t)
                e_loss = self.evidential_loss(alpha, y, epoch)
                
                # Pillar 4: Temporal Ranking Loss (Pairwise)
                r_loss = torch.tensor(0.0, device=self.device)
                # Find pairs of same BID
                bid_to_idx = defaultdict(list)
                for i, bid in enumerate(bids):
                    bid_to_idx[bid].append(i)
                
                pairs = []
                for bid, indices in bid_to_idx.items():
                    if len(indices) > 1:
                        # Create combinations of snapshots
                        for i, j in combinations(indices, 2):
                            # Sort by year (t)
                            if t[i] > t[j]:
                                pairs.append((i, j, 1.0)) # i is newer/more risky
                            elif t[j] > t[i]:
                                pairs.append((j, i, 1.0)) # j is newer
                
                if pairs:
                    idx_a = [p[0] for p in pairs]
                    idx_b = [p[1] for p in pairs]
                    orders = torch.tensor([p[2] for p in pairs], device=self.device)
                    r_loss = self.ranknet_loss(prob[idx_a], prob[idx_b], orders)
                
                loss = e_loss + 0.2 * r_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_auc = val_metrics["roc_auc"]
            logger.info(f"Epoch {epoch}/{self.epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                self.save_model("best_model.pt")
                logger.info(f"New best model saved with AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.load_model("best_model.pt")
        return self.artifact_dir

    def evaluate(self, loader):
        self.model.eval()
        all_probs, all_labels, all_uncs = [], [], []
        with torch.no_grad():
            for batch in loader:
                x_gene = batch["x_gene"].to(self.device)
                x_func = batch["x_func"].to(self.device)
                x_hist = batch["x_hist"].to(self.device)
                x_extra = batch["x_extra"].to(self.device)
                t = batch["t"].to(self.device)
                y = batch["y"].to(self.device)
                
                prob, unc, alpha = self.model(x_gene, x_func, x_hist, x_extra, t)
                all_probs.extend(prob.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_uncs.extend(unc.cpu().numpy())
        
        return calculate_metrics(all_labels, all_probs, all_uncs)

    def save_model(self, name):
        torch.save(self.model.state_dict(), self.artifact_dir / name)
        with open(self.artifact_dir / "config.json", "w") as f:
            json.dump(self.config, f)

    def load_model(self, name):
        path = self.artifact_dir / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
