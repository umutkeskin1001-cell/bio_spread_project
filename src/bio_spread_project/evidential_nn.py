import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EvidentialNN(nn.Module):
    """
    Dirichlet-based Evidential Neural Network for uncertainty quantification.
    """
    def __init__(self, input_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.evidence = nn.Linear(hidden, 2)
        
    def forward(self, x):
        h = self.net(x)
        ev = torch.exp(self.evidence(h))  # ensure evidence > 0
        alpha = ev + 1
        prob = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha, prob

def evidential_loss(alpha, y, lambda_cal=0.1):
    """
    Evidential classification loss with calibration regularizer.
    """
    S = alpha.sum(dim=1)
    p = alpha / S.unsqueeze(1)
    y_onehot = F.one_hot(y.long(), num_classes=2).float()
    
    error = (y_onehot - p) ** 2
    var = p * (1 - p) / (S.unsqueeze(1) + 1)
    loss = (error + 0.1 * var).sum(dim=1).mean()
    
    cal_loss = F.mse_loss(p[:, 1], y.float())
    return loss + lambda_cal * cal_loss

def train_evidential_nn_oof(X, y, groups, n_splits=3, epochs=20, lr=1e-3):
    """
    Generates Out-of-Fold evidential probabilities and uncertainty estimates.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    device = torch.device('cpu')
    n = len(y)
    evid_prob = np.zeros(n)
    evid_unc = np.zeros(n)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in sgkf.split(X, y.astype(int), groups=groups):
        X_tr = torch.FloatTensor(X[tr_idx])
        y_tr = torch.FloatTensor(y[tr_idx])
        
        model = EvidentialNN(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            alpha, _ = model(X_tr)
            loss = evidential_loss(alpha, y_tr)
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            alpha_val, prob_val = model(torch.FloatTensor(X[val_idx]))
            S_val = alpha_val.sum(dim=1)
            evid_prob[val_idx] = prob_val[:, 1].numpy()
            evid_unc[val_idx] = 2.0 / S_val.numpy()  # uncertainty = 2 / total evidence
            
    return evid_prob, evid_unc
