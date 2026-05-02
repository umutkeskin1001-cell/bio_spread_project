import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from torch.optim import Adam

class EvidentialRiskModel(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.logit_head = nn.Linear(hidden, 1)
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus()  # evidence >= 0
        )
    
    def forward(self, x):
        h = self.shared(x)
        logit = self.logit_head(h)
        evidence = self.evidence_head(h)
        return logit, evidence

def evidential_loss(logit, evidence, y_true, lambda_reg=0.1):
    """
    Combine binary cross-entropy for logit and an evidence regularizer.
    """
    # BCE loss on logit (converted to probability via sigmoid)
    bce = F.binary_cross_entropy_with_logits(logit.squeeze(), y_true)
    
    # Evidence regularizer: encourage higher evidence when correct, lower when wrong
    prob = torch.sigmoid(logit.squeeze())
    error = (y_true - prob).abs()
    reg = (evidence.squeeze() * error).mean()
    return bce + lambda_reg * reg

class EvidentialMetaEstimator:
    def __init__(self, input_dim, hidden=64, lambda_reg=0.1, lgb_params=None):
        self.input_dim = input_dim
        self.hidden = hidden
        self.lambda_reg = lambda_reg
        self.evidential_model = None
        self.lgb_model = None
        self.lgb_params = lgb_params or {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'verbose': -1,
            'random_state': 42,
        }
    
    def fit(self, X_train, y_train, **kwargs):
        import numpy as np
        import torch
        from torch.optim import Adam
        from sklearn.model_selection import StratifiedKFold
        import lightgbm as lgb
        
        device = torch.device('cpu')
        
        # Generate OOF predictions for the evidential risk score
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_risk_evidential = np.zeros(len(y_train))
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = torch.tensor(X_train[train_idx], dtype=torch.float32)
            y_fold_train = torch.tensor(y_train[train_idx], dtype=torch.float32)
            X_fold_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
            
            fold_model = EvidentialRiskModel(self.input_dim, self.hidden).to(device)
            opt = Adam(fold_model.parameters(), lr=0.01)
            
            fold_model.train()
            for _ in range(20):
                opt.zero_grad()
                logit, evidence = fold_model(X_fold_train)
                loss = evidential_loss(logit, evidence, y_fold_train, self.lambda_reg)
                loss.backward()
                opt.step()
                
            fold_model.eval()
            with torch.no_grad():
                logit, evidence = fold_model(X_fold_val)
                prob = torch.sigmoid(logit.squeeze())
                evid = evidence.squeeze()
                risk_evidential = prob * torch.tanh(evid)
                oof_risk_evidential[val_idx] = risk_evidential.cpu().numpy()
                
        # Train final evidential model on full data
        self.evidential_model = EvidentialRiskModel(self.input_dim, self.hidden).to(device)
        opt = Adam(self.evidential_model.parameters(), lr=0.01)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        
        self.evidential_model.train()
        for _ in range(20):
            opt.zero_grad()
            logit, evidence = self.evidential_model(X_t)
            loss = evidential_loss(logit, evidence, y_t, self.lambda_reg)
            loss.backward()
            opt.step()
        
        # Prepare final feature matrix for LightGBM using OOF evidential risk
        X_final = np.hstack([X_train, oof_risk_evidential.reshape(-1, 1)])
        
        # Train LightGBM ranker with Platt scaling to improve calibration
        base_lgb = lgb.LGBMClassifier(**self.lgb_params)
        from sklearn.calibration import CalibratedClassifierCV
        self.lgb_model = CalibratedClassifierCV(base_lgb, method='sigmoid', cv=5)
        self.lgb_model.fit(X_final, y_train)
        return self
    
    def predict_proba(self, X):
        import torch
        import numpy as np
        device = torch.device('cpu')
        X_t = torch.tensor(X, dtype=torch.float32)
        self.evidential_model.eval()
        with torch.no_grad():
            logit, evidence = self.evidential_model(X_t)
            prob = torch.sigmoid(logit.squeeze()).cpu().numpy()
            evid = evidence.squeeze().cpu().numpy()
            evid_score = np.tanh(evid)
            risk_evidential = prob * evid_score
        
        X_final = np.hstack([X, risk_evidential.reshape(-1, 1)])
        return self.lgb_model.predict_proba(X_final)
