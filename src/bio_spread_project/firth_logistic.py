import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

class FirthLogistic(BaseEstimator, ClassifierMixin):
    """
    Firth's Penalized Likelihood Logistic Regression.
    Reduces bias in maximum likelihood estimates, especially for small samples or rare events.
    The penalty term is 0.5 * log|X^T W X|, the Jeffreys prior.
    """
    _estimator_type = "classifier"
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 100):
        self.alpha = alpha
        self.max_iter = max_iter

    def _log_likelihood(self, beta, X, y):
        logits = X @ beta
        p = expit(logits)
        # Standard log-likelihood with epsilon for numerical stability
        eps = 1e-15
        ll = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        
        # Memory-efficient Firth penalty: 0.5 * log |X^T W X|
        # W is diagonal with w_ii = p_i * (1 - p_i)
        # X^T W X = (X.T * w) @ X
        w = p * (1 - p)
        xtwx = (X.T * w) @ X
        
        sign, logdet = np.linalg.slogdet(xtwx)
        penalty = 0.5 * logdet if sign > 0 else -1e10
        
        return -(ll + self.alpha * penalty)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X_const = np.hstack([np.ones((X.shape[0], 1)), X])
        self.classes_ = np.unique(y)
        
        # Warm start using standard Logistic Regression to find a good initial neighborhood
        warm_start = LogisticRegression(class_weight="balanced", max_iter=200).fit(X, y)
        initial_beta = np.concatenate([warm_start.intercept_, warm_start.coef_.flatten()])
        
        res = minimize(
            self._log_likelihood, 
            initial_beta, 
            args=(X_const, y), 
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        
        # Fallback if Firth optimization fails: use warm start coefficients
        if not res.success:
            self.intercept_ = initial_beta[0]
            self.coef_ = initial_beta[1:].reshape(1, -1)
        else:
            self.intercept_ = res.x[0]
            self.coef_ = res.x[1:].reshape(1, -1)
            
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        logits = self.intercept_ + X @ self.coef_.T
        p = expit(logits)
        return np.hstack([1 - p, p])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
