import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


class QuantileDAGModel_Linear:
    """
    Linear quantile regression model for each node in the DAG.
    Uses statsmodels QuantReg instead of GradientBoosting.
    Faster and more interpretable, suitable for simpler DAG structures.
    """
    def __init__(self, quantiles=np.arange(0.05, 1.0, 0.05)):
        self.quantiles = np.asarray(quantiles)
        self.models    = {}

    def fit(self, X, y):
        """Fit one linear quantile regression per quantile level."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        X_with_const = np.column_stack([np.ones(len(X_arr)), X_arr])
        print(f"    Fitting {len(self.quantiles)} linear quantile models...")
        for q in self.quantiles:
            model = QuantReg(y_arr, X_with_const).fit(q=q, max_iter=2000)
            self.models[q] = model
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
        """Map uniform noise E to output via linear interpolation across quantile grid."""
        X_arr = np.asarray(X)
        X_with_const = np.column_stack([np.ones(len(X_arr)), X_arr])
        preds = np.array([self.models[q].predict(X_with_const)
                          for q in self.quantiles])
        preds = np.sort(preds, axis=0)

        q   = self.quantiles
        n   = X_arr.shape[0]
        col = np.arange(n)
        idx = np.clip(np.searchsorted(q, E), 1, len(q) - 1)
        lo  = idx - 1
        q0, q1 = q[lo], q[idx]
        p0, p1 = preds[lo, col], preds[idx, col]
        w   = np.where(q1 > q0, (E - q0) / (q1 - q0), 0.0)
        out = p0 + w * (p1 - p0)
        out = np.where(E <= q[0],  preds[0,  col], out)
        out = np.where(E >= q[-1], preds[-1, col], out)
        return out


class QuantileDAGModel_XGBoost:
    """
    Gradient boosting quantile regression model for each node in the DAG.
    More flexible than linear, suitable for complex nonlinear relationships.
    """
    def __init__(self, quantiles=np.arange(0.01, 1.0, 0.02)):
        from sklearn.ensemble import GradientBoostingRegressor
        self.quantiles        = np.asarray(quantiles)
        self.GradientBoosting = GradientBoostingRegressor
        self.models           = {}

    def fit(self, X, y):
        """Fit one GradientBoostingRegressor per quantile level."""
        print(f"    Fitting {len(self.quantiles)} XGBoost quantile models...")
        for q in self.quantiles:
            model = self.GradientBoosting(
                loss='quantile', alpha=q, n_estimators=50, max_depth=3)
            model.fit(X, y)
            self.models[q] = model
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
        """Map uniform noise E to output via linear interpolation across quantile grid."""
        preds = np.array([self.models[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=0)

        q   = self.quantiles
        n   = X.shape[0]
        col = np.arange(n)
        idx = np.clip(np.searchsorted(q, E), 1, len(q) - 1)
        lo  = idx - 1
        q0, q1 = q[lo], q[idx]
        p0, p1 = preds[lo, col], preds[idx, col]
        w   = np.where(q1 > q0, (E - q0) / (q1 - q0), 0.0)
        out = p0 + w * (p1 - p0)
        out = np.where(E <= q[0],  preds[0,  col], out)
        out = np.where(E >= q[-1], preds[-1, col], out)
        return out


# Registry mapping learner name -> class
_LEARNER_REGISTRY = {
    'linear':  QuantileDAGModel_Linear,
    'xgboost': QuantileDAGModel_XGBoost,
}