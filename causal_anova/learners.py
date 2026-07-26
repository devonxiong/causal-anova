import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


class QuantileDAGModel_Linear:
    """
    Linear quantile regression model for each node in the DAG.
    Uses statsmodels QuantReg. Fast and interpretable.
    """
    def __init__(self, quantiles=np.arange(0.05, 1.0, 0.05)):
        self.quantiles = np.asarray(quantiles)
        self.models    = {}

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        X_with_const = np.column_stack([np.ones(len(X_arr)), X_arr])
        print(f"    Fitting {len(self.quantiles)} linear quantile models...")
        for q in self.quantiles:
            model = QuantReg(y_arr, X_with_const).fit(q=q, max_iter=2000)
            self.models[q] = model
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
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
    More flexible than linear, captures nonlinear relationships.
    """
    def __init__(self, quantiles=np.arange(0.01, 1.0, 0.02)):
        from sklearn.ensemble import GradientBoostingRegressor
        self.quantiles        = np.asarray(quantiles)
        self.GradientBoosting = GradientBoostingRegressor
        self.models           = {}

    def fit(self, X, y):
        print(f"    Fitting {len(self.quantiles)} XGBoost quantile models...")
        for q in self.quantiles:
            model = self.GradientBoosting(
                loss='quantile', alpha=q, n_estimators=50, max_depth=3)
            model.fit(X, y)
            self.models[q] = model
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
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


class QuantileDAGModel_NeuralNetwork:
    """
    Neural network quantile regression using PyTorch with pinball loss.
    Trains one network per quantile level with the proper asymmetric pinball loss.
    Smooth and nonlinear, suitable for complex relationships.
    """
    def __init__(self, quantiles=np.arange(0.05, 1.0, 0.05)):
        self.quantiles = np.asarray(quantiles)
        self.models    = {}

    def _build_network(self, input_dim):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def fit(self, X, y):
        import torch
        X_arr  = np.asarray(X, dtype=np.float32)
        y_arr  = np.asarray(y, dtype=np.float32)
        X_t    = torch.tensor(X_arr)
        y_t    = torch.tensor(y_arr).unsqueeze(1)
        n_feat = X_arr.shape[1]
        print(f"    Fitting {len(self.quantiles)} neural network quantile models...")
        for q in self.quantiles:
            net       = self._build_network(n_feat)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            for epoch in range(300):
                net.train()
                optimizer.zero_grad()
                pred = net(X_t)
                err  = y_t - pred
                loss = torch.mean(
                    torch.where(err >= 0, q * err, (q - 1) * err)
                )
                loss.backward()
                optimizer.step()
            net.eval()
            self.models[q] = net
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
        import torch
        X_arr = np.asarray(X, dtype=np.float32)
        X_t   = torch.tensor(X_arr)
        with torch.no_grad():
            preds = np.array([
                self.models[q](X_t).squeeze().numpy()
                for q in self.quantiles
            ])
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


# Registry mapping learner name -> class
_LEARNER_REGISTRY = {
    'linear':         QuantileDAGModel_Linear,
    'xgboost':        QuantileDAGModel_XGBoost,
    'neural_network': QuantileDAGModel_NeuralNetwork,
}
