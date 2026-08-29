import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


class QuantileDAGModel_Linear:
    """
    Linear quantile regression model for each node in the DAG.
    Uses statsmodels QuantReg. Fast and interpretable; assumes linear
    relationships between a node and its upstream nodes.
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
        preds = np.array([self.models[q].predict(X_with_const) for q in self.quantiles])
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
    Uses sklearn GradientBoostingRegressor with pinball loss. More flexible
    than linear; captures nonlinear relationships. Hyperparameters are selected
    via 5-fold cross-validation minimizing the average pinball loss across
    quantile levels q in {0.1, 0.3, 0.5, 0.7, 0.9}.
    """
    def __init__(self, quantiles=np.arange(0.01, 1.0, 0.02)):
        from sklearn.ensemble import GradientBoostingRegressor
        self.quantiles        = np.asarray(quantiles)
        self.GradientBoosting = GradientBoostingRegressor
        self.models           = {}
        self.best_params      = None

    def fit_with_cv(self, X, y, cv=5, param_grid=None):
        """
        Select optimal n_estimators and max_depth via cross-validation,
        then fit the final model on the full dataset using the best parameters.

        Args:
            X          : feature matrix (pandas DataFrame or numpy array)
            y          : target variable
            cv         : number of cross-validation folds (default 5)
            param_grid : dict of parameters to search over
                         default: {'n_estimators': [20, 50, 100],
                                   'max_depth':    [2, 3, 4]}
        """
        from sklearn.model_selection import KFold
        if param_grid is None:
            param_grid = {
                'n_estimators': [20, 50, 100],
                'max_depth':    [2, 3, 4],
            }
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        kf        = KFold(n_splits=cv, shuffle=True, random_state=42)
        q_cv_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_val_loss = float('inf')
        best_params   = None
        n_combos = len(param_grid['n_estimators']) * len(param_grid['max_depth'])
        print(f"    Running {cv}-fold CV over {n_combos} parameter combinations...")
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                fold_losses = []
                for train_idx, val_idx in kf.split(X_arr):
                    X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                    y_train, y_val = y_arr[train_idx], y_arr[val_idx]
                    q_losses = []
                    for q_cv in q_cv_list:
                        model = self.GradientBoosting(
                            loss='quantile', alpha=q_cv,
                            n_estimators=n_est, max_depth=depth
                        )
                        model.fit(X_train, y_train)
                        pred     = model.predict(X_val)
                        err      = y_val - pred
                        val_loss = np.mean(np.where(err >= 0,
                                                    q_cv * err,
                                                    (q_cv - 1) * err))
                        q_losses.append(val_loss)
                    fold_losses.append(np.mean(q_losses))
                avg_loss = np.mean(fold_losses)
                print(f"      n_estimators={n_est}, max_depth={depth}: avg val loss = {avg_loss:.6f}")
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    best_params   = {'n_estimators': n_est, 'max_depth': depth}
        self.best_params = best_params
        print(f"    Best params: {best_params} (val loss = {best_val_loss:.6f})")
        print(f"    Fitting {len(self.quantiles)} XGBoost quantile models with best params...")
        for q in self.quantiles:
            model = self.GradientBoosting(
                loss='quantile', alpha=q,
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth']
            )
            model.fit(X_arr, y_arr)
            self.models[q] = model
        print("    Fitting complete!")

    def fit(self, X, y):
        """Fit with default parameters (n_estimators=50, max_depth=3)."""
        print(f"    Fitting {len(self.quantiles)} XGBoost quantile models...")
        for q in self.quantiles:
            model = self.GradientBoosting(
                loss='quantile', alpha=q, n_estimators=50, max_depth=3)
            model.fit(X, y)
            self.models[q] = model
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
        """Map uniform noise E to output via linear interpolation across quantile grid."""
        X = np.asarray(X, dtype=np.float32)
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
    Neural network quantile regression model using PyTorch with pinball loss.
    Trains one feed-forward network (input -> 64 -> 32 -> 1) per quantile level.
    Uses Early Stopping (patience=50, val_ratio=0.2) to determine the optimal
    number of training epochs, preventing overfitting.
    """
    def __init__(self, quantiles=np.arange(0.05, 1.0, 0.05)):
        self.quantiles = np.asarray(quantiles)
        self.models    = {}

    def _build_network(self, input_dim):
        """Build a two-hidden-layer feed-forward network."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1)
        )

    def fit(self, X, y, val_ratio=0.2, patience=50, plot_loss=False):
        """
        Fit one neural network per quantile level using pinball loss and Early Stopping.

        Args:
            X         : feature matrix
            y         : target variable
            val_ratio : fraction of data held out as validation set (default 0.2)
            patience  : epochs without improvement before stopping (default 50)
            plot_loss : if True, plot training vs validation loss curve for the
                        median quantile (only for the node passed to this call)
        """
        import torch
        import matplotlib.pyplot as plt
        X_arr   = np.asarray(X, dtype=np.float32)
        y_arr   = np.asarray(y, dtype=np.float32)
        n       = len(X_arr)
        n_val   = int(n * val_ratio)
        n_train = n - n_val
        idx       = np.random.permutation(n)
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:]
        X_train = torch.tensor(X_arr[train_idx])
        y_train = torch.tensor(y_arr[train_idx]).unsqueeze(1)
        X_val   = torch.tensor(X_arr[val_idx])
        y_val   = torch.tensor(y_arr[val_idx]).unsqueeze(1)
        n_feat = X_arr.shape[1]
        print(f"    Fitting {len(self.quantiles)} neural network quantile models...")
        plot_q = self.quantiles[len(self.quantiles) // 2]
        for q in self.quantiles:
            net       = self._build_network(n_feat)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_weights  = None
            counter       = 0
            train_losses  = []
            val_losses    = []
            for epoch in range(1000):
                net.train()
                optimizer.zero_grad()
                pred = net(X_train)
                err  = y_train - pred
                loss = torch.mean(torch.where(err >= 0, q * err, (q - 1) * err))
                loss.backward()
                optimizer.step()
                net.eval()
                with torch.no_grad():
                    pred_val = net(X_val)
                    err_val  = y_val - pred_val
                    val_loss = torch.mean(
                        torch.where(err_val >= 0, q * err_val, (q - 1) * err_val)
                    ).item()
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights  = {k: v.clone() for k, v in net.state_dict().items()}
                    counter       = 0
                else:
                    counter += 1
                if counter >= patience:
                    print(f"      q={q:.2f}: early stopping at epoch {epoch}")
                    break
            if plot_loss and abs(q - plot_q) < 0.01:
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Training Loss',   color='#1f77b4')
                plt.plot(val_losses,   label='Validation Loss', color='#d62728')
                plt.axvline(x=len(train_losses) - patience, color='green',
                            linestyle='--', label=f'Early Stopping (epoch {len(train_losses)})')
                plt.xlabel('Epoch')
                plt.ylabel('Pinball Loss')
                plt.title(f'Training vs Validation Loss (q={q:.2f})')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'loss_curve_q{q:.2f}.png', dpi=300, bbox_inches='tight')
                plt.show()
            net.load_state_dict(best_weights)
            net.eval()
            self.models[q] = net
        print("    Fitting complete!")

    def predict_from_noise(self, X, E):
        """Map uniform noise E to output via linear interpolation across quantile grid."""
        import torch
        X_arr = np.asarray(X, dtype=np.float32)
        X_t   = torch.tensor(X_arr)
        with torch.no_grad():
            preds = np.array([self.models[q](X_t).squeeze().numpy() for q in self.quantiles])
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
