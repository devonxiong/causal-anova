# causal-anova

A Python package for **Causal ANOVA explainability analysis** on user-specified DAGs (Directed Acyclic Graphs).
Given a causal graph and observational data, `causal-anova` quantifies how much of the variance in an outcome node is *causally explained* by each node - individually, jointly, and through their interactions - using a variance-decomposition approach analogous to classical ANOVA, but grounded in counterfactuals.

## How It Works
The pipeline consists of two stages:

1. **Structural model fitting.** For every node in the DAG, a quantile regression model is fitted on the nodes that directly influence it. Instead of predicting a single conditional mean, each model learns the full conditional distribution across a grid of quantile levels. This lets us represent each node as a deterministic function of its upstream nodes plus an independent uniform noise term.

2. **CRN explainability estimation.** Explainability scores are estimated with a Pick-Freeze estimator. For each subset *S* of nodes in `roots`, we compare outcomes generated from a base noise draw against outcomes where only the noise of the nodes in *S* has been resampled. Sharing the same noise draws across all subsets within a repetition reduces variance and guarantees non-negative interaction terms. Scores are averaged over multiple independent runs, and standard errors are reported across runs.

Interaction terms are then recovered via **inclusion-exclusion**. For two nodes A and B:

```
ξ(A ∧ B) = ξ(A) + ξ(B) − ξ(A, B)
```

## Installation
```bash
pip install -r requirements.txt
```
Dependencies include:
- `numpy`, `pandas`
- `statsmodels` (linear quantile regression)
- `scikit-learn` (gradient boosting quantile regression)
- `torch` (neural network learner, only required if you use `'neural_network'`)
- `matplotlib`, `matplotlib-venn` (Venn diagram output)

## Project Structure

```
causal_anova/
├── __init__.py    # Package entry point (exports causal_anova)
├── main.py        # Orchestration: causal_anova() public API
├── learners.py    # Quantile regression learners (linear / xgboost / neural network)
├── simulator.py   # Builds the DAG simulator
├── crn.py         # CRN Pick-Freeze estimator and inclusion-exclusion interactions
└── output.py      # Results formatting, console printing, Venn diagrams
```

## Quick Start

The following example is adapted from `examples/example_compas.py`, which demonstrates the package on the real COMPAS recidivism dataset.

```python
from causal_anova import causal_anova

dag = {
    'roots':     ['Sex', 'Race', 'Age_Cat'],
    'mediators': ['Priors_Count', 'Charge_Degree'],
    'outcome':   'Two_Year_Recid',
    'edges': [
        ('Sex', 'Priors_Count'), ('Race', 'Priors_Count'), ('Age_Cat', 'Priors_Count'),
        ('Sex', 'Charge_Degree'), ('Race', 'Charge_Degree'), ('Age_Cat', 'Charge_Degree'),
        ('Priors_Count', 'Charge_Degree'),
        ('Sex', 'Two_Year_Recid'), ('Race', 'Two_Year_Recid'), ('Age_Cat', 'Two_Year_Recid'),
        ('Priors_Count', 'Two_Year_Recid'), ('Charge_Degree', 'Two_Year_Recid'),
    ]
}

# Use 'linear' for speed, 'xgboost' or 'neural_network' for more flexibility
results, mean, inter = causal_anova(
    dag=dag, data=df, learner='linear',
    compute_total=True, compute_pairwise=True,
    num_samples=10000, n_runs=8, plot_venn=True,
)
```

## Specifying the DAG
The `dag` argument is a plain dictionary with four keys:

| Key | Type | Description |
|---|---|---|
| `roots` | `list[str]` | Nodes whose explainability you want to measure |
| `mediators` | `list[str]` | Nodes on causal paths between explanatory and outcome nodes (may be empty `[]`) |
| `outcome` | `str` | The single outcome node |
| `edges` | `list[tuple]` | `(source, target)` pairs defining the causal structure |

Requirements:
- Every node listed under `mediators` and the `outcome` node must have at least one node pointing to it in `edges` (otherwise, a `ValueError` is raised).
- `data` must be a pandas DataFrame containing **all** node columns, with numeric values.
- Nodes listed under `mediators` should be in topological order, since values are computed from left to right along the DAG.
- Venn diagrams are only produced for 2 or 3 nodes in `roots`; total and interaction scores work for any number of nodes (note that subsets grow exponentially, so runtime increases quickly with more nodes).

## Learner Options
The `learner` argument controls which quantile regression model is fitted at each node.

| Option | Description |
|---|---|
| `'linear'` | Linear quantile regression (`statsmodels.QuantReg`). Fast and interpretable; assumes linear relationships in the DAG. |
| `'xgboost'` | Gradient boosting quantile regression (`sklearn.GradientBoostingRegressor` with pinball loss). Flexible for nonlinear relationships; slower. |
| `'neural_network'` | PyTorch feed-forward network (64→32→1) trained per quantile with the pinball loss. Smooth and nonlinear; requires `torch`. |
| `dict` | Per-node control, e.g. `{'Charge_Degree': 'xgboost', 'Two_Year_Recid': 'neural_network'}`. Unlisted nodes default to `'linear'`. |

All learners share the same prediction mechanism: a uniform noise value is mapped to a concrete output value by interpolating across the fitted quantile grid.

## Tuning & Performance Tips

- **Accuracy vs. speed:** increase `num_samples` and `n_runs` for tighter estimates; standard errors shrink with more runs.
- **Learner choice:** `'linear'` is the fastest option and works well when relationships in the DAG are approximately linear. Use `'xgboost'` or `'neural_network'` when you expect nonlinear relationships; note that `'neural_network'` requires `torch` and is the slowest of the three.
- **Quantile grid density:** a denser grid gives a finer approximation of the conditional distribution at the cost of fitting more models.
- **Reproducibility:** all randomness derives from `base_seed`, so identical inputs produce identical results.

### Tuning XGBoost

The XGBoost learner uses **5-fold cross-validation** to automatically select the optimal `n_estimators` and `max_depth` for each node. This ensures the parameter choice is data-driven and independent of the final explainability scores.

For each parameter combination, the data is split into 5 folds. The model is trained on 4 folds and evaluated on the remaining fold, repeated 5 times. The validation loss is computed as the average pinball loss across five quantile levels (q = 0.1, 0.3, 0.5, 0.7, 0.9), giving a balanced evaluation across the full distribution. The parameter combination with the lowest average validation loss is selected, and the final model is fitted on the full dataset using those parameters.

The default search grid covers commonly used values in the gradient boosting literature:

```python
param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth':    [2, 3, 4],
}
```

To use a custom search grid, pass it directly to `fit_with_cv` in `learners.py`:

```python
m = QuantileDAGModel_XGBoost()
m.fit_with_cv(X, y, cv=5, param_grid={
    'n_estimators': [50, 100, 200],
    'max_depth':    [2, 3, 4, 5],
})
```

- **`cv`**: number of cross-validation folds. Default is `5`.
- **`n_estimators`**: number of trees per quantile level. More trees give a better fit but increase runtime.
- **`max_depth`**: maximum depth of each tree. Deeper trees capture more complex relationships but risk overfitting. If the best parameters fall on the boundary of the search grid, consider expanding the grid.

### Tuning Neural Network

The neural network learner trains one feed-forward network (64→32→1) per quantile level using the **pinball loss**, an asymmetric loss function that penalizes under- and over-prediction differently depending on the quantile level. For quantile *q*, the loss is:
```
L(q) = q × (y - ŷ)      if y ≥ ŷ  (underestimate)
L(q) = (q-1) × (y - ŷ)  if y < ŷ  (overestimate)
```

This asymmetry forces the network to predict the correct quantile: at q = 0.9, underestimation is penalized 9× more than overestimation, so the network learns to predict the 90th percentile.

Training uses **Early Stopping** to automatically determine the optimal number of epochs. The data is split into a training set (80%) and a validation set (20%). After each epoch, the model is evaluated on the validation set. If the validation loss does not improve for `patience` consecutive epochs, training stops and the weights from the best epoch are restored.

- **`val_ratio`**: fraction of data held out as a validation set. Default is `0.2` (20%), following standard practice.
- **`patience`**: number of epochs without improvement before stopping. Default is `50`, selected by inspecting the validation loss curve on the COMPAS dataset, which showed that the loss stabilized within 50 epochs of reaching its minimum. Increase this if the model stops too early; decrease it to speed up training.

Additional parameters:

- **Network architecture**: the default network is input → 64 → 32 → 1. To change it, modify `_build_network` in `learners.py`.
- **Learning rate**: the default Adam optimizer learning rate is `1e-3`.
- **Quantile grid**: the neural network uses `quantiles_xgb` for its quantile grid. Pass `quantiles_xgb=np.arange(0.05, 1.0, 0.05)` for faster runs.


## API Reference
```python
results, mean, inter = causal_anova(dag, data, ...)
```

| Parameter | Default | Description |
|---|---|---|
| `dag` | — | DAG specification dictionary (see above) |
| `data` | — | pandas DataFrame with all node columns |
| `learner` | `'linear'` | Learner name (`'linear'`, `'xgboost'`, `'neural_network'`) or per-node dict |
| `compute_total` | `True` | Compute total explainability scores for all subsets of nodes |
| `compute_pairwise` | `True` | Compute pairwise (and three-way) interaction terms |
| `num_samples` | `10000` | Monte Carlo samples per CRN run |
| `n_runs` | `8` | Independent repetitions used for averaging and standard errors |
| `plot_venn` | `True` | Automatically plot a Venn diagram when there are 2–3 nodes in `roots` |
| `quantiles_linear` | `arange(0.05, 1.0, 0.05)` | Quantile grid for linear learner (19 levels) |
| `quantiles_xgb` | `arange(0.01, 1.0, 0.02)` | Quantile grid for xgboost / neural network learners (50 levels) |
| `base_seed` | `0` | Base random seed; run *r* uses seed `base_seed + r` for reproducibility |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `results` | `pandas.DataFrame` | Formatted table with columns `Variables`, `Type` (`Total` / `Interaction`), `Explainability (xi)`, and `SE` |
| `mean` | `dict{frozenset → float}` | Raw total explainability score for every subset of nodes in `roots` |
| `inter` | `dict{frozenset → float}` | Raw interaction terms from inclusion-exclusion |

A results table is also printed to the console, with the Total and Interaction sections separated.

## Interpreting the Output
- **Total explainability ξ(S)** for a subset *S* estimates the share of outcome variability attributable to the nodes in *S*. Values near 0 indicate that the subset barely influences the outcome; values near 1 indicate that it accounts for nearly all of the variation.
- **Interaction terms** capture synergy: how much a group of nodes explains *beyond* what their marginal scores add up to. These are non-negative in expectation, though small negative values can occur from estimation noise (they are clamped to 0 for Venn plotting).
- **The Venn diagram** visualizes the decomposition: each region shows the corresponding total or interaction score, rounded to four decimal places.

## Example

See `examples/example_compas.py` for a full walkthrough using the real COMPAS recidivism dataset, analyzing how much of two-year recidivism is causally explained by sex, race, and age category, with prior record count and charge degree as intermediate nodes in the causal graph.
