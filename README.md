# causal-anova

A Python package for Causal ANOVA explainability analysis on user-specified DAGs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

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

results, mean, inter = causal_anova(
    dag=dag, data=df, learner='linear',
    compute_total=True, compute_pairwise=True,
    num_samples=10000, n_runs=8, plot_venn=True,
)
```

## Learner Options

| Option | Description |
|---|---|
| `'linear'` | Linear quantile regression (fast, interpretable) |
| `'xgboost'` | Gradient boosting (flexible, slower) |
| `dict` | Per-node control, e.g. `{'Charge_Degree': 'xgboost'}` |

## Example

See `examples/example_compas.py` for a full walkthrough using the real COMPAS dataset.