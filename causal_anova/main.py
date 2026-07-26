import numpy as np
from .learners import _LEARNER_REGISTRY, QuantileDAGModel_Linear, QuantileDAGModel_XGBoost, QuantileDAGModel_NeuralNetwork
from .crn        import _crn_explainability, _interaction_terms
from .simulator  import _build_simulator
from .output     import _format_results, _print_results, _plot_venn


def causal_anova(
    dag,
    data,
    learner          = 'linear',
    compute_total    = True,
    compute_pairwise = True,
    num_samples      = 10000,
    n_runs           = 8,
    plot_venn        = True,
    quantiles_linear = np.arange(0.05, 1.0, 0.05),
    quantiles_xgb    = np.arange(0.01, 1.0, 0.02),
    base_seed        = 0,
):
    """
    Run Causal ANOVA explainability analysis on a user-specified DAG.

    Args:
        dag : dict with keys:
            'roots'     : list of root node names (variables to explain)
            'mediators' : list of mediator node names (can be [])
            'outcome'   : string, outcome node name
            'edges'     : list of (src, tgt) tuples defining causal structure

        data : pandas DataFrame containing all node columns (must be numeric)

        learner : str ('linear' or 'xgboost') applied to all non-root nodes,
                  or dict {node_name: learner_name} for per-node control.
                  Nodes not listed in the dict default to 'linear'.
                  - 'linear'  : linear quantile regression (fast, interpretable)
                  - 'xgboost' : gradient boosting (flexible, slower)

        compute_total    : bool, compute total explainability scores (default True)
        compute_pairwise : bool, compute pairwise/three-way interactions (default True)
        num_samples      : Monte Carlo samples per CRN run (default 10000)
        n_runs           : number of independent runs for averaging (default 8)
        plot_venn        : bool, auto-plot Venn Diagram if roots <= 3 (default True)
        quantiles_linear : quantile grid for linear learner
        quantiles_xgb    : quantile grid for xgboost learner
        base_seed        : base random seed for reproducibility (default 0)

    Returns:
        results : pandas DataFrame, formatted explainability output
        mean    : dict {frozenset -> float}, raw total explainability scores
        inter   : dict {frozenset -> float}, raw interaction terms
    """
    roots     = dag['roots']
    mediators = dag.get('mediators', [])
    outcome   = dag['outcome']
    edges     = dag['edges']
    all_nodes = roots + mediators + [outcome]

    # ---- Resolve learner per node ----
    if isinstance(learner, str):
        learner_map = {nd: learner for nd in mediators + [outcome]}
    elif isinstance(learner, dict):
        learner_map = {nd: learner.get(nd, 'linear') for nd in mediators + [outcome]}
    else:
        raise ValueError("`learner` must be a string ('linear'/'xgboost') "
                         "or a dict mapping node name -> learner name.")

    for nd, ln in learner_map.items():
        if ln not in _LEARNER_REGISTRY:
            raise ValueError(f"Unknown learner '{ln}' for node '{nd}'. "
                             f"Choose from {list(_LEARNER_REGISTRY)}.")

    # ---- Build parent lookup ----
    parents = {nd: [] for nd in all_nodes}
    for src, tgt in edges:
        parents[tgt].append(src)

    # ---- Fit models for all non-root nodes ----
    models = {}
    for nd in mediators + [outcome]:
        pa = parents[nd]
        if not pa:
            raise ValueError(f"Node '{nd}' has no parents but is not listed as a root.")

        X  = data[[p for p in pa]]
        y  = data[nd]
        ln = learner_map[nd]

        print(f"\nFitting model for node: {nd}  (parents: {pa}, learner: {ln})")

        if ln == 'linear':
            m = QuantileDAGModel_Linear(quantiles=quantiles_linear)
        elif ln == 'xgboost':
            m = QuantileDAGModel_XGBoost(quantiles=quantiles_xgb)
        elif ln == 'neural_network':
            m = QuantileDAGModel_NeuralNetwork(quantiles=quantiles_xgb)
        else:
            raise ValueError(f"Unknown learner '{ln}'.")
        m.fit(X, y)
        models[nd] = m

    # ---- Build simulator ----
    simulate_fn = _build_simulator(dag, data, models)

    # ---- Run CRN explainability ----
    print(f"\nRunning CRN explainability ({n_runs} runs x {num_samples} samples)...")
    mean, se = _crn_explainability(
        simulate_fn, all_nodes, roots,
        num_samples=num_samples, n_runs=n_runs, base_seed=base_seed
    )

    # ---- Compute interaction terms ----
    inter = _interaction_terms(mean, roots) if compute_pairwise else {}

    # ---- Format and print output ----
    results = _format_results(mean, se, inter, roots, compute_total, compute_pairwise)
    _print_results(results)

    # ---- Venn Diagram ----
    if plot_venn and 2 <= len(roots) <= 3:
        print("\nGenerating Venn Diagram...")
        _plot_venn(mean, inter if inter else _interaction_terms(mean, roots), roots)

    return results, mean, inter
