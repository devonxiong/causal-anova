import numpy as np
from itertools import combinations


def _crn_explainability(simulate_fn, nodes, roots,
                        num_samples=10000, n_runs=8, base_seed=0):
    """
    Estimate CRN explainability scores for all subsets of root nodes.

    Uses the Pick-Freeze estimator with Common Random Numbers (CRN):
    within each repetition, the same noise draws E and E' are shared
    across all subsets S, ensuring non-negative interaction terms.

    Args:
        simulate_fn  : callable (noise dict -> output array)
        nodes        : list of all node names in the DAG
        roots        : list of root node names to explain
        num_samples  : Monte Carlo samples per run
        n_runs       : number of independent runs for averaging
        base_seed    : starting random seed for reproducibility
    Returns:
        mean : dict {frozenset -> float}, mean explainability score per subset
        se   : dict {frozenset -> float}, standard error across runs
    """
    subsets = [frozenset(c) for k in range(1, len(roots) + 1)
               for c in combinations(roots, k)]
    agg = {s: [] for s in subsets}

    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        E   = {nd: rng.uniform(0, 1, num_samples) for nd in nodes}
        Ep  = {nd: rng.uniform(0, 1, num_samples) for nd in nodes}
        Yf, Yp = simulate_fn(E), simulate_fn(Ep)
        den = np.var(Yf - Yp)
        for s in subsets:
            Ecf = {nd: (Ep[nd] if nd in s else E[nd]) for nd in nodes}
            agg[s].append(0.0 if den == 0 else
                          np.var(Yf - simulate_fn(Ecf)) / den)

    mean = {s: float(np.mean(v)) for s, v in agg.items()}
    se   = {s: float(np.std(v)  / np.sqrt(n_runs)) for s, v in agg.items()}
    return mean, se


def _interaction_terms(mean, roots):
    """
    Compute pairwise and three-way interaction terms via inclusion-exclusion.

    For two variables A and B:
        xi(A ^ B) = xi(A) + xi(B) - xi(A, B)
    A positive value means A and B jointly explain more than their
    marginal scores alone would predict.

    Args:
        mean  : dict of total explainability scores from _crn_explainability
        roots : list of root node names
    Returns:
        out : dict {frozenset -> float}
    """
    fs  = frozenset
    out = {}
    for a, b in combinations(roots, 2):
        out[fs({a, b})] = mean[fs({a})] + mean[fs({b})] - mean[fs({a, b})]
    if len(roots) == 3:
        a, b, c = roots
        out[fs({a, b, c})] = (
            mean[fs({a})] + mean[fs({b})] + mean[fs({c})]
            - mean[fs({a, b})] - mean[fs({a, c})] - mean[fs({b, c})]
            + mean[fs({a, b, c})]
        )
    return out