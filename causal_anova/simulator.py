import numpy as np
import pandas as pd


def _build_simulator(dag, data, models):
    """
    Build a simulation function from the DAG structure and fitted models.

    The returned function takes a noise dictionary (node -> uniform array)
    and propagates values through the DAG in topological order, returning
    the simulated outcome array.

    Args:
        dag    : dict with 'roots', 'mediators', 'outcome', 'edges'
        data   : original DataFrame, used for empirical quantile mapping of root nodes
        models : dict {node_name -> fitted learner instance}
    Returns:
        simulate_fn : callable (E_dict -> np.array)
    """
    roots   = dag['roots']
    outcome = dag['outcome']
    edges   = dag['edges']

    # Build parent lookup from edge list
    all_nodes = roots + dag.get('mediators', []) + [outcome]
    parents = {nd: [] for nd in all_nodes}
    for src, tgt in edges:
        parents[tgt].append(src)

    def simulate_fn(E_dict):
        simulated = {}

        # Root nodes: map uniform noise to empirical quantiles
        for r in roots:
            simulated[r] = np.quantile(data[r], E_dict[r], method='lower')

        # Non-root nodes: propagate through DAG in topological order
        for nd in dag.get('mediators', []) + [outcome]:
            pa = parents[nd]
            X  = pd.DataFrame({p: simulated[p] for p in pa})
            simulated[nd] = models[nd].predict_from_noise(X, E_dict[nd])

        return simulated[outcome]

    return simulate_fn