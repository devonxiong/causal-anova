import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def _format_results(mean, se, inter, roots, compute_total, compute_pairwise):
    """
    Format explainability results into a clean pandas DataFrame.

    Args:
        mean             : dict of total explainability scores
        inter            : dict of interaction terms
        roots            : list of root node names
        compute_total    : bool, whether to include total scores
        compute_pairwise : bool, whether to include interaction terms
    Returns:
        df_out : formatted DataFrame with columns Variables, Type, Explainability (xi)
    """
    rows = []
    if compute_total:
        for s in sorted(mean, key=lambda x: (len(x), sorted(x))):
            label = ' , '.join(sorted(s))
            rows.append({
                'Variables':           label,
                'Type':                'Total',
                'Explainability (xi)': f"{mean[s]:.4f}",
                'SE':                  f"{se[s]:.4f}",
            })
    if compute_pairwise:
        for s in sorted(inter, key=lambda x: (len(x), sorted(x))):
            label = ' ^ '.join(sorted(s))
            rows.append({
                'Variables':           label,
                'Type':                'Interaction',
                'Explainability (xi)': f"{inter[s]:.4f}",
                'SE':                  '—',
            })
    return pd.DataFrame(rows)


def _print_results(df_out):
    """
    Pretty-print the results table with Total and Interaction sections separated.

    Args:
        df_out : DataFrame from _format_results
    """
    print("\n" + "=" * 60)
    print("CAUSAL ANOVA RESULTS")
    print("=" * 60)
    total_block = df_out[df_out['Type'] == 'Total']
    inter_block = df_out[df_out['Type'] == 'Interaction']
    if not total_block.empty:
        print("\n-- Total Explainability --")
        print(total_block[['Variables', 'Explainability (xi)', 'SE']].to_string(index=False))
    if not inter_block.empty:
        print("\n-- Interaction Terms (inclusion-exclusion) --")
        print(inter_block[['Variables', 'Explainability (xi)']].to_string(index=False))
    print("=" * 60)


def _plot_venn(mean, inter, roots):
    """
    Plot a Venn diagram of explainability scores for 2 or 3 root nodes.
    Skipped automatically if len(roots) < 2 or len(roots) > 3.

    Args:
        mean  : dict of total explainability scores
        inter : dict of interaction terms
        roots : list of root node names
    """
    fs = frozenset

    def _round_labels(v):
        """Round all subset labels to 4 decimal places."""
        for text in v.subset_labels:
            if text is not None:
                try:
                    text.set_text(f"{float(text.get_text()):.4f}")
                except ValueError:
                    pass

    if len(roots) < 2:
        print("Venn diagram skipped: need at least 2 root nodes.")
        return

    if len(roots) == 2:
        a, b = roots
        xa  = max(mean[fs({a})], 0)
        xb  = max(mean[fs({b})], 0)
        xab = max(inter[fs({a, b})], 0) 

        fig, ax = plt.subplots(figsize=(7, 5))
        v = venn2(subsets=(round(xa, 4), round(xb, 4), round(xab, 4)),
                set_labels=(a, b), ax=ax)
        _round_labels(v)
        ax.set_title('Explainability Venn Diagram', fontsize=14)
        plt.tight_layout()
        plt.show()
    

    elif len(roots) == 3:
        a, b, c = roots
        xa, xb, xc    = mean[fs({a})], mean[fs({b})], mean[fs({c})]
        xab, xac, xbc = inter[fs({a, b})], inter[fs({a, c})], inter[fs({b, c})]
        xabc          = inter[fs({a, b, c})]
        fig, ax = plt.subplots(figsize=(8, 6))
        v = venn3(subsets=(round(xa, 4), round(xb, 4), round(xab, 4),
                           round(xc, 4), round(xac, 4), round(xbc, 4),
                           round(xabc, 4)),
                  set_labels=(a, b, c), ax=ax)
        _round_labels(v)
        ax.set_title('Explainability Venn Diagram', fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        print(f"Venn diagram skipped: {len(roots)} root nodes exceeds the limit of 3.")
