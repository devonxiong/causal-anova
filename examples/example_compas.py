# example_compas.py
# =============================================================
# Example: Causal ANOVA on the COMPAS Recidivism Dataset
#
# This script demonstrates how to use the causal_anova() function
# on the real COMPAS dataset from ProPublica.
# =============================================================

import numpy as np
import pandas as pd
from causal_anova import causal_anova

# =============================================================
# Step 1: Load the real COMPAS dataset
# =============================================================
DATA_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

print("Loading COMPAS dataset...")
raw_df = pd.read_csv(DATA_URL)
print(f"Loaded {len(raw_df)} rows.")

# =============================================================
# Step 2: Preprocess
# All columns must be numeric for causal_anova() to work.
# =============================================================
df = pd.DataFrame()

# Sex: Female = 0, Male = 1
df['Sex'] = (raw_df['sex'] == 'Male').astype(int)

# Race: African-American=0, Caucasian=1, Hispanic=2, Other=3
# Asian and Native American are pooled into Other (small sample size)
race_map = {
    'African-American': 0,
    'Caucasian':        1,
    'Hispanic':         2,
    'Asian':            3,
    'Native American':  3,
    'Other':            3,
}
df['Race'] = raw_df['race'].map(race_map)

# Age Category: Less than 25=0, 25-45=1, Greater than 45=2
age_map = {
    'Less than 25':    0,
    '25 - 45':         1,
    'Greater than 45': 2,
}
df['Age_Cat'] = raw_df['age_cat'].map(age_map)

# Mediator variables
df['Priors_Count']  = raw_df['priors_count'].astype(float)
df['Charge_Degree'] = (raw_df['c_charge_degree'] == 'F').astype(int)

# Outcome
df['Two_Year_Recid'] = raw_df['two_year_recid'].astype(float)

df = df.dropna()
print(f"After preprocessing: {len(df)} rows.")

# =============================================================
# Step 3: Define the causal DAG
# =============================================================
dag = {
    'roots':     ['Sex', 'Race', 'Age_Cat'],
    'mediators': ['Priors_Count', 'Charge_Degree'],
    'outcome':   'Two_Year_Recid',
    'edges': [
        ('Sex',           'Priors_Count'),
        ('Race',          'Priors_Count'),
        ('Age_Cat',       'Priors_Count'),
        ('Sex',           'Charge_Degree'),
        ('Race',          'Charge_Degree'),
        ('Age_Cat',       'Charge_Degree'),
        ('Priors_Count',  'Charge_Degree'),
        ('Sex',           'Two_Year_Recid'),
        ('Race',          'Two_Year_Recid'),
        ('Age_Cat',       'Two_Year_Recid'),
        ('Priors_Count',  'Two_Year_Recid'),
        ('Charge_Degree', 'Two_Year_Recid'),
    ]
}

# =============================================================
# Step 4: Run Causal ANOVA with linear learner
# =============================================================
print("\nRunning Causal ANOVA on real COMPAS data (linear learner)...")

results, mean, inter = causal_anova(
    dag              = dag,
    data             = df,
    learner          = 'linear',
    compute_total    = True,
    compute_pairwise = True,
    num_samples      = 10000,
    n_runs           = 8,
    plot_venn        = True,
    base_seed        = 42,
)

# =============================================================
# Step 5: Inspect raw output
# =============================================================
print("\n--- Raw Total Explainability Scores ---")
for s, v in sorted(mean.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    print(f"  xi({', '.join(sorted(s))}) = {v:.4f}")

print("\n--- Raw Interaction Terms ---")
for s, v in sorted(inter.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    print(f"  {' ^ '.join(sorted(s))} = {v:.4f}")

# =============================================================
# Step 6: Example with mixed learners
# Use xgboost for Charge_Degree, linear for everything else
# =============================================================
print("\nRunning with mixed learners (Charge_Degree = xgboost)...")

results_mixed, _, _ = causal_anova(
    dag              = dag,
    data             = df,
    learner          = {
        'Priors_Count':   'linear',
        'Charge_Degree':  'xgboost',
        'Two_Year_Recid': 'linear',
    },
    compute_total    = True,
    compute_pairwise = True,
    num_samples      = 5000,
    n_runs           = 4,
    plot_venn        = False,
    base_seed        = 42,
)