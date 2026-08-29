# example_compas.py
# =============================================================
# Example: Causal ANOVA on the COMPAS Recidivism Dataset
#
# This script demonstrates how to use the causal_anova() function
# on the real COMPAS dataset from ProPublica. It walks through:
#   1. Loading and preprocessing the real COMPAS data
#   2. Defining the causal DAG structure
#   3. Running causal_anova() with the linear learner
#   4. Interpreting the output
#   5. Inspecting raw scores
#   6. Running with mixed learners per node
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
# Step 2: Preprocess the data
#
# causal_anova() requires all columns to be numeric.
# We encode the categorical variables as integers.
# =============================================================
df = pd.DataFrame()

# Sex: Female = 0, Male = 1
df['Sex'] = (raw_df['sex'] == 'Male').astype(int)

# Race: encode as integer categories
# African-American = 0, Caucasian = 1, Hispanic = 2, Other = 3
# (Asian and Native American are pooled into Other due to small sample size)
race_map = {
    'African-American': 0,
    'Caucasian':        1,
    'Hispanic':         2,
    'Asian':            3,
    'Native American':  3,
    'Other':            3,
}
df['Race'] = raw_df['race'].map(race_map)

# Age Category: Less than 25 = 0, 25-45 = 1, Greater than 45 = 2
age_map = {
    'Less than 25':    0,
    '25 - 45':         1,
    'Greater than 45': 2,
}
df['Age_Cat'] = raw_df['age_cat'].map(age_map)

# Mediator variables (already numeric in the raw data)
df['Priors_Count']  = raw_df['priors_count'].astype(float)
df['Charge_Degree'] = (raw_df['c_charge_degree'] == 'F').astype(int)

# Outcome variable
df['Two_Year_Recid'] = raw_df['two_year_recid'].astype(float)

# Drop any rows with missing values
df = df.dropna()
print(f"After preprocessing: {len(df)} rows.")

# =============================================================
# Step 3: Define the causal DAG
#
# The DAG encodes the following causal assumptions:
#
#   Sex, Race, Age_Cat (root nodes / demographics)
#       --> Priors_Count  (number of prior offenses, mediator)
#       --> Charge_Degree (felony vs misdemeanor, mediator)
#       --> Two_Year_Recid (two-year recidivism, outcome)
#
# Priors_Count also directly influences Charge_Degree.
# All root nodes have direct paths to the outcome as well.
# =============================================================
dag = {
    'roots':     ['Sex', 'Race', 'Age_Cat'],
    'mediators': ['Priors_Count', 'Charge_Degree'],
    'outcome':   'Two_Year_Recid',
    'edges': [
        # Root nodes -> Priors_Count
        ('Sex',          'Priors_Count'),
        ('Race',         'Priors_Count'),
        ('Age_Cat',      'Priors_Count'),
        # Root nodes -> Charge_Degree
        ('Sex',          'Charge_Degree'),
        ('Race',         'Charge_Degree'),
        ('Age_Cat',      'Charge_Degree'),
        # Priors_Count -> Charge_Degree
        ('Priors_Count', 'Charge_Degree'),
        # All nodes -> Outcome
        ('Sex',          'Two_Year_Recid'),
        ('Race',         'Two_Year_Recid'),
        ('Age_Cat',      'Two_Year_Recid'),
        ('Priors_Count', 'Two_Year_Recid'),
        ('Charge_Degree','Two_Year_Recid'),
    ]
}

# =============================================================
# Step 4: Run Causal ANOVA with linear learner (fastest)
# Change learner='xgboost' or 'neural_network' for more flexibility
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
# Step 5: Inspect the raw output
# =============================================================
print("\n--- Raw Total Explainability Scores ---")
for s, v in sorted(mean.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    label = ', '.join(sorted(s))
    print(f"  xi({label}) = {v:.4f}")

print("\n--- Raw Interaction Terms ---")
for s, v in sorted(inter.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
    label = ' ^ '.join(sorted(s))
    print(f"  {label} = {v:.4f}")

# =============================================================
# Step 6: Example — use different learners for different nodes
#
# If you suspect a nonlinear relationship for a particular node,
# you can override the learner for that node only.
# All other nodes will still use 'linear' by default.
# =============================================================
print("\nRunning Causal ANOVA with mixed learners (Charge_Degree uses xgboost)...")

results_mixed, _, _ = causal_anova(
    dag    = dag,
    data   = df,
    learner = {
        'Priors_Count':   'linear',
        'Charge_Degree':  'xgboost',
        'Two_Year_Recid': 'linear',
    },
    num_samples = 5000,   # reduced for demonstration; use 10000 for full analysis
    n_runs      = 4,      # reduced for demonstration; use 8 for full analysis
    plot_venn   = False,
    base_seed   = 42,
)
