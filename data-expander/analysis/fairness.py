import pandas as pd
import numpy as np

def check_disparate_impact(df, sensitive_col, target_col, privileged_group=None, positive_outcome=None):
    """
    Calculates Disparate Impact Ratio (DIR).
    DIR = P(Positive | Unprivileged) / P(Positive | Privileged)
    
    Industry standard: DIR < 0.8 is "Adverse Impact".
    """
    if sensitive_col not in df.columns or target_col not in df.columns:
        return None
        
    df = df.dropna(subset=[sensitive_col, target_col])
    
    # Auto-infer groups if not provided
    if privileged_group is None:
        # Assume majority group is privileged
        privileged_group = df[sensitive_col].mode()[0]
        
    if positive_outcome is None:
        # Assume 1 or "Yes" or True is positive
        vals = df[target_col].unique()
        if 1 in vals: positive_outcome = 1
        elif "Yes" in vals: positive_outcome = "Yes"
        elif "yes" in vals: positive_outcome = "yes"
        else: positive_outcome = vals[0] # Fallback
        
    # Calculate Probabilities
    priv_mask = df[sensitive_col] == privileged_group
    unpriv_mask = df[sensitive_col] != privileged_group
    
    if unpriv_mask.sum() == 0 or priv_mask.sum() == 0:
        return {
            "error": "Not enough data in one of the groups."
        }
        
    # Rate of positive outcome
    priv_rate = df[priv_mask][target_col].eq(positive_outcome).mean()
    unpriv_rate = df[unpriv_mask][target_col].eq(positive_outcome).mean()
    
    if priv_rate == 0:
        dir_score = 0.0 # Avoid div by zero
    else:
        dir_score = unpriv_rate / priv_rate
        
    return {
        "score": dir_score,
        "privileged_group": privileged_group,
        "unprivileged_group": "Others",
        "priv_rate": priv_rate,
        "unpriv_rate": unpriv_rate,
        "is_biased": dir_score < 0.8
    }
