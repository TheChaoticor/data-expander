import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare

def detect_time_travel(df, time_col, target_col=None):
    """
    Checks if the dataset is sorted by time, which is crucial to prevent
    future data leaking into past training data in a random split.
    """
    if time_col not in df.columns:
        return None, "Time column not found"
    
    try:
        # Try to parse soft
        times = pd.to_datetime(df[time_col], errors='coerce').dropna()
        if len(times) < len(df) * 0.9:
            return 0.0, "Column is not a valid datetime"
            
        is_sorted = times.is_monotonic_increasing
        return is_sorted, "Dataset is strictly sorted by time" if is_sorted else "Dataset is NOT sorted by time (Risk of random split leakage)"
    except:
        return None, "Failed to parse time column"

def detect_overlapping_ids(train_df, test_df, id_col):
    """
    Checks if the same entities appear in both train and test.
    Industry std: 0% overlap allowed for entity-based splitting.
    """
    if id_col not in train_df.columns:
        return 0, 0.0
        
    train_ids = set(train_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))
    
    intersection = train_ids.intersection(test_ids)
    overlap_count = len(intersection)
    overlap_pct = overlap_count / len(test_ids) if len(test_ids) > 0 else 0.0
    
    return overlap_count, overlap_pct

def detect_lazy_predictors(df, target_col, threshold=0.95):
    """
    Identifies features that are suspiciously good at predicting the target.
    This often indicates a leak (e.g. 'outcome_desc' predicting 'outcome_status').
    """
    suspicious = []
    
    if target_col not in df.columns:
        return []

    # Simplified correlation check for speed
    # (A full decision tree would be better but heavier)
    if pd.api.types.is_numeric_dtype(df[target_col]):
        corrs = df.corr(numeric_only=True)[target_col].abs()
        for col, val in corrs.items():
            if col != target_col and val > threshold:
                suspicious.append((col, val))
    else:
        # Categorical target - use simple groupby logic or cramer's V approximation
        # Fallback: check text overlap if strings
        pass
        
    return suspicious

def detect_drift(train_df, test_df):
    """
    Checks for statistical distribution drift between train and test.
    Uses Kolmogorov-Smirnov test for numerical columns.
    """
    drift_report = {}
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stat, p_value = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        # If p_value is small (< 0.05), distributions are likely different
        drift_report[col] = {
            "statistic": stat,
            "p_value": p_value,
            "drift_detected": p_value < 0.05
        }
        
    return drift_report
