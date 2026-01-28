import pandas as pd
import numpy as np

def missing_value_score(df):
    ratio = df.isnull().sum().sum() / df.size
    return max(0, 1 - ratio), ratio

def type_consistency_score(df):
    bad = sum(df[c].dropna().map(type).nunique() > 1 for c in df.columns)
    return max(0, 1 - bad / len(df.columns)), bad

def duplicate_score(df):
    ratio = df.duplicated().sum() / len(df)
    return max(0, 1 - ratio), ratio

def dataset_size_score(df):
    n = len(df)
    return 0.3 if n < 100 else 0.6 if n < 1000 else 1.0

def infer_label_column(df):
    return [c for c in df.columns if df[c].nunique() <= 20 and not pd.api.types.is_float_dtype(df[c])]

def label_imbalance_score(df, label):
    dist = df[label].value_counts(normalize=True)
    maxr = dist.max()
    score = 1.0 if maxr <= 0.7 else 0.6 if maxr <= 0.8 else 0.4 if maxr <= 0.9 else 0.2
    return score, maxr, dist

def calculate_overall_health(df):
    """
    Calculates a 0-100 'Data Health Score' based on:
    - Completeness (Missing Values)
    - Uniqueness (Duplicates)
    - Structure (Column Naming)
    - Cardinality/Variance issues
    
    Returns: score (int), report (dict)
    """
    score = 100
    report = {"deductions": []}
    
    # 1. Completeness (-30 max)
    missing_ratio = df.isnull().sum().sum() / max(1, df.size)
    if missing_ratio > 0:
        deduction = min(30, int(missing_ratio * 100 * 2)) # roughly 1% missing = -2 points
        score -= deduction
        report["deductions"].append(f"Missing Values: -{deduction} pts")
        
    # 2. Uniqueness (-20 max)
    dup_ratio = df.duplicated().sum() / max(1, len(df))
    if dup_ratio > 0:
        deduction = min(20, int(dup_ratio * 100 * 3)) # 1% dups = -3 points
        score -= deduction
        report["deductions"].append(f"Duplicates: -{deduction} pts")
        
    # 3. Structural Health (-10 max)
    # Check for spaces/special chars in column names (bad for SQL/Code)
    bad_cols = [c for c in df.columns if " " in c or any(x in c for x in "(),!@#$%^")]
    if bad_cols:
        deduction = min(10, len(bad_cols) * 2)
        score -= deduction
        report["deductions"].append(f"Messy Column Names: -{deduction} pts")
        
    # 4. Content Risk (-20 max)
    # E.g. Single Value Columns (Zero Variance)
    single_val_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if single_val_cols:
        deduction = 10
        score -= deduction
        report["deductions"].append(f"Zero-Variance Columns: -{deduction} pts")
        
    # 5. Type Consistency (Implicit in basic.py checks usually, adding simplified here)
    # (Skipping deep check for speed, focus on metadata)
    
    score = max(0, score)
    
    # Grading
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 60 else "D" if score >= 40 else "F"
    
    report["grade"] = grade
    report["missing_ratio"] = missing_ratio
    report["dup_ratio"] = dup_ratio
    
    return score, report
