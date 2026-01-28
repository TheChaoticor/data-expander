import pandas as pd
import numpy as np

def compute_advanced_stats(df):
    """
    Computes skewness, kurtosis, and outlier counts for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
            
        # Skew & Kurtosis
        skew = series.skew()
        kurt = series.kurtosis()
        
        # Outliers (IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_pct = len(outliers) / len(series)
        
        stats[col] = {
            "skewness": skew,
            "kurtosis": kurt,
            "outlier_count": len(outliers),
            "outlier_pct": outlier_pct
        }
    
    return pd.DataFrame(stats).T
