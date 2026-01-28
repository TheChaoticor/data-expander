import pandas as pd
import re

PII_PATTERNS = {
    "Email": r"[^@]+@[^@]+\.[^@]+",
    "Phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "Credit Card": r"\b(?:\d[ -]*?){13,16}\b",
    "IP Address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "SSN (US)": r"\b\d{3}-\d{2}-\d{4}\b"
}

def scan_pii(df, sample_size=500):
    """
    Scans the dataframe for potential PII.
    Returns a dictionary of {column: [detected_pii_types]}
    """
    results = {}
    
    # Process string columns only
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    
    sample = df[str_cols].sample(min(len(df), sample_size), random_state=42).astype(str)
    
    for col in str_cols:
        col_findings = set()
        for pii_name, pattern in PII_PATTERNS.items():
            # Check if any value in the sample matches the pattern
            # Using simple regex search on joined string for speed, or element-wise
            
            # Element-wise check (more accurate count but slower)
            matches = sample[col].astype(str).str.contains(pattern, regex=True).sum()
            if matches > 0:
                col_findings.add(pii_name)
        
        if col_findings:
            results[col] = list(col_findings)
            
    return results
