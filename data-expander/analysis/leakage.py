import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

@st.cache_data(ttl=600)
def detect_duplicate_leakage(df, test_size):
    df = df.sample(frac=1, random_state=42)
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return len(set(train.apply(tuple, axis=1)) & set(test.apply(tuple, axis=1))) / max(1, len(test))

@st.cache_data(ttl=600)
def detect_entity_leakage(df, col, test_size):
    df = df.sample(frac=1, random_state=42)
    split = int(len(df) * (1 - test_size))
    return len(set(df.iloc[:split][col]) & set(df.iloc[split:][col])) / max(1, df[col].nunique())

@st.cache_data(ttl=600)
def compute_feature_leakage(df, label):
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder().fit_transform(df[label].astype(str))
    scores = {}
    for c in df.columns:
        if c == label:
            continue
        try:
            scores[c] = abs(pd.Series(enc).corr(df[c])) if pd.api.types.is_numeric_dtype(df[c]) else df.groupby(c)[label].value_counts(normalize=True).groupby(level=0).max().mean()
        except:
            scores[c] = 0.0
    return pd.DataFrame.from_dict(scores, orient="index", columns=["leak_score"]).sort_values("leak_score", ascending=False)

def fuzzy_check(df, max_rows=2000):
    if len(df) > max_rows:
        return None, "Skipped (dataset too large)"
    cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])][:3]
    if not cols:
        return None, "No text columns"
    texts = df[cols].astype(str).agg(" ".join, axis=1).tolist()[:500]
    hits = sum(SequenceMatcher(None, texts[i], texts[i+1]).ratio() > 0.95 for i in range(len(texts)-1))
    return hits / max(1, len(texts)), "Sampled fuzzy check"

def validate_target(df, label):
    reasons = []
    score = 1.0

    if label is None:
        return 0.0, ["No target selected"]

    lname = label.lower()

    if any(k in lname for k in ["id", "name", "email", "student"]):
        reasons.append("Looks like an identifier, not a prediction target")
        score -= 0.6

    if lname in ["gender", "sex", "age", "city", "state"]:
        reasons.append("Demographic attribute, unlikely to be ML target")
        score -= 0.6

    nunique = df[label].nunique()
    if nunique <= 1:
        reasons.append("Target has only one class")
        score -= 0.7

    if nunique > 50:
        reasons.append("Very high cardinality for classification target")
        score -= 0.3

    if not pd.api.types.is_numeric_dtype(df[label]) and nunique <= 20:
        score += 0.2  # typical classification target

    return max(0.0, min(1.0, score)), reasons
