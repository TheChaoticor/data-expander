import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.preprocessing import LabelEncoder

def detect_temporal_leakage_impact(df, time_col, target_col):
    """
    Industy Standard 'Split Simulation' for Time Travel.
    Compares Model Performance: Random Split (Optimistic) vs Sequential Split (Realistic).
    
    If Random Split is significantly better, it implies the model relies on 
    interpolation (future data) rather than extrapolation (past data).
    
    Returns:
        random_score, temporal_score, leakage_risk_score (0-1)
    """
    if time_col not in df.columns or target_col not in df.columns:
        return None, None, 0.0, "Missing columns."

    # Sort strictly by time first
    try:
        # Try to convert to datetime for proper temporal sorting
        df = df.copy()
        
        # Check if it's already numeric (e.g. Year), if so, skip to_datetime
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
        # Check if we lost too much data
        n_before = len(df)
        df = df.dropna(subset=[time_col]) # Drop invalid dates
        n_after = len(df)
        
        if n_after < 50:
             return None, None, 0.0, f"Column '{time_col}' could not be parsed as a date (or had too many missing values)."
             
        df = df.sort_values(time_col)
    except Exception as e:
        # Fallback to string/numeric sort if parsing fails
        try:
            df = df.sort_values(time_col)
        except:
            return None, None, 0.0, f"Could not sort by '{time_col}': {str(e)}"
        
    # Reset index to ensure iloc works expectedly if needed (though we use iloc which is position based)
    # But for safety in feature extraction:
    df = df.reset_index(drop=True)
    
    # Prepare X, y
    # CRITICAL FIX: Do not drop categoricals. Leakage often hides in strings!
    df_clean = df.drop(columns=[target_col, time_col]).copy()
    
    # Handle Categoricals (Simple Label Encoding for RF robustness)
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        # Fill NaNs first to avoid issues
        df_clean[col] = df_clean[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
    
    # Fill numeric NaNs
    df_clean = df_clean.fillna(0)
    
    X = df_clean
    y = df[target_col]
    
    if len(df) < 100:
         return None, None, 0.0, "Dataset too small (<100 rows) for split simulation."
         
    if X.shape[1] < 1:
        return None, None, 0.0, "No features found to train on."
        
    is_classifier = not pd.api.types.is_numeric_dtype(y)
    
    if is_classifier:
        # Encode Target if needed
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
    
    # Model Factory
    def get_model():
        if is_classifier:
            return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            
    from sklearn.metrics import f1_score
    
    def get_score(y_true, y_pred):
        if is_classifier:
            # Use F1-Weighted to handle class imbalance better than Accuracy
            # (Leakage often looks like finding the minority class perfectly)
            return f1_score(y_true, y_pred, average='weighted')
        else:
            return r2_score(y_true, y_pred)

    # 1. TEMPORAL SPLIT (The "Honest" Test)
    # Train on first 80%, Test on last 20%
    split_idx = int(len(df) * 0.8)
    X_train_t, X_test_t = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_t, y_test_t = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model_t = get_model()
    model_t.fit(X_train_t, y_train_t)
    score_t = get_score(y_test_t, model_t.predict(X_test_t))
    
    # 2. RANDOM SPLIT (The "Leaky" Test)
    # Randomly shuffle
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_r = get_model()
    model_r.fit(X_train_r, y_train_r)
    score_r = get_score(y_test_r, model_r.predict(X_test_r))
    
    # 3. COMPARE
    # If Random Score is much higher, risk is high.
    # Clip scores to 0-1 range for calculation safety
    s_r = max(0, min(1, score_r))
    s_t = max(0, min(1, score_t))
    
    diff = s_r - s_t
    risk = max(0.0, diff / max(0.1, s_r)) # Normalized relative risk
    
    diff = s_r - s_t
    risk = max(0.0, diff / max(0.1, s_r)) # Normalized relative risk
    
    return score_r, score_t, risk, None

def detect_label_noise(df, target_col, cv=5):
    """
    Uses Confident Learning principles (Northcutt et al.) to find mislabels.
    We trust the model more than the label if confidence is high.
    """
    if target_col not in df.columns:
        return []

    # Drop NaNs
    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) < 50: return []

    # Prepare data (Simple encoding for speed)
    X = df_clean.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df_clean[target_col]
    
    # Needs enough numeric features
    if X.shape[1] < 1:
        return []

    try:
        # Use RF for robustness
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        
        if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 20:
             # Treat continuous variables as regression tasks - simple residual check
             # (Not true CL, but helps find outliers)
             pass 
        else:
            # Classification
            # Encode Y if string
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            
            probs = cross_val_predict(model, X, y_enc, cv=min(cv, len(X)//5), method='predict_proba')
            
            suspects = []
            for i, (actual_code, prob_dist) in enumerate(zip(y_enc, probs)):
                conf_actual = prob_dist[actual_code]
                pred_code = np.argmax(prob_dist)
                conf_pred = prob_dist[pred_code]
                
                # Heuristic: If model is 90% confident it's B, but label is A, and A has < 10% conf.
                if conf_pred > 0.90 and conf_actual < 0.10:
                    suspects.append({
                        "row_id": df_clean.index[i],
                        "actual_label": le.inverse_transform([actual_code])[0],
                        "predicted_label": le.inverse_transform([pred_code])[0],
                        "confidence": conf_pred
                    })
            
            # --- BIAS CHECK ---
            # If >80% of errors are the SAME swap (e.g. Female -> Male), it's likely Model Bias, not Data Noise.
            if len(suspects) > 10:
                error_types = [f"{s['actual_label']} -> {s['predicted_label']}" for s in suspects]
                most_common = pd.Series(error_types).mode()[0]
                count = error_types.count(most_common)
                if count / len(suspects) > 0.8:
                    # Return a special flag in the first element to warn the UI
                    suspects[0]["bias_warning"] = (
                        f"⚠️ Systematic Bias Detected: {count}/{len(suspects)} errors are '{most_common}'. "
                        "The model may be overfitting to the majority class or lacking predictive features. "
                        "These are likely NOT label errors, but model limitations."
                    )
            
            return suspects
    except Exception as e:
        return [] # Fail gracefully
    return []

def estimate_data_valuation(df, target_col):
    """
    Heuristic Data Valuation:
    Hard-to-predict samples (high error) are often 'valuable' (edge cases),
    or 'outliers' (noise). Easy-to-predict are low value.
    """
    if target_col not in df.columns: return None
    df_clean = df.dropna().copy()
    
    X = df_clean.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df_clean[target_col]
    
    if X.shape[1] < 1 or len(X) < 20: return None
    
    # Train shadow model
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 20:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, oob_score=True)
        model.fit(X, y)
        preds = model.oob_prediction_
        if preds is None: return None
        errors = np.abs(y - preds)
    else:
        # Check for strings
        if not pd.api.types.is_numeric_dtype(y):
             le = LabelEncoder()
             y = le.fit_transform(y)
             
        model = RandomForestClassifier(n_estimators=50, max_depth=10, oob_score=True)
        model.fit(X, y)
        if not hasattr(model, "oob_decision_function_"): return None
        
        # Loss as proxy for difficulty (1 - prob of correct class)
        oob_probs = model.oob_decision_function_
        # Handle case where oob might rely on classes
        errors = []
        for i, true_class in enumerate(y):
            # If true_class index is in bounds
            if true_class < oob_probs.shape[1]:
                errors.append(1 - oob_probs[i, true_class])
            else:
                errors.append(1.0)
        errors = np.array(errors)
        
    results = pd.DataFrame({"row_id": df_clean.index, "difficulty_score": errors})
    return results.sort_values("difficulty_score", ascending=True)

def find_failure_slices(df, target_col):
    """
    Meta-Learning for Error Analysis.
    Trains a 'Error Predictor' (Decision Tree) vs Error Residuals.
    Returns: Interpretable Rules describing weak spots (e.g. "Age <= 30 & Income <= 50k").
    """
    if target_col not in df.columns: return []
    df_clean = df.dropna().copy()
    
    # Use simple label encoding for interpretability (OneHot is too sparse for simple text rules here)
    X_orig = df_clean.drop(columns=[target_col])
    X = X_orig.select_dtypes(include=[np.number])
    
    # Add encoded categoricals if they have low criminality (readable rules)
    cat_cols = X_orig.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    for c in cat_cols:
        if X_orig[c].nunique() < 20:
            le = LabelEncoder()
            X[c] = le.fit_transform(X_orig[c].astype(str))
            encoders[c] = le
            
    y = df_clean[target_col]
    
    if X.shape[1] < 1: return []
    
    # 1. Calculate Errors (Residuals)
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 20:
        model = RandomForestRegressor(n_estimators=20, max_depth=5)
        model.fit(X, y)
        errors = np.abs(y - model.predict(X))
    else:
        # For Classification, error is 1 if wrong, 0 if right
        if not pd.api.types.is_numeric_dtype(y):
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
        model = RandomForestClassifier(n_estimators=20, max_depth=5)
        model.fit(X, y)
        preds = model.predict(X)
        errors = (y != preds).astype(int)
        
    # 2. Train Rule Extractor (Decision Tree Regressor on the ERROR)
    # simple depth to keep rules readable (max 3 predicates)
    rule_miner = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(5, int(len(df)*0.05)))
    rule_miner.fit(X, errors)
    
    # 3. Extract Rules from Tree
    slices = []
    
    
    # Helper to traverse tree
    
    def recurse(node, rule_path):
        if rule_miner.tree_.children_left[node] == _tree.TREE_LEAF:
            # We found a leaf. Check if it's a "Failure Slice" (High Error)
            # Value[node] is the predicted error rate in this leaf
            avg_error = rule_miner.tree_.value[node][0][0]
            overall_mean = errors.mean()
            
            # Lift > 1.3 means this slice has 30% more error than average
            if avg_error > max(0.01, overall_mean * 1.3):
                # Clean up rule string
                clean_rules = " AND ".join(rule_path) if rule_path else "Global"
                
                slices.append({
                    "rule": clean_rules,
                    "error_rate": float(avg_error),
                    "lift": float(avg_error / max(1e-6, overall_mean)),
                    "size": int(rule_miner.tree_.n_node_samples[node])
                })
        else:
            # Continue recursion
            name = X.columns[rule_miner.tree_.feature[node]]
            threshold = rule_miner.tree_.threshold[node]
            
            # Decode threshold if categorical
            if name in encoders:
                # Find closest integer
                val = int(threshold)
                # It's a bit tricky with LabelEncoding <= check, but we interpret roughly
                # Ideally OneHot is better for exactness, but this is a heuristic scan
                txt_val = encoders[name].inverse_transform([min(len(encoders[name].classes_)-1, max(0, val))])[0]
                rule_txt_left = f"{name} <= {txt_val}"
                rule_txt_right = f"{name} > {txt_val}"
            else:
                rule_txt_left = f"{name} <= {threshold:.2f}"
                rule_txt_right = f"{name} > {threshold:.2f}"
                
            recurse(rule_miner.tree_.children_left[node], rule_path + [rule_txt_left])
            recurse(rule_miner.tree_.children_right[node], rule_path + [rule_txt_right])
            
    recurse(0, [])
            
    return sorted(slices, key=lambda x: x['lift'], reverse=True)[:3]

