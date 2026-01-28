import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import hashlib
import json
import copy

# ==========================================
# 🧱 INFRASTRUCTURE
# ==========================================

class PipelineContext:
    """Holds the state, logs, and metadata for a pipeline run."""
    def __init__(self, raw_df, config=None):
        self.raw_df = raw_df.copy()
        self.current_df = raw_df.copy()
        self.config = config or {}
        self.audit_log = []
        self.metadata = {
            "run_timestamp": datetime.now().isoformat(),
            "raw_shape": raw_df.shape,
            "raw_hash": self._generate_hash(raw_df)
        }
    
    def log_step(self, step_name, df_in, df_out, metadata=None):
        rows_start = len(df_in)
        rows_end = len(df_out)
        rows_dropped = rows_start - rows_end
        
        entry = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "rows_in": rows_start,
            "rows_out": rows_end,
            "rows_dropped": rows_dropped,
            "retention_rate": round(rows_end / rows_start, 4) if rows_start > 0 else 0,
            "metadata": metadata or {}
        }
        self.audit_log.append(entry)
        
    def _generate_hash(self, df):
        summary = f"{df.shape}-{list(df.columns)}-{df.head(1).to_string()}"
        return hashlib.md5(summary.encode()).hexdigest()[:8]
        
    def get_report(self):
        return {
            "metadata": self.metadata,
            "audit_log": self.audit_log,
            "final_shape": self.current_df.shape,
            "final_hash": self._generate_hash(self.current_df)
        }

# ==========================================
# 🧹 CLEANER (Neutral Transformations)
# ==========================================

class DataCleaner:
    """
    Responsibilities: Format, Types, Imputation.
    Invariant: Tries to preserve rows (imputation > dropping).
    """
    def __init__(self, context: PipelineContext):
        self.ctx = context
    
    def run(self):
        # 1. Type Inference
        self._step("Type Correction", self._fix_types)
        
        # 2. Impossible Values (Validation)
        self._step("Value Validation", self._fix_values)
        
        # 3. Imputation (Neutral)
        self._step("Neutral Imputation", self._impute_missing)
        
        # 4. Dedup (Technical Cleaning)
        self._step("Duplicate Removal", self._deduplicate)
        
        return self.ctx.current_df

    def _step(self, name, func):
        df_in = self.ctx.current_df.copy()
        self.ctx.current_df = func(self.ctx.current_df)
        self.ctx.log_step(name, df_in, self.ctx.current_df)

    def _fix_types(self, df):
        df = df.copy()
        # Logic from previous auto_fix_types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Date Heuristic
                    sample = df[col].dropna().astype(str).head(100)
                    if sample.str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').sum() > len(sample) * 0.5:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Numeric Heuristic
                    cleaned = df[col].astype(str).str.replace('[,$%]', '', regex=True)
                    converted = pd.to_numeric(cleaned, errors='coerce')
                    if converted.notna().sum() > len(df) * 0.8:
                        df[col] = converted
                except:
                    pass
        return df

    def _fix_values(self, df):
        df = df.copy()
        # Logic from previous apply_validation_rules
        # But instead of clipping blindly, we assume NaNs for impossible values if cleaning
        # Or we clip if instructed. Defaulting to clip for safety.
        for col in df.columns:
            col_l = col.lower()
            if pd.api.types.is_numeric_dtype(df[col]):
                if 'age' in col_l: df[col] = df[col].clip(0, 120)
                if '%' in col_l or 'rate' in col_l: df[col] = df[col].clip(0, 100)
                if any(x in col_l for x in ['price', 'salary']): df[col] = df[col].clip(lower=0)
        return df

    def _impute_missing(self, df):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and df[numeric_cols].isnull().sum().sum() > 0:
            try:
                # Use simple median for speed/robustness in 'Neutral' cleaning
                # MICE can be reserved for 'Advanced' or 'Curated' if config specifies
                if self.ctx.config.get('imputation_strategy') == 'mice':
                    imputer = IterativeImputer(random_state=42, max_iter=3)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                else:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            except:
                 df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        return df
        
    def _deduplicate(self, df):
        return df.drop_duplicates()

# ==========================================
# 🧐 CURATOR (Opinionated Filtering)
# ==========================================

class DataCurator:
    """
    Responsibilities: Filtering, Outlier Removal.
    Invariant: Explicitly removes data based on rules.
    """
    def __init__(self, context: PipelineContext):
        self.ctx = context
        
    def run(self):
        # 1. Configured Filters
        filters = self.ctx.config.get('curation_rules', {}).get('filters', [])
        if filters:
            self._step("Rule-Based Filters", lambda df: self._apply_filters(df, filters))
            
        # 2. Outlier Removal
        outlier_cfg = self.ctx.config.get('curation_rules', {}).get('outlier_removal', {})
        if outlier_cfg.get('enabled', False):
            self._step("Outlier Removal", lambda df: self._remove_outliers(df, outlier_cfg))
            
        return self.ctx.current_df

    def _step(self, name, func):
        df_in = self.ctx.current_df.copy()
        self.ctx.current_df = func(self.ctx.current_df)
        self.ctx.log_step(name, df_in, self.ctx.current_df)

    def _apply_filters(self, df, filters):
        df = df.copy()
        for f in filters:
            col = f['column']
            op = f['operator']
            val = f['value']
            
            if col in df.columns:
                if op == '>': df = df[df[col] > val]
                elif op == '>=': df = df[df[col] >= val]
                elif op == '<': df = df[df[col] < val]
                elif op == '<=': df = df[df[col] <= val]
                elif op == '==': df = df[df[col] == val]
                elif op == '!=': df = df[df[col] != val]
        return df

    def _remove_outliers(self, df, cfg):
        df = df.copy()
        method = cfg.get('method', 'z_score')
        threshold = cfg.get('threshold', 3.0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Filter out IDs usually
        target_cols = [c for c in numeric_cols if 'id' not in c.lower()]
        
        if method == 'z_score':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[target_cols]))
            # Keep rows where ALL columns are within threshold
            df = df[(z_scores < threshold).all(axis=1)]
            
        return df

# ==========================================
# 🎻 ORCHESTRATOR
# ==========================================

class ProductionPipeline:
    def __init__(self, raw_df):
        self.raw_df = raw_df
        
    def run(self, config=None):
        config = config or {}
        context = PipelineContext(self.raw_df, config)
        
        # 1. Cleaning Phase (Always Run)
        cleaner = DataCleaner(context)
        cleaner.run()
        
        # 2. Curation Phase (Conditional)
        mode = config.get('mode', 'cleaned_neutral')
        if mode == 'curated':
            curator = DataCurator(context)
            curator.run()
            
        # 3. Final Integrity Check
        self._validate_integrity(context.current_df)
        
        report = context.get_report()
        report['mode'] = mode
        
        return context.current_df, report

    def _validate_integrity(self, df):
        if df.duplicated().sum() > 0:
            raise ValueError("Pipeline Integrity Check Failed: Duplicates detected in output!")
            
def run_pipeline_with_config(df, mode='cleaned_neutral', curation_filters=None, outlier_threshold=3.0):
    """
    Frontend-friendly wrapper to run the pipeline.
    """
    config = {
        "mode": mode,
        "imputation_strategy": "mice",
        "curation_rules": {
            "filters": curation_filters or [],
            "outlier_removal": {
                "enabled": True if mode == 'curated' else False,
                "method": "z_score",
                "threshold": outlier_threshold
            }
        }
    }
    
    pipeline = ProductionPipeline(df)
    return pipeline.run(config)
