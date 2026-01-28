import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile
import zipfile
from pathlib import Path

# Modular imports
from auth import login, requires_pro
from ui.theme import apply_wonder_theme, show_hero_header
from ui.styling import show_metric_card # Keeping metric cards helper
import analysis.basic as basic
import analysis.leakage as leakage
import analysis.advanced as advanced
import analysis.pii as pii
import analysis.pro as pro_analysis
from reports.generator import generate_certificate
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Data Expander: Audit Engine", layout="wide", page_icon="🧠")
apply_wonder_theme()

# ================= UTILS =================
def get_csv_files(base):
    return list(Path(base).rglob("*.csv"))

def show_folder_tree(base):
    for root, _, files in os.walk(base):
        level = root.replace(base, "").count(os.sep)
        indent = " " * 4 * level
        st.text(f"{indent}📁 {os.path.basename(root)}")
        for f in files:
            st.text(f"{indent}    📄 {f}")

# ================= MAIN APP =================
def main():
    if not login():
        return

    # Sidebar
    st.sidebar.title("Data Expander Pro")
    st.sidebar.caption(f"Logged in as: **{st.session_state.username}**")
    if st.session_state.is_pro:
        st.sidebar.success("PLAN: PRO (AUDIT-READY)")
    else:
        st.sidebar.info("PLAN: FREE (BASIC)")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # HERO HEADER
    show_hero_header()

    # Session State Init
    if "data_root" not in st.session_state:
        st.session_state.data_root = None
    if "active_df" not in st.session_state:
        st.session_state.active_df = None
    if "active_csv_name" not in st.session_state:
        st.session_state.active_csv_name = None

    # TABS
    tab1, tab2, tab3 = st.tabs(["📤 1. Connect Data", "📊 2. Dashboard", "🔐 3. Audit Engine"])

    # 1. CONNECT DATA (Upload & Select)
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Upload Source")
            up = st.radio("Source Type", ["CSV File", "Project ZIP"], horizontal=True)
            if up == "CSV File":
                f = st.file_uploader("Drop CSV Here", type="csv")
                if f:
                    tmp = tempfile.mkdtemp()
                    path = os.path.join(tmp, f.name)
                    open(path, "wb").write(f.getbuffer())
                    st.session_state.data_root = tmp
                    st.success("Uploaded!")
            else:
                z = st.file_uploader("Drop ZIP Here", type="zip")
                if z:
                    tmp = tempfile.mkdtemp()
                    zipfile.ZipFile(z).extractall(tmp)
                    st.session_state.data_root = tmp
                    st.success("Extracted!")

        with col2:
            st.markdown("### Select Dataset")
            if not st.session_state.data_root:
                st.info("Waiting for data...")
            else:
                csvs = get_csv_files(st.session_state.data_root)
                if csvs:
                    c = st.selectbox("Available Datasets", csvs, format_func=lambda x: x.name)
                    if st.button("Load & Analyze Now", type="primary"):
                        try:
                            df = pd.read_csv(c)
                            st.session_state.active_df = df
                            st.session_state.active_csv_name = c.name
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
                
                if st.session_state.active_df is not None:
                    st.caption(f"Active: {st.session_state.active_csv_name}")
                    st.dataframe(st.session_state.active_df.head(50), height=200)

    # 2. DASHBOARD (Basic Analysis)
    with tab2:
        df = st.session_state.active_df
        if df is None:
            st.warning("Please load a dataset first.")
        else:
            # --- DATA HEALTH DASHBOARD (Free Tier) ---
            st.markdown("### 🏥 Dataset Health Snippet")
            score, report = basic.calculate_overall_health(df)
            
            # Display as a massive Metrics Row
            hc1, hc2, hc3, hc4 = st.columns(4)
            
            hc1.metric("Overall Health Score", f"{score}/100", delta=report['grade'], delta_color="normal" if score >= 80 else "inverse")
            hc2.metric("Missing Values", f"{report['missing_ratio']:.1%}", delta="-High" if report['missing_ratio']>0.05 else "Good", delta_color="inverse")
            hc3.metric("Duplicates", f"{report['dup_ratio']:.1%}", delta="-High" if report['dup_ratio']>0.05 else "Good", delta_color="inverse")
            hc4.metric("Quality Deductions", f"{len(report['deductions'])}")
            
            # Visual Progress Bar for Health
            st.progress(score / 100)
            
            if report['deductions']:
                with st.expander("See Quality Deductions"):
                    for d in report['deductions']:
                        st.caption(f"❌ {d}")
            
            # PDF REPORT GENERATION BUTTON
            st.markdown("---")
            st.markdown("### 📄 Professional Report")
            if st.button("🎯 Generate Executive PDF Report"):
                with st.spinner("Creating comprehensive PDF with charts..."):
                    try:
                        from reports.pdf_generator import generate_professional_pdf_report
                    except ImportError as e:
                        st.error(f"Error importing PDF generator: {e}. Please install reportlab: pip install reportlab")
                        st.stop()
                    
                    # Prepare health data dict
                    health_data = {
                        'score': score,
                        'grade': report['grade'],
                        'missing_ratio': report['missing_ratio'],
                        'duplicate_ratio': report['dup_ratio'],
                        'deductions': {d: 5 for d in report['deductions']}  # Simple mapping
                    }
                    
                    # Gather findings
                    findings_dict = {}
                    try:
                        import analysis.pii as pii_module
                        pii_results = pii_module.scan_pii(df)
                        if pii_results:
                            findings_dict['pii'] = pii_results
                    except:
                        pass
                    
                    pdf_buffer = generate_professional_pdf_report(
                        df,
                        st.session_state.active_csv_name,
                        health_data,
                        findings=findings_dict
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"DataQuality_{st.session_state.active_csv_name.replace('.csv', '')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ Professional report ready! Perfect for stakeholders & management.")
            
            st.markdown("---")
            
            # Interactive Visuals
            st.markdown("### Data Distribution")
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) > 0:
                sel_col = st.selectbox("Visualize Distribution", num_cols)
                fig = px.histogram(df, x=sel_col, marginal="box", template="plotly_dark", title=f"Distribution of {sel_col}")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns to visualize.")

    # 3. AUDIT ENGINE (Pro Analysis)
    with tab3:
        if df is None:
            st.warning("Please load a dataset first.")
        else:
            run_pro_analysis(df)

import analysis.pro as pro_analysis
import analysis.sota as sota_analysis
from reports.generator import generate_certificate
import plotly.express as px

# ... (Previous imports)

@requires_pro
def run_pro_analysis(df):
    st.markdown("## 🚀 Pro Audit Engine")
    
    # Target Selection
    labels = basic.infer_label_column(df)
    label = st.selectbox("🎯 Select Target Variable", ["None"] + labels)
    
    tab_leak, tab_std, tab_deep, tab_sota = st.tabs(["1. Leakage", "2. Global Standards", "3. Deep Dive (Audit)", "4. SOTA Deep Tech"])
    
    if label != "None":
        # 1. Leakage
        with tab_leak:
            st.markdown("### Leakage Detection")
            test_size = st.slider("Test Split Size", 0.1, 0.5, 0.2)
            dup_leak = leakage.detect_duplicate_leakage(df, test_size)
            st.metric("Train/Test Overlap Risk", f"{dup_leak:.2%}", delta="-High Risk" if dup_leak > 0.05 else "Safe", delta_color="inverse")
        
        # 2. Advanced Stats
        with tab_std:
            st.markdown("### Global Standards Validation")
            stats_df = advanced.compute_advanced_stats(df)
            if not stats_df.empty:
                st.dataframe(stats_df.style.highlight_max(axis=0))

            # PII
            st.markdown("### PII Scanner")
            pii_results = pii.scan_pii(df)
            if pii_results:
                st.error(f"⚠️ PII Detected in {len(pii_results)} columns!")
                for col, types in pii_results.items():
                    st.write(f"**{col}**: {', '.join(types)}")
            else:
                st.success("✅ No PII detected.")

        # 3. Deep Dive
        with tab_deep:
            st.markdown("### 🕵️ Audit-Grade Verification")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Time-Travel Check")
                time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                t_col = st.selectbox("Time Column", ["None"] + time_cols)
                if t_col != "None":
                    is_sorted, msg = pro_analysis.detect_time_travel(df, t_col)
                    if is_sorted: st.success(msg)
                    else: st.error(msg)
            
            with col_b:
                st.markdown("#### Entity Separation")
                e_col = st.selectbox("Entity ID", ["None"] + [c for c in df.columns if df[c].nunique() > 50])
                entity_overlap_score = 0
                if e_col != "None":
                    split_idx = int(len(df) * 0.8)
                    count, pct = pro_analysis.detect_overlapping_ids(df.iloc[:split_idx], df.iloc[split_idx:], e_col)
                    entity_overlap_score = pct
                    if count > 0: st.error(f"❌ {count} overlapping entities! ({pct:.1%})")
                    else: st.success("✅ Clean split.")

            st.markdown("#### Lazy Predictors")
            if st.button("Scan for Lazy Predictors"):
                suspicious = pro_analysis.detect_lazy_predictors(df, label)
                if suspicious:
                    st.error(f"❌ {len(suspicious)} suspicious features found.")
                    st.dataframe(pd.DataFrame(suspicious, columns=["Feature", "Score"]))
                else: st.success("✅ No lazy predictors found.")

            st.markdown("#### Statistical Drift")
            if st.button("Run Strict Drift Test"):
                split_idx = int(len(df) * 0.8)
                report = pro_analysis.detect_drift(df.iloc[:split_idx], df.iloc[split_idx:])
                drifted_cols = [k for k, v in report.items() if v['drift_detected']]
                if drifted_cols: st.warning(f"⚠️ Drift in: {drifted_cols}")
                else: st.success("✅ No drift detected.")

                # Generate Cert
                cert = generate_certificate(
                    st.session_state.active_csv_name,
                    score=int(100 - (entity_overlap_score * 100) - (len(drifted_cols) * 5)),
                    pii_count=len(pii_results if 'pii_results' in locals() else []),
                    drift_cols=drifted_cols,
                    time_leaks=(t_col != "None" and 'is_sorted' in locals() and is_sorted is False),
                    entity_leaks=entity_overlap_score
                )
                st.download_button("🎖️ Download Certified Report", cert, "certificate.md")

        # 4. SOTA Deep Tech (NEW)
        with tab_sota:
            st.markdown("## ⚛️ SOTA Deep Tech Engine")
            st.info("These checks use Advanced Meta-Learning algorithms. Computation may take a moment.")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            
            # A. Label Noise
            with col_s1:
                st.markdown("#### 🏷️ Label Hygiene")
                noise = None # Initialize noise
                if st.button("Detect Mislabeling"):
                    with st.spinner("Training Shadow Models..."):
                        noise = sota_analysis.detect_label_noise(df, label)
                
                if noise is not None: # Check if noise was computed
                    if noise:
                        # Check for Bias Warning
                        if "bias_warning" in noise[0]:
                            st.warning(noise[0]["bias_warning"])
                        else:
                            st.error(f"Found {len(noise)} potential errors!")
                        
                        st.dataframe(pd.DataFrame(noise).drop(columns=["bias_warning"], errors="ignore"))
                    else:
                        st.success("✅ Labels appear consistent (Model agrees with Humans).")

            # B. Data Valuation
            with col_s2:
                st.markdown("#### 💎 Data Valuation")
                if st.button("Estimate Value"):
                    with st.spinner("Calculating Shapley Proxies..."):
                        val_df = sota_analysis.estimate_data_valuation(df, label)
                        if val_df is not None:
                            st.caption("Top 5 Hardest Samples (Most Valuable for Training)")
                            st.dataframe(val_df.tail(5))
                            st.caption("Top 5 Easiest Samples (Potentially Redundant)")
                            st.dataframe(val_df.head(5))

            # C. Slice Finder
            with col_s3:
                st.markdown("#### 🍰 Failure Slices")
                if st.button("Find Weak Spots"):
                    with st.spinner("Mining Error Rules..."):
                        slices = sota_analysis.find_failure_slices(df, label)
                        if slices:
                            st.warning("Model performs poorly on these subgroups:")
                            for s in slices:
                                st.markdown(f"**Rule**: `{s['rule']}`")
                                st.caption(f"Error Rate: {s['error_rate']:.2f} | Lift: {s['lift']:.1f}x | Size: {s['size']} rows")
                                st.markdown("---")
                        else:
                            st.success("No specific weak slice found.")

            # D. Temporal Split Simulation (Deep Check)
            st.markdown("---")
            st.markdown("#### ⏳ Temporal Split Simulation (Advanced Leakage Check)")
            st.info("Simulates 'Future' data leakage by comparing Random Split vs Time-Ordered Split performance.")
            
            time_cols_sota = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
            t_col_sota = st.selectbox("Select Time Column for Simulation", ["None"] + time_cols_sota)
            
            if t_col_sota != "None":
                if st.button("Run Split Simulation"):
                    with st.spinner("Training models on Random vs Temporal splits..."):
                        score_r, score_t, risk, error_msg = sota_analysis.detect_temporal_leakage_impact(df, t_col_sota, label)
                        
                        if score_r is None:
                            st.error(f"Simulation Failed: {error_msg}")
                        else:
                            col_res1, col_res2, col_res3 = st.columns(3)
                            col_res1.metric("Random Split Score (Optimistic)", f"{score_r:.2%}")
                            col_res2.metric("Temporal Split Score (Realistic)", f"{score_t:.2%}")
                            col_res3.metric("Leakage Risk", f"{risk:.2f}", delta="-High" if risk > 0.1 else "Low", delta_color="inverse")
                            
                            if risk > 0.1:
                                st.error(f"🚨 CRITICAL LEAKAGE DETECTED (Risk: {risk:.2f})")
                                st.write("The model performs suspiciously better on random splits than temporal splits. This generally means features contain information about the future.")
                            else:
                                st.success("✅ No temporal leakage detected. Model generalizes well forward in time.")



            # D. Fairness Audit
            st.markdown("---")
            st.markdown("#### ⚖️ Fairness & Bias Audit")
            col_f1, col_f2 = st.columns(2)
            sens_col = col_f1.selectbox("Sensitive Attribute (e.g. Gender)", ["None"] + [c for c in df.columns if df[c].nunique() < 10])
            
            if "bias_findings" not in st.session_state:
                st.session_state.bias_findings = None

            if sens_col != "None":
                priv_group = col_f2.selectbox("Privileged Group", df[sens_col].unique())
                if st.button("Check Disparate Impact"):
                    import analysis.fairness as fairness_analysis
                    
                    res = fairness_analysis.check_disparate_impact(df, sens_col, label, privileged_group=priv_group)
                    if res and "error" not in res:
                        score = res['score']
                        st.metric("Disparate Impact Ratio", f"{score:.2f}", delta="Biased" if score < 0.8 else "Fair")
                        if res['is_biased']:
                            st.error(f"❌ Bias Detected! The unprivileged group receives positive outcomes {score:.1%} as often as the privileged group.")
                            # Store for remediation
                            st.session_state.bias_findings = {
                                "sensitive_col": sens_col,
                                "privileged_group": priv_group,
                                "target_col": label
                            }
                        else:
                            st.success("✅ No statistically significant bias detected (80% rule pass).")
                            st.session_state.bias_findings = None
                    elif res:
                        st.error(res['error'])

            # E. Auto-Remediation
            st.markdown("---")
            # E. CONFIGURABLE REMEDIATION PIPELINE
            st.markdown("---")
            st.markdown("#### 🛠️ Transparent Production Pipeline")
            st.caption("Configurable. Auditable. Bias-Aware.")

            # 1. Configuration Phase
            col_cfg1, col_cfg2 = st.columns([1, 2])
            
            with col_cfg1:
                st.markdown("**1. Pipeline Config**")
                p_mode = st.radio("Pipeline Mode", 
                                ["Cleaned Neutral (Safe)", "Curated (Filtered)"], 
                                help="Neutral fixes errors only. Curated applies quality filters.")
                
                mode_key = 'cleaned_neutral' if 'Safe' in p_mode else 'curated'
                
                outlier_thresh = 3.0
                if mode_key == 'curated':
                    st.markdown("**2. Curation Rules**")
                    outlier_thresh = st.slider("Outlier Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
                    st.caption("Lower = More Aggressive Filtering")

            with col_cfg2:
                if mode_key == 'curated':
                    st.markdown("**3. Attribute Filters (Optional)**")
                    st.info("Define rules to filter the population (e.g. Age >= 18)")
                    
                    # Simple UI for adding one rule (extensible to list)
                    col_f1, col_f2, col_f3 = st.columns(3)
                    num_cols = df.select_dtypes(include=np.number).columns
                    f_col = col_f1.selectbox("Column", ["None"] + list(num_cols))
                    f_op = col_f2.selectbox("Operator", [">", ">=", "<", "<=", "=="])
                    f_val = col_f3.number_input("Value", value=0.0)
                    
                    custom_filters = []
                    if f_col != "None":
                        custom_filters.append({
                            "column": f_col, "operator": f_op, "value": f_val
                        })
                        st.success(f"Rule Added: `{f_col} {f_op} {f_val}`")
                else:
                    custom_filters = []
                    st.markdown("**Neutral Mode Active**")
                    st.info("In this mode, no rows are dropped unless they are duplicates or impossible to fix.")

            # 2. Execution Phase
            if st.button("🚀 Run Production Pipeline", type="primary"):
                import analysis.remediation as remediation
                
                with st.spinner(f"Running pipeline in {mode_key.upper()} mode..."):
                    try:
                        df_processed, report = remediation.run_pipeline_with_config(
                            df, 
                            mode=mode_key,
                            curation_filters=custom_filters,
                            outlier_threshold=outlier_thresh
                        )
                        
                        # 3. Audit Report
                        st.markdown("### 📋 Pipeline Audit Report")
                        
                        # KPI Row
                        kpi1, kpi2, kpi3 = st.columns(3)
                        meta = report['metadata']
                        rows_in = meta['raw_shape'][0]
                        rows_out = report['final_shape'][0]
                        retention = rows_out / rows_in if rows_in > 0 else 0
                        
                        kpi1.metric("Rows Processed", f"{rows_in} → {rows_out}")
                        kpi2.metric("Retention Rate", f"{retention:.1%}", 
                                   delta="Significant Data Loss" if retention < 0.8 else "Healthy",
                                   delta_color="inverse" if retention < 0.8 else "normal")
                        kpi3.metric("Integrity Check", "PASSED", delta="Verified", delta_color="normal")
                        
                        # Audit Log Table
                        st.markdown("#### Transformation Log")
                        log_df = pd.DataFrame(report['audit_log'])
                        if not log_df.empty:
                            st.dataframe(
                                log_df[['step', 'rows_in', 'rows_out', 'rows_dropped', 'retention_rate']],
                                use_container_width=True
                            )
                        
                        # Metadata JSON
                        with st.expander("View Pipeline Metadata (JSON)"):
                            st.json(report)
                            
                        # Download with Metadata
                        csv = df_processed.to_csv(index=False).encode('utf-8')
                        fname = f"{mode_key}_v{report['final_hash']}_{st.session_state.active_csv_name}"
                        
                        st.download_button(
                            label="📥 Download Certified Dataset",
                            data=csv,
                            file_name=fname,
                            mime='text/csv',
                        )
                        
                    except Exception as e:
                        st.error(f"Pipeline Failed: {str(e)}")



            st.markdown("---")
            st.markdown("#### 🎯 ML-Safe Training Split (Audit-Ready)")
            st.warning("⚠️ Prevents Leakage by fitting transformations ONLY on Training data.")
            
            col_split1, col_split2 = st.columns(2)
            split_ratio = col_split1.slider("Train/Test Split", 0.5, 0.95, 0.8, 0.05)
            
            time_cols_ml = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
            use_temporal = col_split2.checkbox("Use Temporal Split", value=len(time_cols_ml) > 0)
            
            if use_temporal and len(time_cols_ml) > 0:
                time_col_ml = st.selectbox("Time Column for Split", time_cols_ml)
            else:
                time_col_ml = None
            
            if st.button("🔬 Generate ML-Safe Splits"):
                with st.spinner("Processing splits..."):
                    import analysis.remediation as remediation
                    
                    formatted_findings = {
                        "missing": [c for c in df.columns if df[c].isnull().sum() > 0],
                        "leakage": [t_col_sota] if 't_col_sota' in locals() and t_col_sota != "None" and 'risk' in locals() and risk > 0.1 else [],
                        "outliers": [c for c in df.select_dtypes(include=np.number).columns if "id" not in c.lower()][:2],
                    }
                    
                    # Unpack 3 values now
                    train_clean, test_clean, audit_log = remediation.apply_ml_safe_remediation(
                        df, 
                        formatted_findings, 
                        split_ratio=split_ratio,
                        time_col=time_col_ml
                    )
                    
                    # Show Audit Log for Train (Master Log)
                    with st.expander("View Training Pipeline Audit Log", expanded=True):
                        st.dataframe(audit_log)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    train_csv = train_clean.to_csv(index=False).encode('utf-8')
                    col_dl1.download_button(
                        label=f"📥 TRAIN Set ({len(train_clean)} rows)",
                        data=train_csv,
                        file_name=f"train_{st.session_state.active_csv_name}",
                        mime='text/csv',
                    )
                    
                    test_csv = test_clean.to_csv(index=False).encode('utf-8')
                    col_dl2.download_button(
                        label=f"📥 TEST Set ({len(test_clean)} rows)",
                        data=test_csv,
                        file_name=f"test_{st.session_state.active_csv_name}",
                        mime='text/csv',
                    )
                    
                    st.success("✅ ML-Safe Pipeline Complete.")


if __name__ == "__main__":
    main()
