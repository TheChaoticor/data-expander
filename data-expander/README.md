# 🦅 Data Expander Pro
### *The Ultimate Data Audit & Remediation Platform*

> **"Don't just clean your data. Certify it."**

Data Expander Pro is an enterprise-grade SaaS platform designed to audit, clean, and certify datasets for high-stakes Machine Learning applications. It moves beyond simple "null value checking" to perform State-of-the-Art (SOTA) analysis, leakage detection, and bias auditing.

---

## 🚀 Key Capabilities

### 1. 🏥 Deep Health Audit
*   **0-100 Health Score**: Comprehensive grading based on completeness, uniqueness, and consistency.
*   **Quality Deductions**: Pinpoints exact reasons for score penalties (e.g., "-5 for High Duplicates").
*   **Statistical Profiling**: Automated skewness, kurtosis, and correlation checks.

### 2. 🔐 Advanced Leakage Detection
*   **Time-Travel Check**: Detects if your data is sorted in a way that leaks future information.
*   **Entity Overlap**: Ensures subject independence between Train/Test splits (critical for medical/financial data).
*   **Lazy Predictors**: Identifies features that are "too good to be true" (proxies for the target).

### 3. ⚛️ SOTA Deep Tech Engine
*   **Label Noise Detection**: Uses Confidence Learning to identify mislabeled training examples.
*   **Data Valuation**: Estimates the Shapley value of rows to find the most impactful training samples.
*   **Failure Slice Discovery**: Automatically finds subgroups where your model will likely fail (e.g., "Males under 25").
*   **Temporal Split Simulation**: Simulates time-based validation to prove forward-generalization.

### 4. 🛠️ Transparent Production Pipeline
*   **Dual-Mode Remediation**:
    *   **⚪ Cleaned Neutral (Safe)**: Non-destructive fixing (Types, Imputation) Only. Guaranteed retention.
    *   **🔵 Curated (Filtered)**: Opinionated outlier removal and attribute filtering.
*   **Audit Log**: Tracks every single row drop with a named rule reason.
*   **Validation Gates**: Automatically blocks output if statistical drift is detected.
*   **ML-Safe Splits**: Performs fit/transform on Train/Test separately to guarantee Zero Leakage.

### 5. 📄 Executive Reporting
*   Generates a **Professional PDF Certification** with:
    *   Executive Summary & Grade.
    *   Visual Quality Compliance Charts.
    *   Prioritized Action Plan.

---

## 🏗️ Architecture

The system is built on a modular "Audit & Fix" architecture:

```mermaid
graph TD
    User[User Upload] --> Validator{Schema Check}
    Validator --> Dashboard[Health Dashboard]
    
    Dashboard --> DeepDive[Deep Dive Engine]
    DeepDive --> SOTA[SOTA Analysis]
    DeepDive --> Leakage[Leakage Scanner]
    DeepDive --> PII[PII Scanner]
    
    Dashboard --> Remediation[Remediation Pipeline]
    Remediation --> Cleaner[DataCleaner (Neutral)]
    Cleaner --> Curator[DataCurator (Opinionated)]
    Curator --> Auditor[Audit Logger]
    
    Auditor --> Export[Certified Dataset]
    Auditor --> Report[PDF Certificate]
```

## 💻 Tech Stack
*   **Frontend**: Streamlit (with Custom "Wonder UI" Theme)
*   **Backend Logic**: Python 3.10+
*   **Data Engine**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn (RandomForest, IterativeImputer), Cleanlab (Label Noise)
*   **Visualization**: Plotly Express, Seaborn, Matplotlib
*   **Reporting**: ReportLab (PDF Generation)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/data-expander.git

# Navigate to directory
cd data-expander

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🎮 Usage Guide

### Tab 1: 📤 Connect Data
Upload your CSV or ZIP file. The system instantly ingests and profiles the data.

### Tab 2: 📊 Dashboard
View the high-level **Health Score**.
*   Click **"Generate Executive PDF Report"** to download the official audit certificate.

### Tab 3: 🔐 Audit Engine (The Pro Features)
*   **Leakage**: Check "Train/Test Overlap Risk".
*   **SOTA Deep Tech**: Run "Label Noise" or "Data Valuation".
*   **Production Pipeline**:
    *   Select **"Curated"** mode.
    *   Set **"Outlier Threshold"**.
    *   Click **"Run Production Pipeline"** to get a cleaned, versioned dataset.

---

## 🛡️ License
Proprietary - For Internal Use Only.
*Built by Data Expander Inc.*
