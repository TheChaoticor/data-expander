from datetime import datetime

def generate_certificate(dataset_name, score, pii_count, drift_cols, time_leaks, entity_leaks):
    """
    Generates a markdown certificate for the dataset.
    """
    grade = "A" if score > 90 else "B" if score > 75 else "C" if score > 50 else "F"
    color = "green" if grade == "A" else "orange" if grade == "B" else "red"
    
    return f"""
# 🎖️ Data Quality Certificate

**Dataset:** `{dataset_name}`  
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Certifier:** Data Expander Pro Engine

---

## 🏆 Final Grade: {grade} (Score: {score}/100)

### 🛡️ Privacy & Security
- **PII Detected:** {pii_count} columns {"✅" if pii_count == 0 else "❌"}

### 🕵️ Anti-Leakage Audit
- **Time-Travel Leakage:** {"Pass ✅" if not time_leaks else "Fail ❌"}
- **Entity Overlap:** {entity_leaks:.1%} {"✅" if entity_leaks == 0 else "❌"}

### 📉 Statistical Integrity
- **Drifting Features:** {len(drift_cols)} found
- **Drift details:** {", ".join(drift_cols) if drift_cols else "None"}

---
*This document certifies the readiness of the dataset for Machine Learning workflows according to industry standards.*
    """.strip()
