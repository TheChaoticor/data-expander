import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Main container polish */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #555;
        }
        
        /* Dark mode overrides (naive) */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stMetric"] {
                background-color: #262730;
                color: white;
            }
            [data-testid="stMetricLabel"] {
                color: #ddd;
            }
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

def show_metric_card(label, value, delta=None):
    st.metric(label, value, delta)
