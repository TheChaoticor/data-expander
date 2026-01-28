import streamlit as st

def apply_wonder_theme():
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* GLOBAL RESET & VARIABLES */
        :root {
            --primary: #00f2ff;
            --secondary: #bd00ff;
            --bg-dark: #0e1117;
            --glass: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* AURORA BACKGROUND */
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(189, 0, 255, 0.15) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(0, 242, 255, 0.15) 0%, transparent 40%),
                        #0e1117;
        }

        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0e1117; 
        }
        ::-webkit-scrollbar-thumb {
            background: #333; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555; 
        }

        /* GLASSMORPHISM CARDS (Metrics) */
        div[data-testid="stMetric"] {
            background: var(--glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 242, 255, 0.1);
            border-color: var(--primary);
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
            font-weight: 300;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(45deg, var(--primary), white);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* GLOWING TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: rgba(255,255,255,0.03);
            border-radius: 10px;
            border: 1px solid transparent;
            color: #ccc;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255,255,255,0.08);
            border-color: rgba(255,255,255,0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(189,0,255,0.1), rgba(0,242,255,0.1)) !important;
            border: 1px solid var(--primary) !important;
            color: white !important;
            box-shadow: 0 0 15px rgba(0,242,255,0.2);
        }

        /* BUTTONS */
        .stButton button {
            background: linear-gradient(45deg, #1f2937, #111827);
            border: 1px solid #374151;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(45deg, var(--secondary), var(--primary));
            border-color: transparent;
            box-shadow: 0 0 15px rgba(189,0,255,0.4);
            transform: scale(1.02);
            color: black;
        }

        /* TEXT INPUTS / SELECTBOXES */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 8px !important;
        }
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 10px rgba(0,242,255,0.2) !important;
        }

        /* MARKDOWN HEADERS */
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        h1 {
            background: linear-gradient(90deg, white, #ccc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2 {
            color: white;
            border-left: 4px solid var(--secondary);
            padding-left: 15px;
        }

        /* ALERTS */
        [data-testid="stNotification"] {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def show_hero_header():
    st.markdown("""
        <div style="text-align: center; padding: 40px 0; margin-bottom: 20px;">
            <h1 style="font-size: 3.5rem; margin-bottom: 10px; background: linear-gradient(to right, #00f2ff, #bd00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Data Expander
            </h1>
            <p style="font-size: 1.2rem; color: #aaa;">
                The Ultimate AI-Powered Dataset Audit & Validation Engine
            </p>
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px;">
                <span style="padding: 5px 15px; background: rgba(0,242,255,0.1); border: 1px solid #00f2ff; border-radius: 20px; color: #00f2ff; font-size: 0.8rem;">
                    🚀 Enterprise Grade
                </span>
                <span style="padding: 5px 15px; background: rgba(189,0,255,0.1); border: 1px solid #bd00ff; border-radius: 20px; color: #bd00ff; font-size: 0.8rem;">
                    🔐 Zero Leakage
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
