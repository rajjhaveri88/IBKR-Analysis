import streamlit as st
import sys
from pathlib import Path
from importlib import import_module

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Options Analytics", layout="wide")

PAGES = {
    "Overview":  "overview",
    "Strategy":  "strategy",
    "UID Runs":  "uid",
    "Trades":    "trades",
    "Cash Flow": "cash",
}
choice = st.sidebar.radio("📊 Navigation", list(PAGES.keys()))
page_mod = import_module(f"dashboard.pages.{PAGES[choice]}")
page_mod.render() 