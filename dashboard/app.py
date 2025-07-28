import streamlit as st
import sys
from pathlib import Path
from importlib import import_module

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure page
st.set_page_config(
    page_title="IBKR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely and create clean navigation
st.markdown("""
<style>
/* Hide sidebar completely */
section[data-testid="stSidebar"] {
    display: none !important;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Clean header styling - Much lighter professional gradient */
.main-header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
    padding: 2rem;
    margin: -2rem -2rem 2rem -2rem;
    border-radius: 0 0 15px 15px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    text-align: center;
    border-bottom: 1px solid #ced4da;
}

.main-title {
    color: #495057;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.nav-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    margin-top: 1rem;
}

.nav-label {
    color: #6c757d;
    font-size: 1.1rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

/* Make all selectboxes shorter by default */
.stSelectbox > div > div {
    width: auto !important;
    min-width: 200px !important;
    max-width: 300px !important;
}

/* Make ONLY the page navigation selectbox full page width */
.page-navigation-container .stSelectbox > div > div {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
}

/* Main content area styling */
.main-content {
    padding: 2rem;
    background: #ffffff;
    border-radius: 15px;
    margin-top: 1rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    border: 1px solid #e9ecef;
}

/* Mobile responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.8rem;
    }
    
    .nav-container {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .nav-label {
        font-size: 1rem;
        margin-right: 0;
        margin-bottom: 0.5rem;
    }
    
    .main-content {
        padding: 1rem;
        margin-top: 0.5rem;
    }
}

/* Extra small mobile devices */
@media (max-width: 480px) {
    .main-title {
        font-size: 1.5rem;
    }
    
    .main-header {
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Professional page mapping
PAGES = {
    "📊 Overview": "overview",
    "🎯 Strategy": "strategy", 
    "👥 UID": "uid",
    "💰 Cash": "cash",
    "📈 Trades": "trades"
}

# Create header with title and navigation (removed duplicate title)
st.markdown("""
<div class="main-header">
    <div class="nav-container">
        <div class="nav-label">Select Page:</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Create dropdown navigation with Overview as default (using default Streamlit styling)
st.markdown('<div class="page-navigation-container">', unsafe_allow_html=True)
page_names = list(PAGES.keys())
selected_page = st.selectbox(
    "Select Page",
    page_names,
    index=0,  # Overview is at index 0
    key="page_navigation"
)
st.markdown('</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Load and render the selected page
page_module = PAGES[selected_page]
try:
    page_mod = import_module(f"dashboard.pages_old.{page_module}")
    page_mod.render()
except Exception as e:
    st.error(f"Error loading page {page_module}: {str(e)}")
    st.info("Please check if the page file exists and has a render() function.")

st.markdown('</div>', unsafe_allow_html=True) 