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
    initial_sidebar_state="expanded"
)

# Custom CSS for professional navigation
st.markdown("""
<style>
/* Professional Navigation Styling */
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Prevent sidebar collapse and ensure it's always visible */
section[data-testid="stSidebar"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

/* Hide the sidebar collapse button */
button[data-testid="collapsedControl"] {
    display: none !important;
}

/* Custom navigation styling */
.nav-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.nav-title {
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

/* Style buttons as clickable headers */
.nav-button {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: white;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
    text-align: center;
    width: 100%;
}

.nav-button:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.nav-button.selected {
    background: rgba(255, 255, 255, 0.25);
    border-color: rgba(255, 255, 255, 0.4);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Professional page mapping - only main pages
PAGES = {
    "📊 Overview": "overview",
    "🎯 Strategy": "strategy", 
    "👥 UID": "uid",
    "💰 Cash": "cash",
    "📈 Trades": "trades"
}

# Create professional navigation header
st.sidebar.markdown("""
<div class="nav-container">
    <div class="nav-title">IBKR Analytics</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Overview"

# Create custom navigation buttons
page_names = list(PAGES.keys())
for page_name in page_names:
    is_selected = st.session_state.current_page == page_name
    button_style = "selected" if is_selected else ""
    
    if st.sidebar.button(
        page_name, 
        key=f"nav_{page_name}",
        help=f"Navigate to {page_name}",
        use_container_width=True
    ):
        st.session_state.current_page = page_name

# Load and render selected page
selected_page_file = PAGES[st.session_state.current_page]
page_mod = import_module(f"dashboard.pages_old.{selected_page_file}")
page_mod.render() 