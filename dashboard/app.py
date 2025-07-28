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

# Force sidebar to always be expanded
st.markdown("""
<style>
/* Force sidebar to always be expanded */
section[data-testid="stSidebar"] {
    min-width: 21rem !important;
    max-width: 21rem !important;
    width: 21rem !important;
    overflow: visible !important;
}

/* Hide Streamlit's default collapse button completely */
button[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

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

/* Sidebar behavior when collapsed */
section[data-testid="stSidebar"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

/* When sidebar is collapsed, hide it completely */
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px !important;
    max-width: 0px !important;
    width: 0px !important;
    overflow: hidden !important;
}

/* Style the sidebar toggle button */
button[key="sidebar_toggle"] {
    position: fixed !important;
    left: 0px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 9999 !important;
    background: #f0f2f6 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 8px 4px !important;
    box-shadow: 2px 0 4px rgba(0,0,0,0.1) !important;
    cursor: pointer !important;
    font-size: 16px !important;
    color: #333 !important;
    min-width: auto !important;
    width: auto !important;
}

button[key="sidebar_toggle"]:hover {
    background: #e0e0e0 !important;
}

/* Force sidebar to always be expanded and hide default toggle */
section[data-testid="stSidebar"] {
    min-width: 21rem !important;
    max-width: 21rem !important;
    width: 21rem !important;
    overflow: visible !important;
    aria-expanded: true !important;
}

/* Hide Streamlit's default collapse button */
button[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Make main content wider when sidebar is collapsed */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* When sidebar is collapsed, expand main content */
section[data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
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