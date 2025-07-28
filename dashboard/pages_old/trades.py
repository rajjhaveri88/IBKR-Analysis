import streamlit as st, pandas as pd
from st_aggrid import AgGrid
from scripts.loaders import load_trades

def render() -> None:
    st.title("🧾 Trades Ledger")
    df = load_trades()
    
    # Add download button
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades Ledger as CSV",
            data=csv,
            file_name="trades_ledger.csv",
            mime="text/csv",
            help="Download the complete trades ledger as a CSV file"
        )
        st.write("")  # Add some spacing
    
    AgGrid(df, height=500, theme="streamlit") 