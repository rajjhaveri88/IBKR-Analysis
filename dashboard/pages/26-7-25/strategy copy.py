import streamlit as st, plotly.express as px
from scripts.loaders import load_trades, load_allocation, load_fund_flow
from scripts.cash_engine import build_cash_timeline
from scripts.metrics import strategy_kpis

def render():
    st.title("📊 Strategy Analytics")
    tl = build_cash_timeline(load_fund_flow(), load_allocation())
    kpi_df = strategy_kpis(load_trades(), tl)

    st.subheader("KPI Table")
    st.dataframe(kpi_df.set_index("Strategy"))

    st.subheader("Cumulative P&L")
    fig = px.line(kpi_df, x="Strategy", y="NetPnL", markers=True)
    st.plotly_chart(fig, use_container_width=True) 