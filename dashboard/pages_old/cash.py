import streamlit as st, pandas as pd, plotly.express as px
from scripts.loaders import load_fund_flow

def render() -> None:
    st.title("💰 Cash Movements")
    cash = load_fund_flow()[["ToDate", "Deposit/Withdrawals", "BrokerInterest"]]
    cash["NetFlow"] = cash["Deposit/Withdrawals"] + cash["BrokerInterest"]
    fig = px.bar(cash, x="ToDate", y="NetFlow", title="Cash Flows")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cash) 