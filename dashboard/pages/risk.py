import streamlit as st, plotly.express as px
from scripts.loaders import load_fund_flow, load_allocation
from scripts.cash_engine import build_cash_timeline

def render():
    st.title("⚠️ Risk Dashboard")
    tl = build_cash_timeline(load_fund_flow(), load_allocation())
    broker = tl.set_index("Date")["BrokerBalance"]
    draw = (broker / broker.cummax() - 1) * 100

    st.subheader("Rolling Draw‑down (%)")
    st.plotly_chart(px.area(draw, labels={"value": "Draw‑down %"}), use_container_width=True)

    st.subheader("Margin Headroom")
    headroom = tl["Spare"] / tl["BrokerBalance"] * 100
    st.plotly_chart(px.line(tl, x="Date", y=headroom,
                            labels={"y": "Spare / Broker (%)"}), use_container_width=True) 