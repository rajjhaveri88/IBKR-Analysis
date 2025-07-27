import streamlit as st
from scripts.loaders import load_uid_margin
from scripts.metrics import uid_kpis

def render():
    st.title("📋 UID Margin Requirements")

    margin_df = load_uid_margin()
    if margin_df.empty:
        st.warning("No UID margin data.")
        return

    # show per‑UID KPIs
    kpi_df = uid_kpis(margin_df)
    st.subheader("Margin KPI by UID")
    st.dataframe(kpi_df, use_container_width=True)

    # plot daily margin
    st.subheader("Daily Margin Requirement")
    chart_df = (
        margin_df
        .pivot(index="Date", columns="UID", values="Margin")
        .fillna(method="ffill")
    )
    st.line_chart(chart_df)
