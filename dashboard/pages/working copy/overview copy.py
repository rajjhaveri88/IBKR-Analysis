# dashboard/pages/overview.py
import streamlit as st, plotly.graph_objects as go
from scripts.loaders import load_fund_flow, load_allocation
from scripts.cash_engine import build_cash_timeline
from scripts.metrics import account_kpis

def render():
    st.title("📈 Account Overview")
    tl = build_cash_timeline(load_fund_flow(), load_allocation())
    kpi = account_kpis(load_fund_flow(), tl)

    # ── Build timeline & scale numeric columns to thousands (K)
    tl_k = tl.copy()
    num_cols = tl_k.select_dtypes("number").columns
    tl_k[num_cols] = tl_k[num_cols] / 1_000

    # Strategy allocation columns
    alloc_cols = [c for c in tl_k.columns if c.startswith("Allocation_")]
    ts_cols = [c for c in alloc_cols if c.startswith("Allocation_TS")]
    ut_cols = [c for c in alloc_cols if c.startswith("Allocation_UT")]
    ss_cols = [c for c in alloc_cols if c.startswith("Allocation_SS")]

    fig = go.Figure()

    # 1️⃣  Broker Balance (legendrank 1)
    fig.add_trace(
        go.Scatter(
            x=tl_k["Date"],
            y=tl_k["BrokerBalance"],
            name="Broker Balance",
            mode="lines",
            line=dict(color="#0E4B9D", width=2),
            legendrank=1,
        )
    )

    # 2️⃣  Spare Capital (legendrank 2)
    fig.add_trace(
        go.Scatter(
            x=tl_k["Date"],
            y=tl_k["Spare"],
            name="Spare Capital",
            mode="lines",
            line=dict(color="rgba(96,96,96,0.85)", width=1.5, dash="dash"),
            legendrank=2,
        )
    )

    # 3️⃣  Total Allocation area (legendrank 3)
    fig.add_trace(
        go.Scatter(
            x=tl_k["Date"],
            y=tl_k["TotalAllocation"],
            name="Total Allocation",
            mode="lines",
            line=dict(color="rgba(128,128,128,0.9)", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(160,160,160,0.25)",
            legendrank=3,
        )
    )

    # 4️⃣  TS_* areas  (legendrank 4)
    for col in ts_cols:
        fig.add_trace(
            go.Scatter(
                x=tl_k["Date"],
                y=tl_k[col],
                name=col.replace("Allocation_", ""),
                mode="lines",
                line=dict(width=0.5, color="rgba(255,160,160,0.9)"),
                fill="tozeroy",
                fillcolor="rgba(255,160,160,0.35)",
                legendrank=4,
            )
        )

    # 5️⃣  UT_* areas  (legendrank 5)
    for col in ut_cols:
        fig.add_trace(
            go.Scatter(
                x=tl_k["Date"],
                y=tl_k[col],
                name=col.replace("Allocation_", ""),
                mode="lines",
                line=dict(width=0.5, color="rgba(173,216,230,0.9)"),
                fill="tozeroy",
                fillcolor="rgba(173,216,230,0.35)",
                legendrank=5,
            )
        )

    # 6️⃣  SS_* areas  (legendrank 6)
    for col in ss_cols:
        fig.add_trace(
            go.Scatter(
                x=tl_k["Date"],
                y=tl_k[col],
                name=col.replace("Allocation_", ""),
                mode="lines",
                line=dict(width=0.5, color="rgba(70,130,180,0.9)"),
                fill="tozeroy",
                fillcolor="rgba(70,130,180,0.35)",
                legendrank=6,
            )
        )

    # Unified hover template (two decimals, thousands, K suffix)
    fig.update_traces(
        hovertemplate="%{fullData.name} — %{y:,.2f} K<extra></extra>"
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Capital (× 1 000)",
        legend_title="Legend",
        margin=dict(t=40, r=20, b=40, l=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # KPI cards (values already in K)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Spare Today", f"{tl['Spare'].iloc[-1]:,.0f}")
    kpi2.metric("Total Allocated", f"{tl['TotalAllocation'].iloc[-1]:,.0f}")
    kpi3.metric("Broker Balance", f"{tl['BrokerBalance'].iloc[-1]:,.0f}")

    c4, c5 = st.columns(2)
    c4.metric("XIRR", f"{kpi['XIRR']*100:,.2f}	%" if kpi['XIRR'] is not None else "N/A")
    c5.metric("Max DD", f"{kpi['MaxDD_pct']:,.2f}	%" if kpi['MaxDD_pct'] is not None else "N/A")
