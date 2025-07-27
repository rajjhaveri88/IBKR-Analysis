# dashboard/pages/overview.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from scripts.loaders     import load_fund_flow, load_allocation, load_nav, load_trades
from scripts.cash_engine import build_cash_timeline
from scripts.metrics     import account_kpis

def render():
    st.title("📈 Account Overview")

    # ─ build the cash timeline
    fundflow = load_fund_flow()
    alloc    = load_allocation()
    tl       = build_cash_timeline(fundflow, alloc)
    kpi      = account_kpis(fundflow, tl)

    # ─ prepare the K‑chart data (in thousands)
    tl_k = tl.copy()
    num_cols = tl_k.select_dtypes("number").columns
    tl_k[num_cols] = tl_k[num_cols] / 1_000

    # ─ identify allocation groups
    alloc_cols = [c for c in tl_k.columns if c.startswith("Allocation_")]
    ts_cols    = [c for c in alloc_cols if "_TS" in c]
    ut_cols    = [c for c in alloc_cols if "_UT" in c]
    ss_cols    = [c for c in alloc_cols if "_SS" in c]

    # ─ build Plotly timeline figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tl_k["Date"], y=tl_k["BrokerBalance"],
        name="Broker Balance", mode="lines",
        line=dict(color="#0E4B9D", width=2), legendrank=1
    ))
    fig.add_trace(go.Scatter(
        x=tl_k["Date"], y=tl_k["Spare"],
        name="Spare Capital", mode="lines",
        line=dict(color="rgba(96,96,96,0.85)", width=1.5, dash="dash"), legendrank=2
    ))
    fig.add_trace(go.Scatter(
        x=tl_k["Date"], y=tl_k["TotalAllocation"],
        name="Total Allocation", mode="lines",
        line=dict(color="rgba(128,128,128,0.9)", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(160,160,160,0.25)", legendrank=3
    ))
    for cols, color, fill, rank in (
        (ts_cols, "rgba(255,160,160,0.9)", "rgba(255,160,160,0.35)", 4),
        (ut_cols, "rgba(173,216,230,0.9)", "rgba(173,216,230,0.35)", 5),
        (ss_cols, "rgba(70,130,180,0.9)",  "rgba(70,130,180,0.35)", 6),
    ):
        for col in cols:
            fig.add_trace(go.Scatter(
                x=tl_k["Date"], y=tl_k[col],
                name=col.replace("Allocation_", ""),
                mode="lines",
                line=dict(width=0.5, color=color),
                fill="tozeroy", fillcolor=fill,
                legendrank=rank,
            ))
    fig.update_traces(hovertemplate="%{fullData.name} — %{y:,.2f} K<extra></extra>")
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Capital (× 1 000)",
        legend_title="Legend",
        margin=dict(t=40, r=20, b=40, l=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─ KPI row 1: Spare Today, Total Allocated, Broker Balance
    c1, c2, c3 = st.columns(3)
    c1.metric("Spare Today",     f"{tl['Spare'].iloc[-1]:,.0f}")
    c2.metric("Total Allocated", f"{tl['TotalAllocation'].iloc[-1]:,.0f}")
    c3.metric("Broker Balance",  f"{tl['BrokerBalance'].iloc[-1]:,.0f}")

    # ─ KPI row 2: Net PnL, Interest, CAGR
    trades = load_trades()
    net_pnl_trades = trades["NetCash"].sum()
    interest       = fundflow.get("BrokerInterest", pd.Series(0)).sum()
    c4, c5, c6 = st.columns(3)
    c4.metric("Net PnL", f"{net_pnl_trades:,.2f}")
    c5.metric("Interest", f"{interest:,.2f}")
    c6.metric("CAGR", f"{kpi['CAGR']*100:,.2f} %" if pd.notna(kpi["CAGR"]) else "N/A")

    # ─ KPI row 3: Max DD, Sharpe (Trades), Profit Factor (Trades)
    # compute trade-based Sharpe & Profit Factor
    trades["TradeDate"]   = pd.to_datetime(trades["TradeDate"], dayfirst=True)
    daily_tr_pnl          = trades.groupby("TradeDate")["NetCash"].sum()
    # risk-free daily rate
    rf_annual = 0.045
    rf_daily  = rf_annual / 252
    # excess PnL over risk-free
    excess_pnl = daily_tr_pnl - rf_daily
    mean_excess = excess_pnl.mean()
    std_pnl     = daily_tr_pnl.std(ddof=0) or 1e-9
    sharpe_tr   = mean_excess / std_pnl * np.sqrt(252)
    gross_win   = daily_tr_pnl[daily_tr_pnl > 0].sum()
    gross_loss  = -daily_tr_pnl[daily_tr_pnl < 0].sum()
    pf_tr       = gross_win / gross_loss if gross_loss else np.nan

    c7, c8, c9 = st.columns(3)
    c7.metric("Max DD", f"{kpi['MaxDD_pct']:,.2f} %" if pd.notna(kpi["MaxDD_pct"]) else "N/A")
    c8.metric("Sharpe (Trades)", f"{sharpe_tr:.2f}")
    c9.metric("Profit Factor (Trades)", f"{pf_tr:.2f}")

    # ─ KPI row 4: Win / Loss Days from trades
    win_days  = int((daily_tr_pnl > 0).sum())
    loss_days = int((daily_tr_pnl < 0).sum())
    st.metric("Win / Loss Days", f"{win_days} / {loss_days}")

    # ─ NAV curve (optional)
    nav = load_nav()
    if not nav.empty:
        st.markdown("### NAV Curve")
        st.line_chart(nav.set_index("Date")[["NAV"]])
