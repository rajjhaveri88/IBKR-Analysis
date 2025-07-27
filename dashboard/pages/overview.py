# dashboard/pages/overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scripts.loaders import load_timeline, load_nav, load_trades

def render():
    st.title("📈 Account Overview")

    # ─── Load data ─────────────────────────────────────────────
    tl     = load_timeline()   # has Date, BrokerBalance & Allocation_* cols
    nav_df = load_nav()        # has Date, NAV
    trades = load_trades()     # your legs.parquet with TradeDate, NetCash, etc.

    if tl.empty or nav_df.empty or trades.empty:
        st.warning("Not enough data to render Overview.")
        return

    # ─── Prepare timeline & compute allocations/spare ──────────
    tl = tl.copy()
    tl["Date"] = pd.to_datetime(tl["Date"], dayfirst=True)
    alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
    
    # Allocation persistence is now handled in ETL, no need for additional forward-fill here
    
    tl["TotalAllocation"] = tl[alloc_cols].sum(axis=1)
    tl["SpareCapital"]    = tl["BrokerBalance"] - tl["TotalAllocation"]

    # ─── Account graph ─────────────────────────────────────────
    fig = go.Figure()

    # 1) strategy stacks (fill areas)
    for col in alloc_cols:
        fig.add_trace(go.Scatter(
            x=tl["Date"], y=tl[col],
            mode="none",
            stackgroup="one",
            name=col.replace("Allocation_","")
        ))

    # 2) total allocation as a grey line on top
    fig.add_trace(go.Scatter(
        x=tl["Date"], y=tl["TotalAllocation"],
        mode="lines",
        name="Total Allocation",
        line=dict(color="lightgrey", width=2)
    ))

    # 3) spare capital as dashed black
    fig.add_trace(go.Scatter(
        x=tl["Date"], y=tl["SpareCapital"],
        mode="lines",
        name="Spare Capital",
        line=dict(dash="dash", width=1, color="black")
    ))

    # 4) broker balance as solid blue
    fig.add_trace(go.Scatter(
        x=tl["Date"], y=tl["BrokerBalance"],
        mode="lines",
        name="Broker Balance",
        line=dict(color="blue", width=2)
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Capital (×1,000)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Top‑row KPIs ───────────────────────────────────────────
    today       = tl.iloc[-1]
    spare       = today["SpareCapital"]
    total_alloc = today["TotalAllocation"]
    broker      = today["BrokerBalance"]
    st.write("")  # small spacer
    c1, c2, c3 = st.columns(3)
    c1.metric("Spare Today",     f"{spare:,.0f}")
    c2.metric("Total Allocated", f"{total_alloc:,.0f}")
    c3.metric("Broker Balance",  f"{broker:,.0f}")
    st.caption(f"Date  {today['Date'].strftime('%d‑%b‑%Y')}")

    # ─── Trade‑based KPIs ───────────────────────────────────────
    tr = trades.copy()
    tr["TradeDate_dt"] = pd.to_datetime(tr["TradeDate"], format="%d/%m/%y", dayfirst=True)
    daily_pnl = tr.groupby("TradeDate_dt")["NetCash"].sum().sort_index()

    net_pnl    = daily_pnl.sum()
    rf_daily   = 0.0  # or your chosen risk‑free
    excess     = daily_pnl - rf_daily
    sharpe     = excess.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
    gross_win  = daily_pnl[daily_pnl>0].sum()
    gross_loss = -daily_pnl[daily_pnl<0].sum()
    pf         = gross_win / gross_loss if gross_loss else np.nan
    win_days   = int((daily_pnl>0).sum())
    loss_days  = int((daily_pnl<0).sum())

    c4, c5, c6 = st.columns(3)
    c4.metric("Net PnL",        f"{net_pnl:,.2f}")
    c5.metric("Sharpe Ratio",   f"{sharpe:.2f}")
    c6.metric("Profit Factor",  f"{pf:.2f}")
    st.metric("Win / Loss Days", f"{win_days} / {loss_days}")

    # ─── NAV‑based KPIs ─────────────────────────────────────────
    nav = nav_df.copy()
    nav["Date"] = pd.to_datetime(nav["Date"], dayfirst=True)
    nav_ser     = nav.set_index("Date")["NAV"].sort_index()
    first_nav   = nav_ser.iloc[0]
    current_nav = nav_ser.iloc[-1]
    max_nav     = nav_ser.max()
    cummax_nav  = nav_ser.cummax()
    dd_ser      = nav_ser / cummax_nav - 1
    current_dd  = dd_ser.iloc[-1]
    max_dd      = dd_ser.min()

    days = (nav_ser.index[-1] - nav_ser.index[0]).days
    cagr = ((current_nav/first_nav) ** (365.25/days) - 1) if days>0 else np.nan

    c7, c8, c9 = st.columns(3)
    c7.metric("Current NAV", f"{current_nav:,.2f}")
    c8.metric("Max NAV",     f"{max_nav:,.2f}")
    c9.metric("Current DD",  f"{current_dd*100:,.2f} %")

    c10, c11, c12 = st.columns(3)
    c10.metric("Max DD", f"{max_dd*100:,.2f} %")
    c11.metric("CAGR",   f"{cagr*100:,.2f} %")
    c12.write("")  # spacer

    # ─── NAV Curve ──────────────────────────────────────────────
    st.markdown("### NAV Curve")
    st.line_chart(nav_ser)
