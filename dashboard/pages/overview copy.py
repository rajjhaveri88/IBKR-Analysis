# dashboard/pages/overview.py
import streamlit as st
import pandas as pd
import numpy as np

from scripts.loaders import load_timeline, load_fund_flow, load_nav, load_trades

def render():
    st.title("📈 Account Overview")

    # ─── Load data ─────────────────────────────────────────────
    tl     = load_timeline()      # timeline with BrokerBalance & allocations
    ff     = load_fund_flow()     # fund‐flow for deposits/withdrawals
    nav_df = load_nav()           # daily NAV series
    trades = load_trades()        # cleaned trade ledger

    if tl.empty or nav_df.empty or trades.empty:
        st.warning("Not enough data available to render the overview.")
        return

    # ─── Top‐row KPIs from timeline ────────────────────────────
    today      = tl.iloc[-1]
    alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
    total_alloc= today[alloc_cols].sum() if alloc_cols else 0.0
    spare      = today["BrokerBalance"] - total_alloc

    c1, c2, c3 = st.columns(3)
    c1.metric("Spare Today",     f"{spare:,.0f}")
    c2.metric("Total Allocated", f"{total_alloc:,.0f}")
    c3.metric("Broker Balance",  f"{today['BrokerBalance']:,.0f}")
    st.caption(f"Date  {pd.to_datetime(today['Date']).strftime('%d‑%b‑%Y')}")

    # ─── NAV‐based KPIs ─────────────────────────────────────────
    nav_series  = nav_df.set_index("Date")["NAV"].sort_index()
    current_nav = nav_series.iloc[-1]
    max_nav     = nav_series.max()
    cummax_nav  = nav_series.cummax()
    current_dd  = (current_nav - cummax_nav.iloc[-1]) / cummax_nav.iloc[-1]
    max_dd      = (nav_series - cummax_nav).min()

    days = (nav_series.index[-1] - nav_series.index[0]).days
    cagr = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (365.25 / days) - 1) if days > 0 else np.nan

    c4, c5, c6 = st.columns(3)
    c4.metric("Current NAV", f"{current_nav:,.2f}")
    c5.metric("Max NAV",     f"{max_nav:,.2f}")
    c6.metric("Current DD",  f"{current_dd*100:,.2f} %")

    c7, c8, c9 = st.columns(3)
    c7.metric("Max DD",       f"{max_dd*100:,.2f} %")
    c8.metric("CAGR",         f"{cagr*100:,.2f} %")
    c9.metric("Net PnL",      f"{trades['NetCash'].sum():,.2f}")

    # ─── Trade‑based KPIs ────────────────────────────────────────
    # compute daily PnL
    trades["TradeDate_dt"] = pd.to_datetime(trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
    daily_pnl = trades.groupby("TradeDate_dt")["NetCash"].sum().sort_index()

    # Sharpe ratio (no risk‐free)
    mean_pnl = daily_pnl.mean()
    std_pnl  = daily_pnl.std(ddof=0) or 1e-9
    sharpe   = mean_pnl / std_pnl * np.sqrt(252)

    # Profit factor = gross wins / gross losses
    gross_win  = trades.loc[trades.NetCash > 0, "NetCash"].sum()
    gross_loss = -trades.loc[trades.NetCash < 0, "NetCash"].sum()
    pf         = gross_win / gross_loss if gross_loss else np.nan

    # Win/Loss days
    win_days  = int((daily_pnl > 0).sum())
    loss_days = int((daily_pnl < 0).sum())

    c10, c11, c12 = st.columns(3)
    c10.metric("Sharpe",        f"{sharpe:.2f}")
    c11.metric("Profit Factor", f"{pf:.2f}")
    c12.metric("Win / Loss Days", f"{win_days} / {loss_days}")

    # ─── NAV Curve ──────────────────────────────────────────────
    st.markdown("### NAV Curve")
    st.line_chart(nav_series)
