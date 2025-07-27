# dashboard/pages/strategy.py

import streamlit as st
import pandas as pd
import numpy as np

from scripts.loaders import (
    load_trades,      # returns DataFrame of all trades ledger
    load_timeline,    # returns DataFrame of raw allocation/margin timeline
    load_strategies,  # returns DataFrame of strategy definitions
    load_nav,         # returns DataFrame of precomputed NAVs (optional)
)


def calculate_strategy_nav(strategy_name, trades_df, timeline_df, selected_underlying=None):
    """
    Compute NAV series for a given strategy (and optional underlying) using:
      • NAV only moves on P&L%
      • New capital injected buys units at that day's NAV
      • Daily P&L% = NetCash_today / allocation_yesterday
    """

    # 0) If timeline_df is in tidy form, pivot to wide form:
    if {"Strategy", "UnderlyingSymbol", "Allocation"}.issubset(timeline_df.columns):
        timeline_df["Date"] = pd.to_datetime(timeline_df["Date"])
        pivot = (
            timeline_df
            .pivot_table(
                index="Date",
                columns=["Strategy", "UnderlyingSymbol"],
                values="Allocation",
                fill_value=0
            )
        )
        pivot.columns = [f"Allocation_{strat}_{und}" for strat, und in pivot.columns]
        timeline_df = pivot.reset_index()

    # 1) Extract the per-day allocation series
    if selected_underlying and selected_underlying != "All":
        alloc_col = f"Allocation_{strategy_name}_{selected_underlying}"
        if alloc_col not in timeline_df.columns:
            st.error(f"Allocation column not found: {alloc_col}")
            return pd.Series(dtype=float)
        strategy_alloc = (
            timeline_df
            .set_index("Date")[alloc_col]
            .sort_index()
        )
    else:
        alloc_cols = [
            c for c in timeline_df.columns
            if c.startswith(f"Allocation_{strategy_name}_")
        ]
        if not alloc_cols:
            st.error(f"No allocation columns found for {strategy_name}")
            return pd.Series(dtype=float)
        strategy_alloc = (
            timeline_df
            .set_index("Date")[alloc_cols]
            .sum(axis=1)
            .sort_index()
        )

    # 2) Build daily P&L series for that strategy
    df_strat = trades_df[trades_df["Strategy"] == strategy_name].copy()
    if df_strat.empty:
        return pd.Series(dtype=float)
    df_strat["Date"] = pd.to_datetime(
        df_strat["TradeDate"], format="%d/%m/%y", dayfirst=True
    )
    daily_pnl = df_strat.groupby("Date")["NetCash"].sum().sort_index()

    # 3) NAV & units loop
    nav_series      = []
    nav             = 100.0
    units           = 0.0
    prev_alloc      = 0.0

    for date, allocation in strategy_alloc.items():
        pnl_amt = daily_pnl.get(date, 0.0)

        # 3a) apply P&L% on yesterday's capital (if any)
        if prev_alloc > 0:
            pnl_pct = pnl_amt / prev_alloc
            nav    *= (1 + pnl_pct)

        # 3b) inject or withdraw capital at today's NAV
        if nav > 0 and allocation != prev_alloc:
            delta = allocation - prev_alloc
            units += delta / nav

        # 3c) record NAV (zero if no allocation)
        nav_series.append(nav if allocation > 0 else 0.0)
        prev_alloc = allocation

    return pd.Series(nav_series, index=strategy_alloc.index)


def render_individual_strategy(trades_df, strategy, timeline_df, nav_df, full_trades_df, selected_underlying):
    """Show metrics and charts for one strategy + optional underlying."""
    # — P&L & risk metrics
    trades_df["Date"] = pd.to_datetime(
        trades_df["TradeDate"], format="%d/%m/%y", dayfirst=True
    )
    daily_pnl    = trades_df.groupby("Date")["NetCash"].sum().sort_index()
    net_pnl      = daily_pnl.sum()
    rf_daily     = 0.045 / 252
    excess_ret   = daily_pnl - rf_daily
    sharpe       = excess_ret.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
    gross_wins   = daily_pnl[daily_pnl>0].sum()
    gross_losses = -daily_pnl[daily_pnl<0].sum()
    profit_fac   = gross_wins / gross_losses if gross_losses else np.nan
    win_days     = int((daily_pnl>0).sum())
    loss_days    = int((daily_pnl<0).sum())

    # — trade counts
    tg = trades_df.groupby(["UID","TradeDate"])
    total_trades = win_trades = loss_trades = 0
    for (_, _), legs in tg:
        cnt = max(1, legs["Quantity"].max())
        total_trades += cnt
        pnl = legs["NetCash"].sum()
        if pnl>0: win_trades+=cnt
        elif pnl<0: loss_trades+=cnt

    # — NAV series
    tl2    = timeline_df.copy()
    tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
    nav_ser = calculate_strategy_nav(
        strategy, full_trades_df, tl2, selected_underlying
    ).dropna()
    
    # — display key metrics
    st.subheader(f"🔹 {strategy}  —  {selected_underlying or 'All'}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Net P&L",       f"{net_pnl:,.0f}")
    c2.metric("Sharpe",        f"{sharpe:.2f}")
    c3.metric("Profit Factor", f"{profit_fac:.2f}")
    c4.metric("Win/Loss Days", f"{win_days}/{loss_days}")

    c5,c6,c7,_ = st.columns(4)
    c5.metric("Total Trades",        f"{total_trades}")
    c6.metric("Win/Loss Trades",     f"{win_trades}/{loss_trades}")
    c7.metric("Win Rate",            f"{(win_trades/total_trades*100):.1f} %" if total_trades else "N/A")

    # — NAV‑based metrics & CAGR
    if not nav_ser.empty and (active := nav_ser[nav_ser>0]).any():
        start_nav, end_nav = active.iloc[0], active.iloc[-1]
        max_nav            = active.max()
        drawdowns          = active/active.cummax() - 1
        curr_dd, max_dd    = drawdowns.iloc[-1], drawdowns.min()
        n_days             = len(active)
        cagr               = ((end_nav/start_nav)**(252/n_days) - 1) if n_days and start_nav>0 else np.nan

        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Current NAV",      f"{end_nav:,.2f}")
        d2.metric("Max NAV",          f"{max_nav:,.2f}")
        d3.metric("Curr Drawdown",    f"{curr_dd*100:.2f} %")
        d4.metric("Max Drawdown",     f"{max_dd*100:.2f} %")

        e1,e2,_,_ = st.columns(4)
        e1.metric("CAGR",             f"{cagr*100:.2f} %")

    # — charts
    st.markdown("#### Cumulative P&L")
    st.line_chart(daily_pnl.cumsum())

    if not nav_ser.empty and (active := nav_ser[nav_ser>0]).any():
        st.markdown("#### Strategy NAV")
        st.line_chart(active)

    else:
        st.info("No NAV data to display.")


def render_consolidated_view(full_trades_df, strategies, timeline_df, nav_df):
    """Show summary table & stats for all strategies."""
    st.subheader("📋 Consolidated Strategy Performance")
    rows = []
    for strategy in strategies:
        df = full_trades_df[full_trades_df["Strategy"] == strategy].copy()
        if df.empty:
            continue

        # P&L & risk
        df["Date"]   = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        pnl          = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl      = pnl.sum()
        rf_daily     = 0.045 / 252
        ex_ret       = pnl - rf_daily
        sharpe       = ex_ret.mean() / (pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
        wins         = pnl[pnl>0].sum()
        losses       = -pnl[pnl<0].sum()
        pf           = wins/losses if losses else np.nan
        win_days     = int((pnl>0).sum())
        loss_days    = int((pnl<0).sum())

        # trades count
        tg = df.groupby(["UID","TradeDate"])
        tot=w=l=0
        for (_, _), legs in tg:
            cnt = max(1, legs["Quantity"].max())
            tot+=cnt
            s = legs["NetCash"].sum()
            if s>0: w+=cnt
            elif s<0: l+=cnt

        # NAV & CAGR
        tl2    = timeline_df.copy()
        tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
        nav_ser = calculate_strategy_nav(strategy, df, tl2).dropna()
        if not nav_ser.empty and (active:=nav_ser[nav_ser>0]).any():
            start_nav, end_nav = active.iloc[0], active.iloc[-1]
            max_nav            = active.max()
            drawdowns          = active/active.cummax() - 1
            curr_dd, max_dd    = drawdowns.iloc[-1], drawdowns.min()
            n_days             = len(active)
            cagr               = ((end_nav/start_nav)**(252/n_days) - 1) if n_days and start_nav>0 else np.nan
        else:
            end_nav = max_nav = curr_dd = max_dd = cagr = np.nan

        rows.append({
            "Strategy":      strategy,
            "Net P&L":       net_pnl,
            "Sharpe":        f"{sharpe:.2f}",
            "Profit Factor": f"{pf:.2f}" if not np.isnan(pf) else "N/A",
            "Win/Loss Days": f"{win_days}/{loss_days}",
            "Trades":        tot,
            "Win Rate %":    f"{(w/tot*100):.1f}" if tot else "N/A",
            "Current NAV":   f"{end_nav:,.2f}" if not np.isnan(end_nav) else "N/A",
            "Max NAV":       f"{max_nav:,.2f}" if not np.isnan(max_nav) else "N/A",
            "Curr DD %":     f"{curr_dd*100:.2f}" if not np.isnan(curr_dd) else "N/A",
            "Max DD %":      f"{max_dd*100:.2f}" if not np.isnan(max_dd) else "N/A",
            "CAGR %":        f"{cagr*100:.2f}" if not np.isnan(cagr) else "N/A",
        })

    if rows:
        dfc = pd.DataFrame(rows)
        st.dataframe(dfc, use_container_width=True)

        # overall summary
        st.subheader("📊 Summary Across All Strategies")
        total_pnl = dfc["Net P&L"].sum()
        st.metric("Total Net P&L", f"{total_pnl:,.0f}")
        st.metric("Avg Sharpe",    f"{dfc['Sharpe'].astype(float).mean():.2f}")


def render():
    st.title("📈 Strategy Analytics")

    trades   = load_trades()
    tl       = load_timeline()
    strat_df = load_strategies()
    nav_df   = load_nav()

    strategies = sorted(strat_df["Strategy_id"].unique())
    view_mode  = st.radio("View Mode", ["Individual Strategy", "Consolidated View"], horizontal=True)

    if view_mode == "Individual Strategy":
        strategy = st.selectbox("Select Strategy", strategies)

        # underlying filter
        all_unds = sorted(trades["UnderlyingSymbol"].unique())
        selected_underlying = st.selectbox("Filter by Underlying", ["All"] + all_unds)

        # date range filter
        presets = {
            "All to Date":   "all",
            "Last 1 Day":     1,
            "Last 7 Days":    7,
            "Last 30 Days":  30,
            "Last 6 Months":180,
            "Last 1 Year":  365,
            "Custom Range": "custom"
        }
        choice = st.selectbox("Date Range", list(presets.keys()), index=0)
        if choice == "Custom Range":
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start Date", value=None)
            end_date   = c2.date_input("End Date",   value=None)
        else:
            days_back = presets[choice]
            if days_back == "all":
                start_date = end_date = None
                st.write("📅 **All to Date**")
            else:
                days_back = int(days_back)
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.write(f"📅 {choice}: {start_date} to {end_date}")

        # filter trades
        df = trades[trades["Strategy"] == strategy].copy()
        if selected_underlying != "All":
            df = df[df["UnderlyingSymbol"] == selected_underlying]
        if start_date and end_date:
            df["TradeDate_dt"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            df = df[
                (df["TradeDate_dt"].dt.date >= start_date) &
                (df["TradeDate_dt"].dt.date <= end_date)
            ]
            df.drop("TradeDate_dt", axis=1, inplace=True)

        if df.empty:
            st.warning("No trades found for these filters.")
        else:
            render_individual_strategy(df, strategy, tl, nav_df, trades, selected_underlying)

    else:
        render_consolidated_view(trades, strategies, tl, nav_df)
