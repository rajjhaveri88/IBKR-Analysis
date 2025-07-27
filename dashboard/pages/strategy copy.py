# dashboard/pages/strategy.py
import streamlit as st
import pandas as pd
import numpy as np

from scripts.loaders import (
    load_trades,      # legs.parquet
    load_timeline,    # timeline.parquet
    load_strategies,  # strategies.parquet
    load_nav,         # nav.parquet
)

def calculate_strategy_nav(strategy_name, trades_df, timeline_df, selected_underlying=None):
    """
    Calculate strategy NAV using mutual‑fund logic:
      • NAV only changes due to P&L
      • Allocation changes buy/sell units at current NAV
      • Daily P&L % = Daily P&L / Allocated Capital (previous day)
      • If an underlying is specified, only use that slice of allocation
    """
    # 1) Extract allocation series for this strategy (and underlying, if set)
    if selected_underlying and selected_underlying != "All":
        alloc_col = f"Allocation_{strategy_name}_Allocation_{selected_underlying}"
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

    # 2) Build daily P&L series
    strat_trades = trades_df[trades_df["Strategy"] == strategy_name].copy()
    if strat_trades.empty:
        return pd.Series(dtype=float)
    strat_trades["Date"] = pd.to_datetime(
        strat_trades["TradeDate"],
        format="%d/%m/%y",
        dayfirst=True
    )
    daily_pnl = strat_trades.groupby("Date")["NetCash"].sum().sort_index()

    # 3) Loop through each date, adjusting NAV then units
    nav_series      = []
    nav             = 100.0    # starting NAV
    units           = 0.0
    prev_allocation = 0.0

    for date in strategy_alloc.index:
        allocation        = strategy_alloc.loc[date]
        daily_pnl_amount  = daily_pnl.get(date, 0.0)

        if allocation > 0:
            if units == 0:
                # First‑time injection
                units = allocation / nav if nav > 0 else allocation / 100.0
                nav_series.append(nav)
            else:
                # Inject or withdraw at today's NAV
                if allocation != prev_allocation and nav > 0:
                    delta = allocation - prev_allocation
                    units += delta / nav

                # Apply performance
                if daily_pnl_amount != 0 and prev_allocation > 0:
                    pnl_pct = daily_pnl_amount / prev_allocation
                    nav     = nav * (1 + pnl_pct)

                nav_series.append(nav)
        else:
            # Strategy inactive
            nav_series.append(0.0)

        prev_allocation = allocation

    return pd.Series(nav_series, index=strategy_alloc.index)


def render_individual_strategy(df, strategy, tl, nav, selected_underlying):
    """Render metrics & charts for one strategy / underlying slice."""
    # 1) Basic P&L & risk metrics
    df["Date"]     = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
    daily_pnl      = df.groupby("Date")["NetCash"].sum().sort_index()
    net_pnl        = daily_pnl.sum()
    rf_daily       = 0.045 / 252
    excess_ret     = daily_pnl - rf_daily
    sharpe         = excess_ret.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
    gross_wins     = daily_pnl[daily_pnl>0].sum()
    gross_losses   = -daily_pnl[daily_pnl<0].sum()
    profit_factor  = gross_wins / gross_losses if gross_losses else np.nan
    win_days       = int((daily_pnl>0).sum())
    loss_days      = int((daily_pnl<0).sum())

    # 2) Win/Loss trades count
    trade_groups    = df.groupby(["UID", "TradeDate"])
    total_trades    = winning_trades = losing_trades = 0
    for (_, _), legs in trade_groups:
        cnt = max(1, legs["Quantity"].max())
        total_trades += cnt
        pnl = legs["NetCash"].sum()
        if pnl > 0:      winning_trades += cnt
        elif pnl < 0:    losing_trades  += cnt

    # 3) NAV series & unit logic
    tl2       = tl.copy()
    tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
    nav_ser   = calculate_strategy_nav(strategy, df, tl2, selected_underlying).dropna()

    # 4) Display metrics
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net PnL",        f"{net_pnl:,.0f}")
    c2.metric("Sharpe Ratio",   f"{sharpe:.2f}")
    c3.metric("Profit Factor",  f"{profit_factor:.2f}")
    c4.metric("Win / Loss Days", f"{win_days} / {loss_days}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades",       f"{total_trades}")
    c6.metric("Win / Loss Trades",  f"{winning_trades} / {losing_trades}")
    c7.metric("Win Rate",           f"{(winning_trades/total_trades*100):.1f} %" if total_trades else "N/A")
    c8.write("")

    # 5) NAV‑based metrics & CAGR
    if not nav_ser.empty:
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            start_nav  = active_nav.iloc[0]
            end_nav    = active_nav.iloc[-1]
            max_nav    = active_nav.max()
            dd_s       = active_nav / active_nav.cummax() - 1
            current_dd = dd_s.iloc[-1]
            max_dd     = dd_s.min()
            n_days     = len(active_nav)
            cagr       = ((end_nav / start_nav) ** (252 / n_days) - 1) if n_days and start_nav>0 else np.nan

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Current NAV",      f"{end_nav:,.2f}")
            d2.metric("Max NAV",          f"{max_nav:,.2f}")
            d3.metric("Current Drawdown", f"{current_dd*100:.2f} %")
            d4.metric("Max Drawdown",     f"{max_dd*100:.2f} %")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("CAGR",             f"{cagr*100:.2f} %")
            e2.write("") 
            e3.write("") 
            e4.write("")
        else:
            st.info(f"Strategy {strategy} has no active periods (allocation > 0).")
    else:
        st.info(f"No allocation data for {strategy}. Showing only trade‑based metrics.")

    # 6) Charts
    st.markdown("#### Cumulative P&L")
    st.line_chart(daily_pnl.cumsum())

    if not nav_ser.empty:
        st.markdown("#### Strategy NAV (Performance)")
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            st.line_chart(active_nav)
            if selected_underlying != "All":
                st.info(f"📊 NAV for {selected_underlying} only")
            else:
                st.info("📊 Overall strategy NAV")
        else:
            st.info("No active NAV data to display.")
    else:
        st.info("No NAV chart available—no allocation data.")


def render_consolidated_view(trades, strategies, tl, nav):
    """Render a summary table for all strategies."""
    st.subheader("Consolidated Strategy Performance")
    consolidated = []

    for strategy in strategies:
        df = trades[trades["Strategy"] == strategy].copy()
        if df.empty:
            continue

        # P&L & risk metrics
        df["Date"]     = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        pnl            = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl        = pnl.sum()
        rf_daily       = 0.045 / 252
        excess_ret     = pnl - rf_daily
        sharpe         = excess_ret.mean() / (pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
        wins           = pnl[pnl>0].sum()
        losses         = -pnl[pnl<0].sum()
        pf             = wins / losses if losses else np.nan
        win_days       = int((pnl>0).sum())
        loss_days      = int((pnl<0).sum())

        # Trades count
        tg       = df.groupby(["UID", "TradeDate"])
        tot, w, l = 0, 0, 0
        for (_, _), legs in tg:
            cnt = max(1, legs["Quantity"].max())
            tot += cnt
            s = legs["NetCash"].sum()
            if s>0:   w += cnt
            elif s<0: l += cnt

        # NAV & CAGR
        tl2       = tl.copy()
        tl2["Date"]= pd.to_datetime(tl2["Date"], dayfirst=True)
        nav_ser   = calculate_strategy_nav(strategy, trades, tl2).dropna()
        if not nav_ser.empty and (a:=nav_ser[nav_ser>0]).any():
            start_nav  = a.iloc[0]
            end_nav    = a.iloc[-1]
            max_nav    = a.max()
            dd_s       = a / a.cummax() - 1
            curr_dd    = dd_s.iloc[-1]
            max_dd     = dd_s.min()
            n_days     = len(a)
            cagr       = ((end_nav / start_nav) ** (252 / n_days) - 1) if n_days and start_nav>0 else np.nan
        else:
            end_nav = max_nav = curr_dd = max_dd = cagr = np.nan

        win_rate = (w / tot * 100) if tot else 0
        consolidated.append({
            "Strategy": strategy,
            "Net PnL":        net_pnl,
            "Sharpe":         f"{sharpe:.2f}",
            "Profit Factor":  f"{pf:.2f}" if not np.isnan(pf) else "N/A",
            "Win/Loss Days":  f"{win_days}/{loss_days}",
            "Total Trades":   tot,
            "Win/Loss Trades":f"{w}/{l}",
            "Win Rate %":     f"{win_rate:.1f}",
            "Current NAV":    f"{end_nav:,.2f}" if not np.isnan(end_nav) else "N/A",
            "Max NAV":        f"{max_nav:,.2f}" if not np.isnan(max_nav) else "N/A",
            "Curr DD %":      f"{curr_dd*100:.2f}" if not np.isnan(curr_dd) else "N/A",
            "Max DD %":       f"{max_dd*100:.2f}" if not np.isnan(max_dd) else "N/A",
            "CAGR %":         f"{cagr*100:.2f}" if not np.isnan(cagr) else "N/A",
        })

    if consolidated:
        dfc = pd.DataFrame(consolidated)
        st.dataframe(dfc, use_container_width=True)

        st.subheader("Summary Statistics")
        s1, s2, s3, s4 = st.columns(4)
        total_pnl        = dfc["Net PnL"].sum()
        total_trades_all = dfc["Total Trades"].sum()
        avg_wr           = dfc["Win Rate %"].str.rstrip(" %").astype(float).mean()
        avg_sh           = dfc["Sharpe"].astype(float).mean()
        pf_vals          = dfc[dfc["Profit Factor"]!="N/A"]["Profit Factor"].astype(float)
        avg_pf           = pf_vals.mean() if not pf_vals.empty else 0

        s1.metric("Total P&L",             f"{total_pnl:,.0f}")
        s2.metric("Total Trades",          f"{total_trades_all}")
        s3.metric("Avg Win Rate",          f"{avg_wr:.1f} %")
        s4.metric("Strategies Count",      len(dfc))

        s5, s6, _, _ = st.columns(4)
        s5.metric("Avg Sharpe Ratio",      f"{avg_sh:.2f}")
        s6.metric("Avg Profit Factor",     f"{avg_pf:.2f}" if avg_pf else "N/A")


def render():
    st.title("📊 Strategy Analytics")

    # Load source data
    trades   = load_trades()
    tl       = load_timeline()
    strat_df = load_strategies()
    nav      = load_nav()

    strategies = sorted(strat_df["Strategy_id"].unique())
    view_mode  = st.radio("View Mode", ["Individual Strategy", "Consolidated View"], horizontal=True)

    if view_mode == "Individual Strategy":
        strategy = st.selectbox("Select Strategy", strategies)
        col1, col2 = st.columns(2)

        with col1:
            all_unders = sorted(trades["UnderlyingSymbol"].unique())
            selected_underlying = st.selectbox("Filter by Underlying", ["All"] + all_unders)

        with col2:
            st.write("**Date Range Filter**")
            presets = {
                "All to Date": "all",
                "Last 1 Day":   1,
                "Last 7 Days":  7,
                "Last 30 Days": 30,
                "Last 6 Months":180,
                "Last 1 Year":  365,
                "Custom Range":"custom"
            }
            choice = st.selectbox("Quick Select", list(presets.keys()), index=0)
            if choice == "Custom Range":
                c1, c2 = st.columns(2)
                with c1:
                    start_date = st.date_input("Start Date", value=None)
                with c2:
                    end_date   = st.date_input("End Date",   value=None)
            elif choice == "All to Date":
                start_date = end_date = None
                st.write("📅 **All to Date**")
            else:
                days_back  = presets[choice]
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.write(f"📅 {choice}: {start_date} to {end_date}")

        df = trades[trades["Strategy"] == strategy].copy()
        if selected_underlying != "All":
            df = df[df["UnderlyingSymbol"] == selected_underlying]
        if start_date and end_date:
            df["TradeDate_dt"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            df = df[(df["TradeDate_dt"].dt.date >= start_date) &
                    (df["TradeDate_dt"].dt.date <= end_date)]
            df.drop("TradeDate_dt", axis=1, inplace=True)

        if df.empty:
            st.warning("No trades found for the selected filters.")
            return

        render_individual_strategy(df, strategy, tl, nav, selected_underlying)

    else:
        render_consolidated_view(trades, strategies, tl, nav)
