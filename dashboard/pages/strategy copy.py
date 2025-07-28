# dashboard/pages/strategy.py

import streamlit as st
import pandas as pd
import numpy as np

from scripts.loaders import (
    load_trades,      # DataFrame of your trades ledger
    load_timeline,    # DataFrame of your raw allocation/margin timeline
    load_strategies,  # DataFrame listing available strategies
)


def calculate_strategy_nav_with_units(strategy_name, trades_df, timeline_df, selected_underlying=None):
    """
    Returns a DataFrame indexed by Date with columns:
      - nav   : Strategy NAV (starts at 100, steps on pnl%)
      - units : Total units held (adjusts only when allocation changes)
    Assumes allocation columns named:
      Allocation_<Strategy>_Allocation_<Underlying>
    """
    # 0) Pivot tidy timeline_df into wide form if needed
    if {"Strategy", "UnderlyingSymbol", "Allocation"}.issubset(timeline_df.columns):
        timeline_df["Date"] = pd.to_datetime(timeline_df["Date"])
        pivot = (
            timeline_df
              .pivot_table(
                  index="Date",
                  columns=["Strategy", "UnderlyingSymbol"],
                  values="Allocation",
                  aggfunc="sum",
                  fill_value=0
              )
        )
        pivot.columns = [
            f"Allocation_{s}_Allocation_{u}"
            for s, u in pivot.columns
        ]
        pivot = pivot.sort_index()
        timeline_df = pivot.reset_index()

    # 1) Extract allocation series
    if selected_underlying and selected_underlying != "All":
        alloc_col = f"Allocation_{strategy_name}_Allocation_{selected_underlying}"
        if alloc_col not in timeline_df.columns:
            # Return empty DataFrame for strategies without allocation data
            return pd.DataFrame(columns=["nav", "units"])
        alloc_ser = timeline_df.set_index("Date")[alloc_col].sort_index()
    else:
        alloc_cols = [
            c for c in timeline_df.columns
            if c.startswith(f"Allocation_{strategy_name}_Allocation_")
        ]
        if not alloc_cols:
            # Return empty DataFrame for strategies without allocation data
            return pd.DataFrame(columns=["nav", "units"])
        alloc_ser = timeline_df.set_index("Date")[alloc_cols].sum(axis=1).sort_index()

    # 2) Carry forward until changed, then drop zeros
    alloc_ser = alloc_ser.ffill().fillna(0)
    alloc_ser = alloc_ser[alloc_ser > 0]
    
    # Check if we have any allocation data
    if alloc_ser.empty:
        return pd.DataFrame(columns=["nav", "units"])

    # 3) Build daily P&L series
    df_strat = trades_df[trades_df["Strategy"] == strategy_name].copy()
    df_strat["Date"] = pd.to_datetime(
        df_strat["TradeDate"], format="%d/%m/%y", dayfirst=True
    )
    pnl_ser = df_strat.groupby("Date")["NetCash"].sum()

    # 4) Merge allocation + P&L, flag relevant days
    df = (
        pd.DataFrame({"allocation": alloc_ser})
          .join(pnl_ser.rename("pnl"), how="left")
          .fillna({"pnl": 0})
          .sort_index()
    )
    df["prev_alloc"] = df["allocation"].shift(1)
    # Only set prev_alloc if DataFrame is not empty
    if not df.empty:
        df.loc[df.index[0], "prev_alloc"] = df["allocation"].iloc[0]
    df["alloc_change"] = df["allocation"] != df["prev_alloc"]
    df["relevant"]    = (df["pnl"] != 0) | df["alloc_change"]
    df = df.loc[df["relevant"]]

    # 5) Loop through relevant days, tracking NAV & units
    nav_list, units_list = [], []
    nav   = 100.0
    units = df["allocation"].iloc[0] / nav

    for _, row in df.iterrows():
        # apply that day's P&L% on previous capital
        nav *= (1 + row["pnl"] / row["prev_alloc"])
        # adjust units only on allocation change
        if row["alloc_change"]:
            delta = row["allocation"] - row["prev_alloc"]
            units += delta / nav
        # record
        nav_list.append(nav)
        units_list.append(units)

    nav_df = pd.DataFrame({"nav": nav_list, "units": units_list}, index=df.index)

    # 6) Prepend NAV=100 on day before first active date
    first_date = nav_df.index.min()
    prev_day   = first_date - pd.Timedelta(days=1)
    start_units = nav_df["units"].iloc[0]
    prepend = pd.DataFrame(
        {"nav": [100.0], "units": [start_units]},
        index=[prev_day]
    )
    nav_df = pd.concat([prepend, nav_df]).sort_index()

    return nav_df


def render_individual_strategy(df, strategy, timeline_df, filtered_trades_df, selected_underlying):
    """
    Display metrics & charts for one strategy + (optional) underlying,
    including NAV, units, and portfolio value.
    """
    # — P&L & risk metrics
    df["Date"]  = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
    daily_pnl   = df.groupby("Date")["NetCash"].sum().sort_index()
    net_pnl     = daily_pnl.sum()
    rf_daily    = 0.045 / 252
    excess_ret  = daily_pnl - rf_daily
    sharpe      = excess_ret.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
    wins        = daily_pnl[daily_pnl > 0].sum()
    losses      = -daily_pnl[daily_pnl < 0].sum()
    profit_fac  = wins / losses if losses else np.nan
    win_days    = int((daily_pnl > 0).sum())
    loss_days   = int((daily_pnl < 0).sum())

    # — trade counts
    tg            = df.groupby(["UID", "TradeDate"])
    total, win_t, loss_t = 0, 0, 0
    for (_, _), legs in tg:
        cnt = max(1, legs["Quantity"].max())
        total += cnt
        s = legs["NetCash"].sum()
        if s > 0: win_t += cnt
        elif s < 0: loss_t += cnt

    # — slippage metrics
    total_slippage = df["Slippage"].sum() if "Slippage" in df.columns else 0
    
    # Calculate weighted average slippage percentage (weighted by planned loss)
    if "Slippage_pct" in df.columns and "Slippage" in df.columns:
        # Only consider trades with actual slippage (Slippage > 0)
        slippage_trades = df[df["Slippage"] > 0]
        if not slippage_trades.empty:
            # Weight by the actual slippage amount (dollar value)
            weighted_avg = (slippage_trades["Slippage"] * slippage_trades["Slippage_pct"]).sum() / slippage_trades["Slippage"].sum()
            avg_slippage_pct = weighted_avg
        else:
            avg_slippage_pct = 0
    else:
        avg_slippage_pct = 0

    # — NAV & units series (using filtered trades for date range consistency)
    tl2 = timeline_df.copy()
    tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
    nav_units = calculate_strategy_nav_with_units(
        strategy, filtered_trades_df, tl2, selected_underlying
    )
    
    # Check if NAV data is available
    if nav_units.empty:
        nav_ser = pd.Series(dtype=float)
        units_ser = pd.Series(dtype=float)
        val_ser = pd.Series(dtype=float)
        has_nav_data = False
    else:
        nav_ser = nav_units["nav"]
        units_ser = nav_units["units"]
        val_ser = nav_ser * units_ser
        has_nav_data = True

    # — display key metrics
    st.subheader(f"🔹 {strategy}  —  {selected_underlying or 'All'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net P&L",       f"{net_pnl:,.0f}")
    c2.metric("Sharpe Ratio",  f"{sharpe:.2f}")
    c3.metric("Profit Factor", f"{profit_fac:.2f}")
    c4.metric("Win/Loss Days", f"{win_days}/{loss_days}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades",     f"{total}")
    c6.metric("Win/Loss Trades",  f"{win_t}/{loss_t}")
    c7.metric("Win Rate",         f"{win_t/total*100:.1f} %")
    if has_nav_data and not nav_ser.empty:
        c8.metric("Current NAV",      f"{nav_ser.iloc[-1]:.2f}")
    else:
        c8.metric("Current NAV",      "N/A")

    c9, c10 = st.columns(2)
    if has_nav_data and not units_ser.empty:
        c9.metric("Current Units",    f"{units_ser.iloc[-1]:,.2f}")
        c10.metric("Portfolio Value", f"{val_ser.iloc[-1]:,.0f}")
    else:
        c9.metric("Current Units",    "N/A")
        c10.metric("Portfolio Value", "N/A")

    # — slippage metrics
    c11, c12 = st.columns(2)
    c11.metric("Total Slippage", f"{total_slippage:,.0f}")
    c12.metric("Avg Slippage %", f"{avg_slippage_pct:.2f} %")

    # — NAV metrics & CAGR
    if has_nav_data:
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            start_nav = active_nav.iloc[0]  # now the prepended 100
            end_nav   = active_nav.iloc[-1]
            # Calculate only active trading days (when NAV data exists)
            active_days = len(active_nav) - 1  # Number of active trading days
            cagr      = (end_nav / start_nav)**(252.0 / active_days) - 1

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Max NAV",       f"{active_nav.max():.2f}")
            d2.metric("Curr Drawdown", f"{(active_nav/active_nav.cummax()-1).iloc[-1]*100:.2f} %")
            d3.metric("Max Drawdown",  f"{(active_nav/active_nav.cummax()-1).min()*100:.2f} %")
            d4.metric("CAGR",          f"{cagr*100:.2f} %")
        else:
            st.info("No NAV data available for this strategy.")
    else:
        st.info("No allocation data available for NAV calculation.")

    # — charts
    st.markdown("#### Cumulative P&L")
    st.line_chart(daily_pnl.cumsum())

    if has_nav_data and not nav_ser.empty:
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            st.markdown("#### NAV vs Time")
            st.line_chart(active_nav)
            st.markdown("#### Units vs Time")
            st.line_chart(units_ser)
        else:
            st.info("No NAV data to display.")
    else:
        st.info("No NAV charts available - allocation data required.")


def render_consolidated_view(full_trades_df, strategies, timeline_df):
    """
    Display a comprehensive summary table with all KPI metrics for all strategies.
    """
    st.subheader("📋 Consolidated Strategy Performance")
    rows = []
    
    for strategy in strategies:
        df = full_trades_df[full_trades_df["Strategy"] == strategy].copy()
        if df.empty:
            continue

        # — P&L & risk metrics (same as individual view)
        df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl = daily_pnl.sum()
        rf_daily = 0.045 / 252
        excess_ret = daily_pnl - rf_daily
        sharpe = excess_ret.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
        wins = daily_pnl[daily_pnl > 0].sum()
        losses = -daily_pnl[daily_pnl < 0].sum()
        profit_fac = wins / losses if losses else np.nan
        win_days = int((daily_pnl > 0).sum())
        loss_days = int((daily_pnl < 0).sum())

        # — trade counts (same as individual view)
        tg = df.groupby(["UID", "TradeDate"])
        total, win_t, loss_t = 0, 0, 0
        for (_, _), legs in tg:
            cnt = max(1, legs["Quantity"].max())
            total += cnt
            s = legs["NetCash"].sum()
            if s > 0: 
                win_t += cnt
            elif s < 0: 
                loss_t += cnt
        
        win_rate = (win_t / total * 100) if total > 0 else 0

        # — slippage metrics
        total_slippage = df["Slippage"].sum() if "Slippage" in df.columns else 0
        
        # Calculate weighted average slippage percentage (weighted by planned loss)
        if "Slippage_pct" in df.columns and "Slippage" in df.columns:
            # Only consider trades with actual slippage (Slippage > 0)
            slippage_trades = df[df["Slippage"] > 0]
            if not slippage_trades.empty:
                # Weight by the actual slippage amount (dollar value)
                weighted_avg = (slippage_trades["Slippage"] * slippage_trades["Slippage_pct"]).sum() / slippage_trades["Slippage"].sum()
                avg_slippage_pct = weighted_avg
            else:
                avg_slippage_pct = 0
        else:
            avg_slippage_pct = 0

        # — NAV & units
        tl2 = timeline_df.copy()
        tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
        nav_units = calculate_strategy_nav_with_units(strategy, df, tl2)
        
        if nav_units.empty:
            cagr = np.nan
            current_nav = np.nan
            current_units = np.nan
            portfolio_value = np.nan
        else:
            nav_ser = nav_units["nav"]
            units_ser = nav_units["units"]
            active_nav = nav_ser[nav_ser > 0]

            if not active_nav.empty:
                end_nav = active_nav.iloc[-1]
                # Calculate only active trading days (when NAV data exists)
                active_days = len(active_nav) - 1  # Number of active trading days
                cagr = (end_nav / active_nav.iloc[0])**(252.0 / active_days) - 1
                current_nav = end_nav
                current_units = units_ser.iloc[-1] if not units_ser.empty else np.nan
                portfolio_value = current_nav * current_units if not np.isnan(current_nav) and not np.isnan(current_units) else np.nan
            else:
                cagr = np.nan
                current_nav = np.nan
                current_units = np.nan
                portfolio_value = np.nan

        rows.append({
            "Strategy": strategy,
            "Net P&L": f"{net_pnl:,.0f}",
            "Sharpe": f"{sharpe:.2f}",
            "Profit Factor": f"{profit_fac:.2f}" if not np.isnan(profit_fac) else "N/A",
            "Total Trades": total,
            "Win/Loss Trades": f"{win_t}/{loss_t}",
            "Win Rate %": f"{win_rate:.1f}",
            "CAGR %": f"{cagr*100:.2f} %" if not np.isnan(cagr) else "N/A",
            "Current NAV": f"{current_nav:.2f}" if not np.isnan(current_nav) else "N/A",
            "Current Units": f"{current_units:,.2f}" if not np.isnan(current_units) else "N/A",
            "Total Slippage": f"{total_slippage:,.0f}",
            "Avg Slippage %": f"{avg_slippage_pct:.2f} %",
        })

    if rows:
        dfc = pd.DataFrame(rows)
        st.dataframe(dfc, use_container_width=True)

        # — Summary metrics across all strategies
        st.subheader("📊 Summary Across All Strategies")
        
        # Calculate totals and averages
        total_pnl = sum(float(row["Net P&L"].replace(",", "")) for row in rows)
        sharpe_values = [float(row["Sharpe"]) for row in rows]
        avg_sharpe = np.mean(sharpe_values) if sharpe_values else 0
        
        # Calculate total trades and win rates
        total_trades = sum(row["Total Trades"] for row in rows)
        total_win_trades = sum(int(row["Win/Loss Trades"].split("/")[0]) for row in rows)
        total_loss_trades = sum(int(row["Win/Loss Trades"].split("/")[1]) for row in rows)
        overall_win_rate = (total_win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate total slippage
        total_slippage = sum(float(row["Total Slippage"].replace(",", "")) for row in rows if row["Total Slippage"] != "0")
        slippage_pcts = [float(row["Avg Slippage %"].replace(" %", "")) for row in rows if row["Avg Slippage %"] != "0.00 %"]
        avg_slippage_pct = np.mean(slippage_pcts) if slippage_pcts else 0
        
        # Display summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Net P&L", f"{total_pnl:,.0f}")
        c2.metric("Average Sharpe", f"{avg_sharpe:.2f}")
        c3.metric("Total Trades", f"{total_trades}")
        c4.metric("Overall Win Rate", f"{overall_win_rate:.1f} %")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win/Loss Trades", f"{total_win_trades}/{total_loss_trades}")
        c6.metric("Total Slippage", f"{total_slippage:,.0f}")
        c7.metric("Avg Slippage %", f"{avg_slippage_pct:.2f} %")
        c8.metric("Active Strategies", len([r for r in rows if r["Current NAV"] != "N/A"]))


def render():
    st.title("📈 Strategy Analytics")

    trades     = load_trades()
    timeline   = load_timeline()
    strat_df   = load_strategies()
    strategies = sorted(strat_df["Strategy_id"].unique())

    view_mode = st.radio(
        "View Mode",
        ["Individual Strategy", "Consolidated View"],
        horizontal=True
    )

    if view_mode == "Individual Strategy":
        strategy = st.selectbox("Select Strategy", strategies)
        all_unds = sorted(trades["UnderlyingSymbol"].unique())
        selected_underlying = st.selectbox(
            "Filter by Underlying", ["All"] + all_unds
        )

        # Date range filter
        presets = {
            "All to Date":   "all",
            "Last 1 Day":     1,
            "Last 7 Days":    7,
            "Last 30 Days":   30,
            "Last 6 Months":180,
            "Last 1 Year":    365,
            "Custom Range":  "custom"
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
                st.write("📅 All to Date")
            else:
                days_back = int(days_back)
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.write(f"📅 {choice}: {start_date} to {end_date}")

        # filter trades
        df = trades[trades["Strategy"] == strategy].copy()
        if selected_underlying != "All":
            df = df[df["UnderlyingSymbol"] == selected_underlying]
        if 'start_date' in locals() and start_date and end_date:
            df["_dt"] = pd.to_datetime(
                df["TradeDate"], format="%d/%m/%y", dayfirst=True
            )
            df = df[(df["_dt"].dt.date >= start_date) &
                    (df["_dt"].dt.date <= end_date)]
            df.drop("_dt", axis=1, inplace=True)

        if df.empty:
            st.warning("No trades found for these filters.")
        else:
            render_individual_strategy(
                df, strategy, timeline, df, selected_underlying
            )
    else:
        render_consolidated_view(trades, strategies, timeline)
