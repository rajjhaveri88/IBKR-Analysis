# dashboard/pages/strategy.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scripts.loaders import (
    load_trades,      # DataFrame of your trades ledger
    load_timeline,    # DataFrame of your raw allocation/margin timeline
    load_strategies,  # DataFrame listing available strategies
)


def filter_trades_by_dates(trades_df, strategy_name, underlying_symbol, start_date, end_date):
    """Filter trades by date range"""
    filtered = trades_df[trades_df["Strategy"] == strategy_name].copy()
    if underlying_symbol != "All":
        filtered = filtered[filtered["UnderlyingSymbol"] == underlying_symbol]
    if start_date and end_date:
        filtered["_dt"] = pd.to_datetime(
            filtered["TradeDate"], format="%d/%m/%y", dayfirst=True
        )
        filtered = filtered[(filtered["_dt"].dt.date >= start_date) &
                          (filtered["_dt"].dt.date <= end_date)]
        filtered.drop("_dt", axis=1, inplace=True)
    return filtered


def calculate_pct_change(primary_value, comparison_value):
    """Calculate percentage change between two values"""
    if comparison_value == 0:
        return 0 if primary_value == 0 else float('inf') if primary_value > 0 else float('-inf')
    return ((primary_value - comparison_value) / abs(comparison_value)) * 100


def render_comparison_metric(label, primary_value, comparison_value, format_type="number", prefix=""):
    """Render a metric with comparison data side-by-side"""
    if comparison_value is not None:
        pct_change = calculate_pct_change(primary_value, comparison_value)
        primary_color = "positive-value" if primary_value >= 0 else "negative-value" if primary_value < 0 else "neutral-value"
        comp_color = "positive-value" if comparison_value >= 0 else "negative-value" if comparison_value < 0 else "neutral-value"
        pct_color = "positive-value" if pct_change > 0 else "negative-value" if pct_change < 0 else "neutral-value"
        
        if format_type == "percentage":
            primary_display = f"{primary_value:.1f} %"
            comp_display = f"{comparison_value:.1f} %"
        elif format_type == "number":
            primary_display = f"{prefix}{primary_value:,.0f}"
            comp_display = f"{prefix}{comparison_value:,.0f}"
        else:
            primary_display = f"{prefix}{primary_value:.2f}"
            comp_display = f"{prefix}{comparison_value:.2f}"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;">
                <div style="flex: 1; text-align: center;">
                    <div class="metric-value {primary_color}" style="font-size: 1.3rem;">{primary_display}</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">Primary</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div class="metric-value {comp_color}" style="font-size: 1.3rem;">{comp_display}</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">Comparison</div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div class="metric-value {pct_color}" style="font-size: 1.3rem;">{pct_change:+.1f}%</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">Change</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Original single metric display
        primary_color = "positive-value" if primary_value >= 0 else "negative-value" if primary_value < 0 else "neutral-value"
        if format_type == "percentage":
            primary_display = f"{primary_value:.1f} %"
        elif format_type == "number":
            primary_display = f"{prefix}{primary_value:,.0f}"
        else:
            primary_display = f"{prefix}{primary_value:.2f}"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value {primary_color}">{primary_display}</div>
        </div>
        """, unsafe_allow_html=True)


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


def render_individual_strategy(df, strategy, timeline_df, filtered_trades_df, selected_underlying, comp_df=None, comparison_enabled=False):
    """Render individual strategy view with professional styling"""
    st.markdown(f'<h3 class="subsection-header">📊 Strategy: {strategy}</h3>', unsafe_allow_html=True)
    
    if selected_underlying:
        st.markdown(f'<h4 class="subsection-header">🎯 Underlying: {selected_underlying}</h4>', unsafe_allow_html=True)
    
    # Helper function to calculate percentage change
    def calculate_pct_change(primary_value, comparison_value):
        """Calculate percentage change between primary and comparison values"""
        if comparison_value == 0:
            return 0 if primary_value == 0 else float('inf')
        return ((primary_value - comparison_value) / abs(comparison_value)) * 100


    
    # — P&L & risk metrics
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
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
    
    # — IB Commission metrics
    total_ib_commission = df["IBCommission"].sum() if "IBCommission" in df.columns else 0

    # Initialize comparison variables
    comp_nav_units = None
    comp_current_nav = None
    comp_max_nav = None
    comp_current_dd = None
    comp_max_dd = None
    comp_cagr = None
    
    # Calculate comparison metrics if comparison is enabled
    if comparison_enabled and comp_df is not None and not comp_df.empty:
        comp_df = comp_df.copy()
        comp_df["Date"] = pd.to_datetime(comp_df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        comp_daily_pnl = comp_df.groupby("Date")["NetCash"].sum().sort_index()
        comp_net_pnl = comp_daily_pnl.sum()
        comp_excess_ret = comp_daily_pnl - rf_daily
        comp_sharpe = comp_excess_ret.mean() / (comp_daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
        comp_wins = comp_daily_pnl[comp_daily_pnl > 0].sum()
        comp_losses = -comp_daily_pnl[comp_daily_pnl < 0].sum()
        comp_profit_fac = comp_wins / comp_losses if comp_losses else np.nan
        comp_win_days = int((comp_daily_pnl > 0).sum())
        comp_loss_days = int((comp_daily_pnl < 0).sum())
        
        # Comparison trade counts
        comp_tg = comp_df.groupby(["UID", "TradeDate"])
        comp_total, comp_win_t, comp_loss_t = 0, 0, 0
        for (_, _), legs in comp_tg:
            cnt = max(1, legs["Quantity"].max())
            comp_total += cnt
            s = legs["NetCash"].sum()
            if s > 0: comp_win_t += cnt
            elif s < 0: comp_loss_t += cnt
        
        # Comparison slippage metrics
        comp_total_slippage = comp_df["Slippage"].sum() if "Slippage" in comp_df.columns else 0
        comp_total_ib_commission = comp_df["IBCommission"].sum() if "IBCommission" in comp_df.columns else 0
        
        # Comparison weighted average slippage percentage
        if "Slippage_pct" in comp_df.columns and "Slippage" in comp_df.columns:
            comp_slippage_trades = comp_df[comp_df["Slippage"] > 0]
            if not comp_slippage_trades.empty:
                comp_weighted_avg = (comp_slippage_trades["Slippage"] * comp_slippage_trades["Slippage_pct"]).sum() / comp_slippage_trades["Slippage"].sum()
                comp_avg_slippage_pct = comp_weighted_avg
            else:
                comp_avg_slippage_pct = 0
        else:
            comp_avg_slippage_pct = 0
    else:
        comp_net_pnl = None
        comp_sharpe = None
        comp_wins = None
        comp_losses = None
        comp_profit_fac = None
        comp_win_days = None
        comp_loss_days = None
        comp_total = None
        comp_win_t = None
        comp_loss_t = None
        comp_total_slippage = None
        comp_total_ib_commission = None
        comp_avg_slippage_pct = None
        
        # NAV comparison metrics
        comp_current_nav = None
        comp_max_nav = None
        comp_current_dd = None
        comp_max_dd = None
        comp_cagr = None
        
        # Advanced performance metrics
        comp_avg_monthly_return = None
        comp_current_7d_return = None
        comp_current_30d_return = None
        comp_annualized_vol = None
        comp_sortino_ratio = None
        comp_return_over_dd = None
        
        # Risk metrics
        comp_avg_drawdown = None
        comp_max_dd_duration = None
        comp_var_95 = None
        comp_cvar_95 = None
        comp_win_rate = None
        comp_kelly = None
        
        # Trade efficiency metrics
        comp_avg_trade_return = None
        comp_avg_win_amount = None
        comp_avg_loss_amount = None
        comp_expectancy = None
        comp_rr_ratio = None
        comp_breakeven_win_pct = None
        
        # Cost metrics
        comp_cost_return_ratio = None
        comp_avg_cost_per_trade = None
        
        # Rolling metrics
        comp_current_7d_pnl = None
        comp_current_30d_pnl = None
        comp_current_7d_sharpe = None
        comp_current_30d_sharpe = None

    # — NAV & units series (using filtered trades for date range consistency)
    tl2 = timeline_df.copy()
    tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
    
    # Filter timeline to match the date range of filtered trades
    if not filtered_trades_df.empty:
        trade_dates = pd.to_datetime(filtered_trades_df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        min_date = trade_dates.min()
        max_date = trade_dates.max()
        tl2 = tl2[(tl2["Date"] >= min_date) & (tl2["Date"] <= max_date)]
    
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

    # Calculate comparison NAV metrics if comparison is enabled
    if comparison_enabled and comp_df is not None and not comp_df.empty:
        # Calculate comparison NAV using the same logic
        comp_tl2 = timeline_df.copy()
        comp_tl2["Date"] = pd.to_datetime(comp_tl2["Date"], dayfirst=True)
        
        # Filter timeline to match the comparison date range
        comp_trade_dates = pd.to_datetime(comp_df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        comp_min_date = comp_trade_dates.min()
        comp_max_date = comp_trade_dates.max()
        comp_tl2 = comp_tl2[(comp_tl2["Date"] >= comp_min_date) & (comp_tl2["Date"] <= comp_max_date)]
        
        comp_nav_units = calculate_strategy_nav_with_units(
            strategy, comp_df, comp_tl2, selected_underlying
        )
        
        if not comp_nav_units.empty:
            comp_nav_ser = comp_nav_units["nav"]
            comp_units_ser = comp_nav_units["units"]
            comp_active_nav = comp_nav_ser[comp_nav_ser > 0]
            
            if not comp_active_nav.empty:
                comp_current_nav = comp_active_nav.iloc[-1]
                comp_max_nav = comp_active_nav.max()
                comp_current_dd = (comp_active_nav/comp_active_nav.cummax()-1).iloc[-1]*100
                comp_max_dd = (comp_active_nav/comp_active_nav.cummax()-1).min()*100
                
                # Calculate comparison CAGR
                comp_start_nav = comp_active_nav.iloc[0]
                comp_end_nav = comp_active_nav.iloc[-1]
                comp_active_days = len(comp_active_nav) - 1
                comp_cagr = (comp_end_nav / comp_start_nav)**(252.0 / comp_active_days) - 1 if comp_active_days > 0 else 0
            else:
                comp_current_nav = None
                comp_max_nav = None
                comp_current_dd = None
                comp_max_dd = None
                comp_cagr = None
        else:
            comp_current_nav = None
            comp_max_nav = None
            comp_current_dd = None
            comp_max_dd = None
            comp_cagr = None
        
        # Calculate comparison advanced performance metrics
        if not comp_nav_units.empty:
            comp_nav_ser = comp_nav_units["nav"]
            comp_active_nav = comp_nav_ser[comp_nav_ser > 0]
            
            if not comp_active_nav.empty and len(comp_active_nav) > 7:
                comp_daily_returns = comp_active_nav.pct_change().dropna()
                
                # Monthly Returns
                comp_monthly_returns = comp_daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                comp_avg_monthly_return = comp_monthly_returns.mean() * 100 if not comp_monthly_returns.empty else 0
                
                # Rolling Returns (7d and 30d)
                comp_rolling_7d = comp_daily_returns.rolling(7).apply(lambda x: (1 + x).prod() - 1)
                comp_rolling_30d = comp_daily_returns.rolling(30).apply(lambda x: (1 + x).prod() - 1)
                comp_current_7d_return = comp_rolling_7d.iloc[-1] * 100 if not comp_rolling_7d.empty and not np.isnan(comp_rolling_7d.iloc[-1]) else 0
                comp_current_30d_return = comp_rolling_30d.iloc[-1] * 100 if not comp_rolling_30d.empty and not np.isnan(comp_rolling_30d.iloc[-1]) else 0
                
                # Annualized Volatility
                comp_annualized_vol = comp_daily_returns.std() * np.sqrt(252) * 100
                
                # Sortino Ratio
                comp_downside_returns = comp_daily_returns[comp_daily_returns < 0]
                comp_downside_vol = comp_downside_returns.std() * np.sqrt(252) if len(comp_downside_returns) > 0 else 1e-9
                comp_sortino_ratio = (comp_daily_returns.mean() * 252) / comp_downside_vol if comp_downside_vol > 0 else 0
                
                # Return Over Max Drawdown
                comp_max_dd_val = (comp_active_nav / comp_active_nav.cummax() - 1).min()
                comp_return_over_dd = (comp_cagr / abs(comp_max_dd_val)) if comp_max_dd_val != 0 else np.nan
                
                # Risk metrics
                comp_drawdown_series = comp_active_nav / comp_active_nav.cummax() - 1
                comp_drawdown_episodes = comp_drawdown_series[comp_drawdown_series < 0]
                comp_avg_drawdown = comp_drawdown_episodes.mean() * 100 if len(comp_drawdown_episodes) > 0 else 0
                
                # Drawdown Duration
                comp_peak_indices = (comp_drawdown_series == 0).astype(int)
                comp_recovery_periods = comp_peak_indices.groupby(comp_peak_indices.cumsum()).cumcount()
                comp_max_dd_duration = comp_recovery_periods.max()
                
                # Value at Risk (95% confidence)
                comp_var_95 = np.percentile(comp_daily_returns, 5) * 100
                
                # Expected Shortfall (CVaR)
                comp_cvar_95 = comp_daily_returns[comp_daily_returns <= np.percentile(comp_daily_returns, 5)].mean() * 100
                
                # Kelly Criterion
                comp_win_rate = len(comp_daily_returns[comp_daily_returns > 0]) / len(comp_daily_returns)
                comp_avg_win = comp_daily_returns[comp_daily_returns > 0].mean()
                comp_avg_loss = abs(comp_daily_returns[comp_daily_returns < 0].mean())
                comp_rr_ratio = comp_avg_win / comp_avg_loss if comp_avg_loss != 0 else 0
                comp_kelly = comp_win_rate - ((1 - comp_win_rate) / comp_rr_ratio) if comp_rr_ratio > 0 else 0
                
                # Trade efficiency metrics
                comp_total_trades = len(comp_df.groupby(["UID", "TradeDate"]))
                comp_avg_trade_return = comp_net_pnl / comp_total_trades if comp_total_trades > 0 else 0
                
                # Expectancy per Trade
                comp_winning_trades = comp_df[comp_df.groupby(["UID", "TradeDate"])["NetCash"].transform('sum') > 0]
                comp_losing_trades = comp_df[comp_df.groupby(["UID", "TradeDate"])["NetCash"].transform('sum') < 0]
                
                comp_avg_win_amount = comp_winning_trades.groupby(["UID", "TradeDate"])["NetCash"].sum().mean() if len(comp_winning_trades) > 0 else 0
                comp_avg_loss_amount = abs(comp_losing_trades.groupby(["UID", "TradeDate"])["NetCash"].sum().mean()) if len(comp_losing_trades) > 0 else 0
                
                comp_win_rate_trades = len(comp_winning_trades.groupby(["UID", "TradeDate"])) / comp_total_trades if comp_total_trades > 0 else 0
                comp_expectancy = (comp_win_rate_trades * comp_avg_win_amount) - ((1 - comp_win_rate_trades) * comp_avg_loss_amount)
                
                # Reward/Risk Ratio
                comp_rr_ratio_trades = comp_avg_win_amount / comp_avg_loss_amount if comp_avg_loss_amount > 0 else 0
                
                # Break-even Win %
                comp_breakeven_win_pct = 1 / (1 + comp_rr_ratio_trades) * 100 if comp_rr_ratio_trades > 0 else 0
                
                # Cost metrics
                comp_cost_return_ratio = (comp_total_slippage / comp_net_pnl * 100) if comp_net_pnl != 0 else 0
                comp_avg_cost_per_trade = comp_total_slippage / comp_total_trades if comp_total_trades > 0 else 0
                
                # Rolling metrics
                comp_daily_pnl = comp_df.groupby("Date")["NetCash"].sum().sort_index()
                comp_rolling_7d_pnl = comp_daily_pnl.rolling(7).sum()
                comp_rolling_30d_pnl = comp_daily_pnl.rolling(30).sum()
                comp_rolling_7d_avg = comp_daily_pnl.rolling(7).mean()
                comp_rolling_30d_avg = comp_daily_pnl.rolling(30).mean()
                comp_rolling_7d_std = comp_daily_pnl.rolling(7).std()
                comp_rolling_30d_std = comp_daily_pnl.rolling(30).std()
                comp_rolling_7d_sharpe = (comp_rolling_7d_avg / comp_rolling_7d_std) * np.sqrt(252)
                comp_rolling_30d_sharpe = (comp_rolling_30d_avg / comp_rolling_30d_std) * np.sqrt(252)
                comp_rolling_7d_sharpe = comp_rolling_7d_sharpe.fillna(0)
                comp_rolling_30d_sharpe = comp_rolling_30d_sharpe.fillna(0)
                
                comp_current_7d_pnl = comp_rolling_7d_pnl.iloc[-1] if not comp_rolling_7d_pnl.empty and not np.isnan(comp_rolling_7d_pnl.iloc[-1]) else 0
                comp_current_30d_pnl = comp_rolling_30d_pnl.iloc[-1] if not comp_rolling_30d_pnl.empty and not np.isnan(comp_rolling_30d_pnl.iloc[-1]) else 0
                comp_current_7d_sharpe = comp_rolling_7d_sharpe.iloc[-1] if not comp_rolling_7d_sharpe.empty and not np.isnan(comp_rolling_7d_sharpe.iloc[-1]) else 0
                comp_current_30d_sharpe = comp_rolling_30d_sharpe.iloc[-1] if not comp_rolling_30d_sharpe.empty and not np.isnan(comp_rolling_30d_sharpe.iloc[-1]) else 0
            else:
                # Set all advanced metrics to None if insufficient data
                comp_avg_monthly_return = None
                comp_current_7d_return = None
                comp_current_30d_return = None
                comp_annualized_vol = None
                comp_sortino_ratio = None
                comp_return_over_dd = None
                comp_avg_drawdown = None
                comp_max_dd_duration = None
                comp_var_95 = None
                comp_cvar_95 = None
                comp_win_rate = None
                comp_kelly = None
                comp_avg_trade_return = None
                comp_avg_win_amount = None
                comp_avg_loss_amount = None
                comp_expectancy = None
                comp_rr_ratio = None
                comp_breakeven_win_pct = None
                comp_cost_return_ratio = None
                comp_avg_cost_per_trade = None
                comp_current_7d_pnl = None
                comp_current_30d_pnl = None
                comp_current_7d_sharpe = None
                comp_current_30d_sharpe = None
        else:
            # Set all advanced metrics to None if no NAV data
            comp_avg_monthly_return = None
            comp_current_7d_return = None
            comp_current_30d_return = None
            comp_annualized_vol = None
            comp_sortino_ratio = None
            comp_return_over_dd = None
            comp_avg_drawdown = None
            comp_max_dd_duration = None
            comp_var_95 = None
            comp_cvar_95 = None
            comp_win_rate = None
            comp_kelly = None
            comp_avg_trade_return = None
            comp_avg_win_amount = None
            comp_avg_loss_amount = None
            comp_expectancy = None
            comp_rr_ratio = None
            comp_breakeven_win_pct = None
            comp_cost_return_ratio = None
            comp_avg_cost_per_trade = None
            comp_current_7d_pnl = None
            comp_current_30d_pnl = None
            comp_current_7d_sharpe = None
            comp_current_30d_sharpe = None

    # Display metrics in 4-column layout with comparison data embedded
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_comparison_metric("💵 Net P&L", net_pnl, comp_net_pnl, prefix="$")
        if not comparison_enabled or comp_net_pnl is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From trade NetCash data</div>', unsafe_allow_html=True)
        
        render_comparison_metric("📈 Sharpe Ratio", sharpe, comp_sharpe, prefix="")
        if not comparison_enabled or comp_sharpe is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Excess return / Standard deviation</div>', unsafe_allow_html=True)
        
        render_comparison_metric("⚖️ Profit Factor", profit_fac, comp_profit_fac, prefix="")
        if not comparison_enabled or comp_profit_fac is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total wins / Total losses</div>', unsafe_allow_html=True)
    
    with col2:
        # For win/loss days, we need to handle the combined display
        if comparison_enabled and comp_win_days is not None:
            win_loss_display = f"{win_days} / {loss_days}"
            comp_win_loss_display = f"{comp_win_days} / {comp_loss_days}"
            pct_change = calculate_pct_change(win_days, comp_win_days)
            color_class = "positive-value" if pct_change > 0 else "negative-value" if pct_change < 0 else "neutral-value"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Win/Loss Days</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;">
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value neutral-value" style="font-size: 1.3rem;">{win_loss_display}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Primary</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value neutral-value" style="font-size: 1.3rem;">{comp_win_loss_display}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Comparison</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value {color_class}" style="font-size: 1.3rem;">{pct_change:+.1f}%</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Change</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Win/Loss Days</div>
                <div class="metric-value neutral-value">{win_days} / {loss_days}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Trading days with positive/negative P&L</div>
            </div>
            """, unsafe_allow_html=True)
        
        render_comparison_metric("🔄 Total Trades", total, comp_total, prefix="")
        if not comparison_enabled or comp_total is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total number of trades executed</div>', unsafe_allow_html=True)
        
        # For win/loss trades, we need to handle the combined display
        if comparison_enabled and comp_win_t is not None:
            win_loss_trades_display = f"{win_t} / {loss_t}"
            comp_win_loss_trades_display = f"{comp_win_t} / {comp_loss_t}"
            pct_change = calculate_pct_change(win_t, comp_win_t)
            color_class = "positive-value" if pct_change > 0 else "negative-value" if pct_change < 0 else "neutral-value"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Win/Loss Trades</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;">
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value neutral-value" style="font-size: 1.3rem;">{win_loss_trades_display}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Primary</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value neutral-value" style="font-size: 1.3rem;">{comp_win_loss_trades_display}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Comparison</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value {color_class}" style="font-size: 1.3rem;">{pct_change:+.1f}%</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Change</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Win/Loss Trades</div>
                <div class="metric-value neutral-value">{win_t} / {loss_t}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Number of profitable/unprofitable trades</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        win_rate = (win_t/total*100) if total > 0 else 0
        comp_win_rate = (comp_win_t/comp_total*100) if comp_total and comp_total > 0 else 0
        render_comparison_metric("📊 Win Rate", win_rate, comp_win_rate, prefix="", format_type="percentage")
        if not comparison_enabled or comp_win_rate is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Percentage of profitable trades</div>', unsafe_allow_html=True)
        
        if has_nav_data and not nav_ser.empty:
            current_nav = nav_ser.iloc[-1]
            render_comparison_metric("📈 Current NAV", current_nav, None, prefix="")  # NAV comparison would need separate calculation
            if not comparison_enabled:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest NAV from allocation data</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Current NAV</div>
                <div class="metric-value neutral-value">N/A</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">No NAV data available</div>
            </div>
            """, unsafe_allow_html=True)
        
        render_comparison_metric("💰 Total Slippage", total_slippage, comp_total_slippage, prefix="$")
        if not comparison_enabled or comp_total_slippage is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all trade slippage costs</div>', unsafe_allow_html=True)
    
    with col4:
        render_comparison_metric("📊 Avg Slippage %", avg_slippage_pct, comp_avg_slippage_pct, prefix="", format_type="percentage")
        if not comparison_enabled or comp_avg_slippage_pct is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average slippage percentage per trade</div>', unsafe_allow_html=True)
        
        if has_nav_data and not units_ser.empty:
            current_units = units_ser.iloc[-1]
            render_comparison_metric("📊 Current Units", current_units, None, prefix="")  # Units comparison would need separate calculation
            if not comparison_enabled:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest units from allocation data</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Current Units</div>
                <div class="metric-value neutral-value">N/A</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">No units data available</div>
            </div>
            """, unsafe_allow_html=True)
        
        render_comparison_metric("🏦 IB Commission", total_ib_commission, comp_total_ib_commission, prefix="$")
        if not comparison_enabled or comp_total_ib_commission is None:
            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total IB commission paid</div>', unsafe_allow_html=True)

    # — NAV metrics & CAGR
    if has_nav_data:
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            start_nav = active_nav.iloc[0]  # now the prepended 100
            end_nav   = active_nav.iloc[-1]
            # Calculate only active trading days (when NAV data exists)
            active_days = len(active_nav) - 1  # Number of active trading days
            cagr      = (end_nav / start_nav)**(252.0 / active_days) - 1

            st.markdown('<h3 class="subsection-header">📈 NAV Performance Metrics</h3>', unsafe_allow_html=True)
            
            d1, d2, d3, d4 = st.columns(4)
            
            with d1:
                render_comparison_metric("🏆 Max NAV", active_nav.max(), comp_max_nav, prefix="")
                if not comparison_enabled or comp_max_nav is None:
                    st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Highest NAV achieved</div>', unsafe_allow_html=True)
            
            with d2:
                current_dd = (active_nav/active_nav.cummax()-1).iloc[-1]*100
                render_comparison_metric("📉 Curr Drawdown", current_dd, comp_current_dd, prefix="", format_type="percentage")
                if not comparison_enabled or comp_current_dd is None:
                    st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Current decline from peak NAV</div>', unsafe_allow_html=True)
            
            with d3:
                max_dd = (active_nav/active_nav.cummax()-1).min()*100
                render_comparison_metric("📉 Max Drawdown", max_dd, comp_max_dd, prefix="", format_type="percentage")
                if not comparison_enabled or comp_max_dd is None:
                    st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Largest decline from peak NAV</div>', unsafe_allow_html=True)
            
            with d4:
                render_comparison_metric("📊 CAGR", cagr*100, comp_cagr*100 if comp_cagr is not None else None, prefix="", format_type="percentage")
                if not comparison_enabled or comp_cagr is None:
                    st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Compound annual growth rate</div>', unsafe_allow_html=True)
        else:
            st.info("No NAV data available for this strategy.")
    else:
        st.info("No allocation data available for NAV calculation.")

    # — charts
    st.markdown('<h3 class="subsection-header">📊 Performance Charts</h3>', unsafe_allow_html=True)
    
    st.markdown('<h4 class="subsection-header">📈 Cumulative P&L</h4>', unsafe_allow_html=True)
    
    # Create Cumulative P&L chart with comparison if enabled
    if comparison_enabled and comp_df is not None and not comp_df.empty:
        # Calculate comparison daily P&L
        comp_daily_pnl = comp_df.groupby("Date")["NetCash"].sum().sort_index()
        
        # Create figure with both series
        fig = go.Figure()
        
        # Primary period P&L
        fig.add_trace(go.Scatter(
            x=daily_pnl.cumsum().index,
            y=daily_pnl.cumsum().values,
            mode='lines',
            name=f'Primary Period',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Comparison period P&L
        fig.add_trace(go.Scatter(
            x=comp_daily_pnl.cumsum().index,
            y=comp_daily_pnl.cumsum().values,
            mode='lines',
            name=f'Comparison Period',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Cumulative P&L: {strategy}",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Set y-axis range based on combined data
        all_data = pd.concat([daily_pnl.cumsum(), comp_daily_pnl.cumsum()])
        if pd.notna(all_data.min()) and pd.notna(all_data.max()):
            y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
            fig.update_layout(yaxis=dict(range=y_range))
    else:
        # Original single period chart
        fig = px.line(
            daily_pnl.cumsum(),
            title=f"Cumulative P&L: {strategy}",
            labels={"value": "Cumulative P&L", "index": "Date"}
        )
        fig.update_layout(height=400, showlegend=False)
        
        # Set y-axis range based on data
        cumsum_data = daily_pnl.cumsum()
        if pd.notna(cumsum_data.min()) and pd.notna(cumsum_data.max()):
            y_range = [cumsum_data.min() * 0.95, cumsum_data.max() * 1.05]
            fig.update_layout(yaxis=dict(range=y_range))
    
    st.plotly_chart(fig, use_container_width=True)

    if has_nav_data and not nav_ser.empty:
        active_nav = nav_ser[nav_ser > 0]
        if not active_nav.empty:
            st.markdown('<h4 class="subsection-header">📈 NAV vs Time</h4>', unsafe_allow_html=True)
            
            # Create NAV chart with comparison if enabled
            if comparison_enabled and comp_nav_units is not None and not comp_nav_units.empty:
                comp_nav_ser = comp_nav_units["nav"]
                comp_active_nav = comp_nav_ser[comp_nav_ser > 0]
                
                if not comp_active_nav.empty:
                    # Create figure with both series
                    fig = go.Figure()
                    
                    # Primary period NAV
                    fig.add_trace(go.Scatter(
                        x=active_nav.index,
                        y=active_nav.values,
                        mode='lines',
                        name=f'Primary Period',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Comparison period NAV
                    fig.add_trace(go.Scatter(
                        x=comp_active_nav.index,
                        y=comp_active_nav.values,
                        mode='lines',
                        name=f'Comparison Period',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"NAV Performance: {strategy}",
                        xaxis_title="Date",
                        yaxis_title="NAV",
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set y-axis range based on combined data
                    all_data = pd.concat([active_nav, comp_active_nav])
                    if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                        y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                        fig.update_layout(yaxis=dict(range=y_range))
                else:
                    # Fallback to original chart if no comparison NAV data
                    fig = px.line(
                        active_nav,
                        title=f"NAV Performance: {strategy}",
                        labels={"value": "NAV", "index": "Date"}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    
                    # Set y-axis range based on data
                    if pd.notna(active_nav.min()) and pd.notna(active_nav.max()):
                        y_range = [active_nav.min() * 0.95, active_nav.max() * 1.05]
                        fig.update_layout(yaxis=dict(range=y_range))
            else:
                # Original single period chart
                fig = px.line(
                    active_nav,
                    title=f"NAV Performance: {strategy}",
                    labels={"value": "NAV", "index": "Date"}
                )
                fig.update_layout(height=400, showlegend=False)
                
                # Set y-axis range based on data
                if pd.notna(active_nav.min()) and pd.notna(active_nav.max()):
                    y_range = [active_nav.min() * 0.95, active_nav.max() * 1.05]
                    fig.update_layout(yaxis=dict(range=y_range))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling Drawdown Chart
            st.markdown('<h4 class="subsection-header">📉 Rolling Drawdown</h4>', unsafe_allow_html=True)
            drawdown = (active_nav / active_nav.cummax() - 1) * 100
            
            # Create Drawdown chart with comparison if enabled
            if comparison_enabled and comp_nav_units is not None and not comp_nav_units.empty:
                comp_nav_ser = comp_nav_units["nav"]
                comp_active_nav = comp_nav_ser[comp_nav_ser > 0]
                
                if not comp_active_nav.empty:
                    comp_drawdown = (comp_active_nav / comp_active_nav.cummax() - 1) * 100
                    
                    # Create figure with both series
                    fig = go.Figure()
                    
                    # Primary period drawdown
                    fig.add_trace(go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        mode='lines',
                        name=f'Primary Period',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Comparison period drawdown
                    fig.add_trace(go.Scatter(
                        x=comp_drawdown.index,
                        y=comp_drawdown.values,
                        mode='lines',
                        name=f'Comparison Period',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Rolling Drawdown: {strategy}",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set y-axis range based on combined data
                    all_data = pd.concat([drawdown, comp_drawdown])
                    if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                        y_range = [all_data.min() * 1.05, all_data.max() * 0.95]
                        fig.update_layout(yaxis=dict(range=y_range))
                else:
                    # Fallback to original chart if no comparison NAV data
                    fig = px.line(
                        drawdown,
                        title=f"Rolling Drawdown: {strategy}",
                        labels={"value": "Drawdown (%)", "index": "Date"}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_traces(line_color='red')
                    
                    # Set y-axis range based on data
                    if pd.notna(drawdown.min()) and pd.notna(drawdown.max()):
                        y_range = [drawdown.min() * 1.05, drawdown.max() * 0.95]
                        fig.update_layout(yaxis=dict(range=y_range))
            else:
                # Original single period chart
                fig = px.line(
                    drawdown,
                    title=f"Rolling Drawdown: {strategy}",
                    labels={"value": "Drawdown (%)", "index": "Date"}
                )
                fig.update_layout(height=400, showlegend=False)
                fig.update_traces(line_color='red')
                
                # Set y-axis range based on data
                if pd.notna(drawdown.min()) and pd.notna(drawdown.max()):
                    y_range = [drawdown.min() * 1.05, drawdown.max() * 0.95]
                    fig.update_layout(yaxis=dict(range=y_range))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<h4 class="subsection-header">📊 Units vs Time</h4>', unsafe_allow_html=True)
            
            # Create Units chart with comparison if enabled
            if comparison_enabled and comp_nav_units is not None and not comp_nav_units.empty:
                comp_units_ser = comp_nav_units["units"]
                
                if not comp_units_ser.empty:
                    # Create figure with both series
                    fig = go.Figure()
                    
                    # Primary period units
                    fig.add_trace(go.Scatter(
                        x=units_ser.index,
                        y=units_ser.values,
                        mode='lines',
                        name=f'Primary Period',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Comparison period units
                    fig.add_trace(go.Scatter(
                        x=comp_units_ser.index,
                        y=comp_units_ser.values,
                        mode='lines',
                        name=f'Comparison Period',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Units Held: {strategy}",
                        xaxis_title="Date",
                        yaxis_title="Units",
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set y-axis range based on combined data
                    all_data = pd.concat([units_ser, comp_units_ser])
                    if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                        y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                        fig.update_layout(yaxis=dict(range=y_range))
                else:
                    # Fallback to original chart if no comparison units data
                    fig = px.line(
                        units_ser,
                        title=f"Units Held: {strategy}",
                        labels={"value": "Units", "index": "Date"}
                    )
                    fig.update_layout(height=400, showlegend=False)
            else:
                # Original single period chart
                fig = px.line(
                    units_ser,
                    title=f"Units Held: {strategy}",
                    labels={"value": "Units", "index": "Date"}
                )
                fig.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Performance Metrics for Individual Strategy
            with st.expander("📊 Advanced Performance Metrics", expanded=False):
                st.markdown('<h4 class="subsection-header">📈 Return-Based Metrics</h4>', unsafe_allow_html=True)
                
                # Calculate advanced metrics using NAV data
                daily_returns = active_nav.pct_change().dropna()
                
                # Check if we have sufficient data for calculations
                if len(daily_returns) < 7:
                    st.info("Insufficient data for advanced metrics in selected period.")
                else:
                    # Monthly Returns
                    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                    avg_monthly_return = monthly_returns.mean() * 100 if not monthly_returns.empty else 0
                    
                    # Rolling Returns (7d and 30d)
                    rolling_7d = daily_returns.rolling(7).apply(lambda x: (1 + x).prod() - 1)
                    rolling_30d = daily_returns.rolling(30).apply(lambda x: (1 + x).prod() - 1)
                    current_7d_return = rolling_7d.iloc[-1] * 100 if not rolling_7d.empty and not np.isnan(rolling_7d.iloc[-1]) else 0
                    current_30d_return = rolling_30d.iloc[-1] * 100 if not rolling_30d.empty and not np.isnan(rolling_30d.iloc[-1]) else 0
                    
                    # Annualized Volatility
                    annualized_vol = daily_returns.std() * np.sqrt(252) * 100
                    
                    # Sortino Ratio
                    downside_returns = daily_returns[daily_returns < 0]
                    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
                    sortino_ratio = (daily_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
                    
                    # Return Over Max Drawdown
                    max_dd = (active_nav / active_nav.cummax() - 1).min()
                    return_over_dd = (cagr / abs(max_dd)) if max_dd != 0 else np.nan
                
                    # Display Return-Based Metrics
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        render_comparison_metric("📊 Avg Monthly Return", avg_monthly_return, comp_avg_monthly_return, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_avg_monthly_return is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average of monthly returns from NAV data</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("📈 7-Day Rolling Return", current_7d_return, comp_current_7d_return, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_current_7d_return is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day rolling return from NAV data</div>', unsafe_allow_html=True)
                    with col_r2:
                        render_comparison_metric("📈 30-Day Rolling Return", current_30d_return, comp_current_30d_return, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_current_30d_return is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day rolling return from NAV data</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("📊 Annualized Volatility", annualized_vol, comp_annualized_vol, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_annualized_vol is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily returns std × √252</div>', unsafe_allow_html=True)
                    with col_r3:
                        render_comparison_metric("📈 Sortino Ratio", sortino_ratio, comp_sortino_ratio, prefix="")
                        if not comparison_enabled or comp_sortino_ratio is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Return / Downside volatility</div>', unsafe_allow_html=True)
                        
                        return_over_dd_display = f"{return_over_dd:.2f}" if not np.isnan(return_over_dd) else "N/A"
                        comp_return_over_dd_display = f"{comp_return_over_dd:.2f}" if comp_return_over_dd is not None and not np.isnan(comp_return_over_dd) else "N/A"
                        render_comparison_metric("📊 Return/Max DD", return_over_dd, comp_return_over_dd, prefix="")
                        if not comparison_enabled or comp_return_over_dd is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">CAGR / Maximum drawdown</div>', unsafe_allow_html=True)
                
                    st.markdown('<h4 class="subsection-header">🛡️ Risk Metrics</h4>', unsafe_allow_html=True)
                    
                    # Calculate risk metrics
                    drawdown_series = active_nav / active_nav.cummax() - 1
                    drawdown_episodes = drawdown_series[drawdown_series < 0]
                    avg_drawdown = drawdown_episodes.mean() * 100 if len(drawdown_episodes) > 0 else 0
                    
                    # Drawdown Duration (simplified)
                    peak_indices = (drawdown_series == 0).astype(int)
                    recovery_periods = peak_indices.groupby(peak_indices.cumsum()).cumcount()
                    max_dd_duration = recovery_periods.max()
                    
                    # Value at Risk (95% confidence)
                    var_95 = np.percentile(daily_returns, 5) * 100
                    
                    # Expected Shortfall (CVaR)
                    cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
                    
                    # Kelly Criterion (simplified)
                    win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
                    avg_win = daily_returns[daily_returns > 0].mean()
                    avg_loss = abs(daily_returns[daily_returns < 0].mean())
                    rr_ratio = avg_win / avg_loss if avg_loss != 0 else 0
                    kelly = win_rate - ((1 - win_rate) / rr_ratio) if rr_ratio > 0 else 0
                    
                    # Display Risk Metrics
                    col_risk1, col_risk2, col_risk3 = st.columns(3)
                    with col_risk1:
                        render_comparison_metric("📉 Average Drawdown", avg_drawdown, comp_avg_drawdown, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_avg_drawdown is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average of all drawdown periods</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("⏱️ Max DD Duration", max_dd_duration, comp_max_dd_duration, prefix="")
                        if not comparison_enabled or comp_max_dd_duration is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Longest recovery period</div>', unsafe_allow_html=True)
                    with col_risk2:
                        render_comparison_metric("⚠️ VaR (95%)", var_95, comp_var_95, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_var_95 is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">95% Value at Risk from daily returns</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("⚠️ CVaR (95%)", cvar_95, comp_cvar_95, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_cvar_95 is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected shortfall at 95% confidence</div>', unsafe_allow_html=True)
                    with col_risk3:
                        render_comparison_metric("🎯 Kelly Criterion", kelly, comp_kelly, prefix="")
                        if not comparison_enabled or comp_kelly is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Optimal leverage calculation</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("📊 Win Rate", win_rate*100, comp_win_rate*100 if comp_win_rate is not None else None, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_win_rate is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Percentage of positive daily returns</div>', unsafe_allow_html=True)
                    
                    st.markdown('<h4 class="subsection-header">📊 Trade Efficiency Metrics</h4>', unsafe_allow_html=True)
                
                    # Calculate trade efficiency metrics using existing logic
                    total_trades = len(df.groupby(["UID", "TradeDate"]))
                    avg_trade_return = net_pnl / total_trades if total_trades > 0 else 0
                    
                    # Expectancy per Trade
                    winning_trades = df[df.groupby(["UID", "TradeDate"])["NetCash"].transform('sum') > 0]
                    losing_trades = df[df.groupby(["UID", "TradeDate"])["NetCash"].transform('sum') < 0]
                    
                    avg_win_amount = winning_trades.groupby(["UID", "TradeDate"])["NetCash"].sum().mean() if len(winning_trades) > 0 else 0
                    avg_loss_amount = abs(losing_trades.groupby(["UID", "TradeDate"])["NetCash"].sum().mean()) if len(losing_trades) > 0 else 0
                    
                    win_rate_trades = len(winning_trades.groupby(["UID", "TradeDate"])) / total_trades if total_trades > 0 else 0
                    expectancy = (win_rate_trades * avg_win_amount) - ((1 - win_rate_trades) * avg_loss_amount)
                    
                    # Reward/Risk Ratio
                    rr_ratio_trades = avg_win_amount / avg_loss_amount if avg_loss_amount > 0 else 0
                    
                    # Break-even Win %
                    breakeven_win_pct = 1 / (1 + rr_ratio_trades) * 100 if rr_ratio_trades > 0 else 0
                    
                    # Display Trade Efficiency Metrics
                    col_eff1, col_eff2, col_eff3 = st.columns(3)
                    with col_eff1:
                        render_comparison_metric("📈 Avg Trade Return", avg_trade_return, comp_avg_trade_return, prefix="$")
                        if not comparison_enabled or comp_avg_trade_return is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Net P&L / Total trades</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("📊 Expectancy per Trade", expectancy, comp_expectancy, prefix="$")
                        if not comparison_enabled or comp_expectancy is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected value per trade</div>', unsafe_allow_html=True)
                    with col_eff2:
                        render_comparison_metric("⚖️ Reward/Risk Ratio", rr_ratio_trades, comp_rr_ratio, prefix="")
                        if not comparison_enabled or comp_rr_ratio is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Avg Win / Avg Loss</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("🎯 Break-even Win %", breakeven_win_pct, comp_breakeven_win_pct, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_breakeven_win_pct is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win rate needed to break even</div>', unsafe_allow_html=True)
                    with col_eff3:
                        render_comparison_metric("💰 Avg Win Amount", avg_win_amount, comp_avg_win_amount, prefix="$")
                        if not comparison_enabled or comp_avg_win_amount is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average profit per winning trade</div>', unsafe_allow_html=True)
                        
                        render_comparison_metric("💸 Avg Loss Amount", avg_loss_amount, comp_avg_loss_amount, prefix="$")
                        if not comparison_enabled or comp_avg_loss_amount is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average loss per losing trade</div>', unsafe_allow_html=True)
                    
                    st.markdown('<h4 class="subsection-header">🧮 Cost Metrics</h4>', unsafe_allow_html=True)
                    
                    # Calculate cost metrics
                    cost_return_ratio = (total_slippage / net_pnl * 100) if net_pnl != 0 else 0
                    avg_cost_per_trade = total_slippage / total_trades if total_trades > 0 else 0
                    
                    # Display Cost Metrics
                    col_cost1, col_cost2, col_cost3 = st.columns(3)
                    with col_cost1:
                        render_comparison_metric("💰 Total Slippage", total_slippage, comp_total_slippage, prefix="$")
                        if not comparison_enabled or comp_total_slippage is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all trade slippage costs</div>', unsafe_allow_html=True)
                    
                    with col_cost2:
                        render_comparison_metric("📊 Cost/Return Ratio", cost_return_ratio, comp_cost_return_ratio, prefix="", format_type="percentage")
                        if not comparison_enabled or comp_cost_return_ratio is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Slippage / Net P&L × 100</div>', unsafe_allow_html=True)
                    
                    with col_cost3:
                        render_comparison_metric("💸 Avg Cost per Trade", avg_cost_per_trade, comp_avg_cost_per_trade, prefix="$")
                        if not comparison_enabled or comp_avg_cost_per_trade is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total slippage / Total trades</div>', unsafe_allow_html=True)
                
                # Rolling Performance Charts
                st.markdown('<h4 class="subsection-header">📈 Rolling Performance Charts</h4>', unsafe_allow_html=True)
                
                # Calculate rolling metrics
                rolling_7d_pnl = daily_pnl.rolling(7).sum()
                rolling_30d_pnl = daily_pnl.rolling(30).sum()
                rolling_7d_avg = daily_pnl.rolling(7).mean()
                rolling_30d_avg = daily_pnl.rolling(30).mean()
                
                # Calculate rolling Sharpe
                rolling_7d_std = daily_pnl.rolling(7).std()
                rolling_30d_std = daily_pnl.rolling(30).std()
                rolling_7d_sharpe = (rolling_7d_avg / rolling_7d_std) * np.sqrt(252)
                rolling_30d_sharpe = (rolling_30d_avg / rolling_30d_std) * np.sqrt(252)
                # Handle division by zero by filling NaN values with 0
                rolling_7d_sharpe = rolling_7d_sharpe.fillna(0)
                rolling_30d_sharpe = rolling_30d_sharpe.fillna(0)
                
                # Display rolling charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("7-Day Rolling P&L")
                    if len(rolling_7d_pnl.dropna()) > 0:
                        # Create 7-day rolling P&L chart with comparison if enabled
                        if comparison_enabled and comp_df is not None and not comp_df.empty:
                            # Filter comparison trades for this specific strategy
                            comp_strategy_trades = comp_df[comp_df["Strategy"] == strategy]
                            if not comp_strategy_trades.empty:
                                # Calculate comparison rolling P&L
                                comp_daily_pnl = comp_strategy_trades.groupby("TradeDate")["NetCash"].sum()
                                comp_daily_pnl.index = pd.to_datetime(comp_daily_pnl.index, format="%d/%m/%y", dayfirst=True)
                                comp_daily_pnl = comp_daily_pnl.sort_index()
                                comp_rolling_7d_pnl = comp_daily_pnl.rolling(7).sum()
                                
                                # Create figure with both series
                                fig = go.Figure()
                                
                                # Primary period rolling P&L
                                fig.add_trace(go.Scatter(
                                    x=rolling_7d_pnl.index,
                                    y=rolling_7d_pnl.values,
                                    mode='lines',
                                    name=f'Primary Period',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Comparison period rolling P&L
                                fig.add_trace(go.Scatter(
                                    x=comp_rolling_7d_pnl.index,
                                    y=comp_rolling_7d_pnl.values,
                                    mode='lines',
                                    name=f'Comparison Period',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="7-Day Rolling P&L",
                                    xaxis_title="Date",
                                    yaxis_title="7-Day Rolling P&L",
                                    height=300,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Set y-axis range based on combined data
                                all_data = pd.concat([rolling_7d_pnl, comp_rolling_7d_pnl])
                                if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                    y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                    fig.update_layout(yaxis=dict(range=y_range))
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # No comparison data for this strategy
                                st.line_chart(rolling_7d_pnl, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_7d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling P&L chart")
                    
                    st.subheader("7-Day Rolling Sharpe")
                    if len(rolling_7d_sharpe.dropna()) > 0:
                        # Create 7-day rolling Sharpe chart with comparison if enabled
                        if comparison_enabled and comp_df is not None and not comp_df.empty:
                            # Filter comparison trades for this specific strategy
                            comp_strategy_trades = comp_df[comp_df["Strategy"] == strategy]
                            if not comp_strategy_trades.empty:
                                # Calculate comparison rolling Sharpe
                                comp_daily_pnl = comp_strategy_trades.groupby("TradeDate")["NetCash"].sum()
                                comp_daily_pnl.index = pd.to_datetime(comp_daily_pnl.index, format="%d/%m/%y", dayfirst=True)
                                comp_daily_pnl = comp_daily_pnl.sort_index()
                                comp_rolling_7d_avg = comp_daily_pnl.rolling(7).mean()
                                comp_rolling_7d_std = comp_daily_pnl.rolling(7).std()
                                comp_rolling_7d_sharpe = (comp_rolling_7d_avg / comp_rolling_7d_std) * np.sqrt(252)
                                comp_rolling_7d_sharpe = comp_rolling_7d_sharpe.fillna(0)
                                
                                # Create figure with both series
                                fig = go.Figure()
                                
                                # Primary period rolling Sharpe
                                fig.add_trace(go.Scatter(
                                    x=rolling_7d_sharpe.index,
                                    y=rolling_7d_sharpe.values,
                                    mode='lines',
                                    name=f'Primary Period',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Comparison period rolling Sharpe
                                fig.add_trace(go.Scatter(
                                    x=comp_rolling_7d_sharpe.index,
                                    y=comp_rolling_7d_sharpe.values,
                                    mode='lines',
                                    name=f'Comparison Period',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="7-Day Rolling Sharpe",
                                    xaxis_title="Date",
                                    yaxis_title="7-Day Rolling Sharpe",
                                    height=300,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Set y-axis range based on combined data
                                all_data = pd.concat([rolling_7d_sharpe, comp_rolling_7d_sharpe])
                                if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                    y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                    fig.update_layout(yaxis=dict(range=y_range))
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # No comparison data for this strategy
                                st.line_chart(rolling_7d_sharpe, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_7d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling Sharpe chart")
                
                with col_chart2:
                    st.subheader("30-Day Rolling P&L")
                    if len(rolling_30d_pnl.dropna()) > 0:
                        # Create 30-day rolling P&L chart with comparison if enabled
                        if comparison_enabled and comp_df is not None and not comp_df.empty:
                            # Filter comparison trades for this specific strategy
                            comp_strategy_trades = comp_df[comp_df["Strategy"] == strategy]
                            if not comp_strategy_trades.empty:
                                # Calculate comparison rolling P&L
                                comp_daily_pnl = comp_strategy_trades.groupby("TradeDate")["NetCash"].sum()
                                comp_daily_pnl.index = pd.to_datetime(comp_daily_pnl.index, format="%d/%m/%y", dayfirst=True)
                                comp_daily_pnl = comp_daily_pnl.sort_index()
                                comp_rolling_30d_pnl = comp_daily_pnl.rolling(30).sum()
                                
                                # Create figure with both series
                                fig = go.Figure()
                                
                                # Primary period rolling P&L
                                fig.add_trace(go.Scatter(
                                    x=rolling_30d_pnl.index,
                                    y=rolling_30d_pnl.values,
                                    mode='lines',
                                    name=f'Primary Period',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Comparison period rolling P&L
                                fig.add_trace(go.Scatter(
                                    x=comp_rolling_30d_pnl.index,
                                    y=comp_rolling_30d_pnl.values,
                                    mode='lines',
                                    name=f'Comparison Period',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="30-Day Rolling P&L",
                                    xaxis_title="Date",
                                    yaxis_title="30-Day Rolling P&L",
                                    height=300,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Set y-axis range based on combined data
                                all_data = pd.concat([rolling_30d_pnl, comp_rolling_30d_pnl])
                                if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                    y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                    fig.update_layout(yaxis=dict(range=y_range))
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # No comparison data for this strategy
                                st.line_chart(rolling_30d_pnl, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_30d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling P&L chart")
                    
                    st.subheader("30-Day Rolling Sharpe")
                    if len(rolling_30d_sharpe.dropna()) > 0:
                        # Create 30-day rolling Sharpe chart with comparison if enabled
                        if comparison_enabled and comp_df is not None and not comp_df.empty:
                            # Filter comparison trades for this specific strategy
                            comp_strategy_trades = comp_df[comp_df["Strategy"] == strategy]
                            if not comp_strategy_trades.empty:
                                # Calculate comparison rolling Sharpe
                                comp_daily_pnl = comp_strategy_trades.groupby("TradeDate")["NetCash"].sum()
                                comp_daily_pnl.index = pd.to_datetime(comp_daily_pnl.index, format="%d/%m/%y", dayfirst=True)
                                comp_daily_pnl = comp_daily_pnl.sort_index()
                                comp_rolling_30d_avg = comp_daily_pnl.rolling(30).mean()
                                comp_rolling_30d_std = comp_daily_pnl.rolling(30).std()
                                comp_rolling_30d_sharpe = (comp_rolling_30d_avg / comp_rolling_30d_std) * np.sqrt(252)
                                comp_rolling_30d_sharpe = comp_rolling_30d_sharpe.fillna(0)
                                
                                # Create figure with both series
                                fig = go.Figure()
                                
                                # Primary period rolling Sharpe
                                fig.add_trace(go.Scatter(
                                    x=rolling_30d_sharpe.index,
                                    y=rolling_30d_sharpe.values,
                                    mode='lines',
                                    name=f'Primary Period',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                
                                # Comparison period rolling Sharpe
                                fig.add_trace(go.Scatter(
                                    x=comp_rolling_30d_sharpe.index,
                                    y=comp_rolling_30d_sharpe.values,
                                    mode='lines',
                                    name=f'Comparison Period',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="30-Day Rolling Sharpe",
                                    xaxis_title="Date",
                                    yaxis_title="30-Day Rolling Sharpe",
                                    height=300,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Set y-axis range based on combined data
                                all_data = pd.concat([rolling_30d_sharpe, comp_rolling_30d_sharpe])
                                if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                    y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                    fig.update_layout(yaxis=dict(range=y_range))
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # No comparison data for this strategy
                                st.line_chart(rolling_30d_sharpe, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_30d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling Sharpe chart")
                
                # Display current rolling metrics only if we have sufficient data
                if len(daily_pnl) > 7:  # Need at least 7 days for rolling calculations
                    col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                    with col_roll1:
                        current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty and not np.isnan(rolling_7d_pnl.iloc[-1]) else 0
                        render_comparison_metric("📈 7-Day P&L", current_7d_pnl, comp_current_7d_pnl, prefix="$")
                        if not comparison_enabled or comp_current_7d_pnl is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative P&L</div>', unsafe_allow_html=True)
                    
                    with col_roll2:
                        current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty and not np.isnan(rolling_30d_pnl.iloc[-1]) else 0
                        render_comparison_metric("📈 30-Day P&L", current_30d_pnl, comp_current_30d_pnl, prefix="$")
                        if not comparison_enabled or comp_current_30d_pnl is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative P&L</div>', unsafe_allow_html=True)
                    
                    with col_roll3:
                        current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty and not np.isnan(rolling_7d_sharpe.iloc[-1]) else 0
                        render_comparison_metric("📊 7-Day Sharpe", current_7d_sharpe, comp_current_7d_sharpe, prefix="")
                        if not comparison_enabled or comp_current_7d_sharpe is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day rolling Sharpe ratio</div>', unsafe_allow_html=True)
                    
                    with col_roll4:
                        current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty and not np.isnan(rolling_30d_sharpe.iloc[-1]) else 0
                        render_comparison_metric("📊 30-Day Sharpe", current_30d_sharpe, comp_current_30d_sharpe, prefix="")
                        if not comparison_enabled or comp_current_30d_sharpe is None:
                            st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day rolling Sharpe ratio</div>', unsafe_allow_html=True)
                else:
                    st.info("Insufficient data for rolling metrics in selected period.")
        else:
            st.info("No NAV data to display.")
    else:
        st.info("No NAV charts available - allocation data required.")


def render_consolidated_view(full_trades_df, strategies, timeline_df, filtered_trades_df=None, comp_trades=None, comparison_enabled=False):
    """Render consolidated view with professional styling"""
    st.markdown('<h2 class="section-header">📊 Consolidated Strategy Performance</h2>', unsafe_allow_html=True)
    
    # Use filtered trades if provided, otherwise use full trades
    trades_df = filtered_trades_df if filtered_trades_df is not None else full_trades_df
    
    rows = []
    
    for strategy in strategies:
        df = trades_df[trades_df["Strategy"] == strategy].copy()
        if df.empty:
            continue

        # — P&L & risk metrics (same as individual view)
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
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
        
        # — IB Commission metrics
        total_ib_commission = df["IBCommission"].sum() if "IBCommission" in df.columns else 0

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
                
                # Calculate drawdown metrics
                drawdown_series = (active_nav / active_nav.cummax() - 1) * 100
                current_drawdown = drawdown_series.iloc[-1]
                max_drawdown = drawdown_series.min()
                avg_drawdown = drawdown_series[drawdown_series < 0].mean() if len(drawdown_series[drawdown_series < 0]) > 0 else 0
            else:
                cagr = np.nan
                current_nav = np.nan
                current_units = np.nan
                portfolio_value = np.nan
                current_drawdown = np.nan
                max_drawdown = np.nan
                avg_drawdown = np.nan

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
            "Current DD %": f"{current_drawdown:.2f} %" if not np.isnan(current_drawdown) else "N/A",
            "Max DD %": f"{max_drawdown:.2f} %" if not np.isnan(max_drawdown) else "N/A",
            "Avg DD %": f"{avg_drawdown:.2f} %" if not np.isnan(avg_drawdown) else "N/A",
            "Total Slippage": f"{total_slippage:,.0f}",
            "Avg Slippage %": f"{avg_slippage_pct:.2f} %",
            "IB Commission": f"${total_ib_commission:,.2f}",
        })

    if rows:
        dfc = pd.DataFrame(rows)
        st.markdown('<h3 class="subsection-header">📋 Strategy Performance Summary</h3>', unsafe_allow_html=True)
        st.dataframe(dfc, use_container_width=True)

        # — Summary metrics across all strategies
        st.markdown('<h3 class="subsection-header">📊 Summary Across All Strategies</h3>', unsafe_allow_html=True)
        
        # Calculate totals and averages
        # Calculate total P&L from raw values to avoid rounding errors
        total_pnl = 0
        for strategy in strategies:
            df = trades_df[trades_df["Strategy"] == strategy].copy()
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
            net_pnl = daily_pnl.sum()
            total_pnl += net_pnl
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
        
        # Calculate total IB Commission
        total_ib_commission = sum(float(row["IB Commission"].replace("$", "").replace(",", "")) for row in rows if row["IB Commission"] != "$0.00")
        
        # Calculate summary metrics
        total_strategies = len(strategies)
        
        # Calculate active strategies (those with trades on the last trade date)
        active_strategies = 0
        last_trade_date = None
        
        # First, find the last trade date across all strategies
        for strategy in strategies:
            strategy_trades = trades_df[trades_df["Strategy"] == strategy].copy()
            if not strategy_trades.empty:
                strategy_trades["Date"] = pd.to_datetime(strategy_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                max_date = strategy_trades["Date"].max()
                if last_trade_date is None or max_date > last_trade_date:
                    last_trade_date = max_date
        
        # Now count strategies that had trades on the last trade date
        if last_trade_date is not None:
            for strategy in strategies:
                strategy_trades = trades_df[trades_df["Strategy"] == strategy].copy()
                if not strategy_trades.empty:
                    strategy_trades["Date"] = pd.to_datetime(strategy_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                    if last_trade_date in strategy_trades["Date"].values:
                        active_strategies += 1
        
        # Calculate comparison metrics if enabled
        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
            # Calculate comparison period metrics
            comp_total_pnl = 0
            comp_sharpe_values = []
            comp_total_trades = 0
            comp_total_win_trades = 0
            comp_total_loss_trades = 0
            comp_total_slippage = 0
            comp_slippage_pcts = []
            comp_total_ib_commission = 0
            
            for strategy in strategies:
                comp_df = comp_trades[comp_trades["Strategy"] == strategy].copy()
                if comp_df.empty:
                    continue
                comp_df["Date"] = pd.to_datetime(comp_df["TradeDate"], format="%d/%m/%y", dayfirst=True)
                comp_daily_pnl = comp_df.groupby("Date")["NetCash"].sum().sort_index()
                comp_net_pnl = comp_daily_pnl.sum()
                comp_total_pnl += comp_net_pnl
                
                # Calculate Sharpe for comparison
                comp_rf_daily = 0.045 / 252
                comp_excess_ret = comp_daily_pnl - comp_rf_daily
                comp_sharpe = comp_excess_ret.mean() / (comp_daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
                comp_sharpe_values.append(comp_sharpe)
                
                # Calculate trade counts for comparison
                comp_tg = comp_df.groupby(["Strategy", "TradeDate"])
                comp_total, comp_win_t, comp_loss_t = 0, 0, 0
                for (_, _), legs in comp_tg:
                    cnt = max(1, legs["Quantity"].max())
                    comp_total += cnt
                    s = legs["NetCash"].sum()
                    if s > 0: 
                        comp_win_t += cnt
                    elif s < 0: 
                        comp_loss_t += cnt
                
                comp_total_trades += comp_total
                comp_total_win_trades += comp_win_t
                comp_total_loss_trades += comp_loss_t
                
                # Calculate slippage for comparison
                comp_total_slippage += comp_df["Slippage"].sum() if "Slippage" in comp_df.columns else 0
                if "Slippage_pct" in comp_df.columns and "Slippage" in comp_df.columns:
                    comp_slippage_trades = comp_df[comp_df["Slippage"] > 0]
                    if not comp_slippage_trades.empty:
                        comp_weighted_avg = (comp_slippage_trades["Slippage"] * comp_slippage_trades["Slippage_pct"]).sum() / comp_slippage_trades["Slippage"].sum()
                        comp_slippage_pcts.append(comp_weighted_avg)
                
                # Calculate IB Commission for comparison
                comp_total_ib_commission += comp_df["IBCommission"].sum() if "IBCommission" in comp_df.columns else 0
            
            comp_avg_sharpe = np.mean(comp_sharpe_values) if comp_sharpe_values else 0
            comp_overall_win_rate = (comp_total_win_trades / comp_total_trades * 100) if comp_total_trades > 0 else 0
            comp_avg_slippage_pct = np.mean(comp_slippage_pcts) if comp_slippage_pcts else 0
        else:
            comp_total_pnl = None
            comp_avg_sharpe = None
            comp_total_trades = None
            comp_overall_win_rate = None
            comp_total_win_trades = None
            comp_total_loss_trades = None
            comp_total_slippage = None
            comp_avg_slippage_pct = None
            comp_total_ib_commission = None

        # Display summary metrics with comparison
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_comparison_metric("💵 Total Net P&L", total_pnl, comp_total_pnl, "number", "$")
        
        with col2:
            render_comparison_metric("📊 Average Sharpe", avg_sharpe, comp_avg_sharpe, "number")
        
        with col3:
            render_comparison_metric("📈 Total Trades", total_trades, comp_total_trades, "number")
        
        with col4:
            render_comparison_metric("🎯 Overall Win Rate", overall_win_rate, comp_overall_win_rate, "percentage")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if comparison_enabled and comp_total_win_trades is not None and comp_total_loss_trades is not None:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">✅❌ Win/Loss Trades</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="metric-value neutral-value">{total_win_trades}/{total_loss_trades}</span>
                        <span style="font-size: 1.3rem; font-weight: 600; color: #ff7f0e;">{comp_total_win_trades}/{comp_total_loss_trades}</span>
                        <span style="font-size: 1.3rem; font-weight: 600; color: {'#059669' if (comp_total_win_trades/comp_total_loss_trades if comp_total_loss_trades > 0 else 0) > (total_win_trades/total_loss_trades if total_loss_trades > 0 else 0) else '#dc2626' if (comp_total_win_trades/comp_total_loss_trades if comp_total_loss_trades > 0 else 0) < (total_win_trades/total_loss_trades if total_loss_trades > 0 else 0) else '#6b7280'}">
                            {((comp_total_win_trades/comp_total_loss_trades if comp_total_loss_trades > 0 else 0) / (total_win_trades/total_loss_trades if total_loss_trades > 0 else 1) - 1) * 100:+.1f}%
                        </span>
                    </div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total winning / Total losing trades</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">✅❌ Win/Loss Trades</div>
                    <div class="metric-value neutral-value">{total_win_trades}/{total_loss_trades}</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total winning / Total losing trades</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col6:
            render_comparison_metric("💰 Total Slippage", total_slippage, comp_total_slippage, "number", "$")
        
        with col7:
            render_comparison_metric("📊 Avg Slippage %", avg_slippage_pct, comp_avg_slippage_pct, "percentage")
        
        with col8:
            render_comparison_metric("💳 Total IB Commission", total_ib_commission, comp_total_ib_commission, "number", "$")
        
        # Calculate Trading Efficiency Metrics for consolidated view
        avg_trade_return = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss amounts across all strategies
        all_winning_trades = []
        all_losing_trades = []
        
        for strategy in strategies:
            strategy_trades = trades_df[trades_df["Strategy"] == strategy].copy()
            if not strategy_trades.empty:
                daily_pnl = strategy_trades.groupby("TradeDate")["NetCash"].sum()
                winning_days = daily_pnl[daily_pnl > 0]
                losing_days = daily_pnl[daily_pnl < 0]
                all_winning_trades.extend(winning_days.values)
                all_losing_trades.extend(losing_days.values)
        
        avg_win_amount = np.mean(all_winning_trades) if all_winning_trades else 0
        avg_loss_amount = abs(np.mean(all_losing_trades)) if all_losing_trades else 0
        
        # Expectancy per Trade
        expectancy = (overall_win_rate/100 * avg_win_amount) - ((1 - overall_win_rate/100) * avg_loss_amount)
        
        # Reward/Risk Ratio
        rr_ratio = avg_win_amount / avg_loss_amount if avg_loss_amount > 0 else 0
        
        # Break-even Win %
        breakeven_win_pct = 1 / (1 + rr_ratio) * 100 if rr_ratio > 0 else 0
        
        # Cost metrics
        cost_return_ratio = (total_slippage / total_pnl * 100) if total_pnl != 0 else 0
        avg_cost_per_trade = total_slippage / total_trades if total_trades > 0 else 0
        
        # Display Trading Efficiency Metrics
        st.markdown('<h4 class="subsection-header">📊 Trading Efficiency Metrics</h4>', unsafe_allow_html=True)
        col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
        
        with col_eff1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Avg Trade Return</div>
                <div class="metric-value neutral-value">${avg_trade_return:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Net P&L / Total trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Expectancy per Trade</div>
                <div class="metric-value neutral-value">${expectancy:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected value per trade</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Avg Win Amount</div>
                <div class="metric-value positive-value">${avg_win_amount:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average profit per winning trade</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💸 Avg Loss Amount</div>
                <div class="metric-value negative-value">${avg_loss_amount:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average loss per losing trade</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display Cost Metrics
        st.markdown('<h4 class="subsection-header">🧮 Cost Analysis</h4>', unsafe_allow_html=True)
        col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
        
        with col_cost1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Total Slippage</div>
                <div class="metric-value negative-value">${total_slippage:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all trade slippage costs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_cost2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Cost/Return Ratio</div>
                <div class="metric-value neutral-value">{cost_return_ratio:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Slippage / Net P&L × 100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_cost3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💸 Avg Cost per Trade</div>
                <div class="metric-value neutral-value">${avg_cost_per_trade:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total slippage / Total trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_cost4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚖️ Reward/Risk Ratio</div>
                <div class="metric-value neutral-value">{rr_ratio:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Avg Win / Avg Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Total Strategies</div>
                <div class="metric-value neutral-value">{total_strategies}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Number of unique strategies</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add IB Commission metric
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💳 Total IB Commission</div>
                <div class="metric-value neutral-value">${total_ib_commission:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all strategy IB commissions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced Rolling Metrics (Separate Section)
        with st.expander("📈 Rolling Performance Charts", expanded=False):
            st.markdown('<h4 class="subsection-header">📊 Rolling Metrics Analysis</h4>', unsafe_allow_html=True)
            
            # Calculate rolling metrics for all strategies combined
            if not filtered_trades_df.empty:
                # Create daily PnL series for all strategies
                daily_pnl_all = filtered_trades_df.groupby("TradeDate")["NetCash"].sum()
                daily_pnl_all.index = pd.to_datetime(daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                daily_pnl_all = daily_pnl_all.sort_index()
                
                # Calculate rolling metrics
                rolling_7d_pnl = daily_pnl_all.rolling(7).sum()
                rolling_30d_pnl = daily_pnl_all.rolling(30).sum()
                rolling_7d_avg = daily_pnl_all.rolling(7).mean()
                rolling_30d_avg = daily_pnl_all.rolling(30).mean()
                
                # Calculate rolling Sharpe (simplified)
                rolling_7d_std = daily_pnl_all.rolling(7).std()
                rolling_30d_std = daily_pnl_all.rolling(30).std()
                rolling_7d_sharpe = (rolling_7d_avg / rolling_7d_std) * np.sqrt(252)
                rolling_30d_sharpe = (rolling_30d_avg / rolling_30d_std) * np.sqrt(252)
                # Handle division by zero by filling NaN values with 0
                rolling_7d_sharpe = rolling_7d_sharpe.fillna(0)
                rolling_30d_sharpe = rolling_30d_sharpe.fillna(0)
                
                # Display rolling charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("7-Day Rolling P&L")
                    if len(rolling_7d_pnl.dropna()) > 0:
                        # Create 7-day rolling P&L chart with comparison if enabled
                        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                            # Calculate comparison rolling P&L
                            comp_daily_pnl_all = comp_trades.groupby("TradeDate")["NetCash"].sum()
                            comp_daily_pnl_all.index = pd.to_datetime(comp_daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                            comp_daily_pnl_all = comp_daily_pnl_all.sort_index()
                            comp_rolling_7d_pnl = comp_daily_pnl_all.rolling(7).sum()
                            
                            # Create figure with both series
                            fig = go.Figure()
                            
                            # Primary period rolling P&L
                            fig.add_trace(go.Scatter(
                                x=rolling_7d_pnl.index,
                                y=rolling_7d_pnl.values,
                                mode='lines',
                                name=f'Primary Period',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Comparison period rolling P&L
                            fig.add_trace(go.Scatter(
                                x=comp_rolling_7d_pnl.index,
                                y=comp_rolling_7d_pnl.values,
                                mode='lines',
                                name=f'Comparison Period',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="7-Day Rolling P&L",
                                xaxis_title="Date",
                                yaxis_title="7-Day Rolling P&L",
                                height=300,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Set y-axis range based on combined data
                            all_data = pd.concat([rolling_7d_pnl, comp_rolling_7d_pnl])
                            if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                fig.update_layout(yaxis=dict(range=y_range))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_7d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling P&L chart")
                    
                    st.subheader("7-Day Rolling Sharpe")
                    if len(rolling_7d_sharpe.dropna()) > 0:
                        # Create 7-day rolling Sharpe chart with comparison if enabled
                        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                            # Calculate comparison rolling Sharpe
                            comp_daily_pnl_all = comp_trades.groupby("TradeDate")["NetCash"].sum()
                            comp_daily_pnl_all.index = pd.to_datetime(comp_daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                            comp_daily_pnl_all = comp_daily_pnl_all.sort_index()
                            comp_rolling_7d_avg = comp_daily_pnl_all.rolling(7).mean()
                            comp_rolling_7d_std = comp_daily_pnl_all.rolling(7).std()
                            comp_rolling_7d_sharpe = (comp_rolling_7d_avg / comp_rolling_7d_std) * np.sqrt(252)
                            comp_rolling_7d_sharpe = comp_rolling_7d_sharpe.fillna(0)
                            
                            # Create figure with both series
                            fig = go.Figure()
                            
                            # Primary period rolling Sharpe
                            fig.add_trace(go.Scatter(
                                x=rolling_7d_sharpe.index,
                                y=rolling_7d_sharpe.values,
                                mode='lines',
                                name=f'Primary Period',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Comparison period rolling Sharpe
                            fig.add_trace(go.Scatter(
                                x=comp_rolling_7d_sharpe.index,
                                y=comp_rolling_7d_sharpe.values,
                                mode='lines',
                                name=f'Comparison Period',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="7-Day Rolling Sharpe",
                                xaxis_title="Date",
                                yaxis_title="7-Day Rolling Sharpe",
                                height=300,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Set y-axis range based on combined data
                            all_data = pd.concat([rolling_7d_sharpe, comp_rolling_7d_sharpe])
                            if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                fig.update_layout(yaxis=dict(range=y_range))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_7d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling Sharpe chart")
                
                with col_chart2:
                    st.subheader("30-Day Rolling P&L")
                    if len(rolling_30d_pnl.dropna()) > 0:
                        # Create 30-day rolling P&L chart with comparison if enabled
                        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                            # Calculate comparison rolling P&L
                            comp_daily_pnl_all = comp_trades.groupby("TradeDate")["NetCash"].sum()
                            comp_daily_pnl_all.index = pd.to_datetime(comp_daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                            comp_daily_pnl_all = comp_daily_pnl_all.sort_index()
                            comp_rolling_30d_pnl = comp_daily_pnl_all.rolling(30).sum()
                            
                            # Create figure with both series
                            fig = go.Figure()
                            
                            # Primary period rolling P&L
                            fig.add_trace(go.Scatter(
                                x=rolling_30d_pnl.index,
                                y=rolling_30d_pnl.values,
                                mode='lines',
                                name=f'Primary Period',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Comparison period rolling P&L
                            fig.add_trace(go.Scatter(
                                x=comp_rolling_30d_pnl.index,
                                y=comp_rolling_30d_pnl.values,
                                mode='lines',
                                name=f'Comparison Period',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="30-Day Rolling P&L",
                                xaxis_title="Date",
                                yaxis_title="30-Day Rolling P&L",
                                height=300,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Set y-axis range based on combined data
                            all_data = pd.concat([rolling_30d_pnl, comp_rolling_30d_pnl])
                            if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                fig.update_layout(yaxis=dict(range=y_range))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_30d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling P&L chart")
                    
                    st.subheader("30-Day Rolling Sharpe")
                    if len(rolling_30d_sharpe.dropna()) > 0:
                        # Create 30-day rolling Sharpe chart with comparison if enabled
                        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                            # Calculate comparison rolling Sharpe
                            comp_daily_pnl_all = comp_trades.groupby("TradeDate")["NetCash"].sum()
                            comp_daily_pnl_all.index = pd.to_datetime(comp_daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                            comp_daily_pnl_all = comp_daily_pnl_all.sort_index()
                            comp_rolling_30d_avg = comp_daily_pnl_all.rolling(30).mean()
                            comp_rolling_30d_std = comp_daily_pnl_all.rolling(30).std()
                            comp_rolling_30d_sharpe = (comp_rolling_30d_avg / comp_rolling_30d_std) * np.sqrt(252)
                            comp_rolling_30d_sharpe = comp_rolling_30d_sharpe.fillna(0)
                            
                            # Create figure with both series
                            fig = go.Figure()
                            
                            # Primary period rolling Sharpe
                            fig.add_trace(go.Scatter(
                                x=rolling_30d_sharpe.index,
                                y=rolling_30d_sharpe.values,
                                mode='lines',
                                name=f'Primary Period',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Comparison period rolling Sharpe
                            fig.add_trace(go.Scatter(
                                x=comp_rolling_30d_sharpe.index,
                                y=comp_rolling_30d_sharpe.values,
                                mode='lines',
                                name=f'Comparison Period',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="30-Day Rolling Sharpe",
                                xaxis_title="Date",
                                yaxis_title="30-Day Rolling Sharpe",
                                height=300,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Set y-axis range based on combined data
                            all_data = pd.concat([rolling_30d_sharpe, comp_rolling_30d_sharpe])
                            if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                                y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                                fig.update_layout(yaxis=dict(range=y_range))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Original single period chart
                            st.line_chart(rolling_30d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling Sharpe chart")
                
                # Display current rolling metrics only if we have sufficient data
                if len(daily_pnl_all) > 7:  # Need at least 7 days for rolling calculations
                    col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                    with col_roll1:
                        current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty and not np.isnan(rolling_7d_pnl.iloc[-1]) else 0
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📈 7-Day P&L</div>
                            <div class="metric-value neutral-value">${current_7d_pnl:,.0f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative P&L</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_roll2:
                        current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty and not np.isnan(rolling_30d_pnl.iloc[-1]) else 0
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📈 30-Day P&L</div>
                            <div class="metric-value neutral-value">${current_30d_pnl:,.0f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative P&L</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_roll3:
                        current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty and not np.isnan(rolling_7d_sharpe.iloc[-1]) else 0
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 7-Day Sharpe</div>
                            <div class="metric-value neutral-value">{current_7d_sharpe:.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day rolling Sharpe ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_roll4:
                        current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty and not np.isnan(rolling_30d_sharpe.iloc[-1]) else 0
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 30-Day Sharpe</div>
                            <div class="metric-value neutral-value">{current_30d_sharpe:.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day rolling Sharpe ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Insufficient data for rolling metrics in selected period.")
            else:
                st.info("Rolling metrics require trade data.")
        
        # Display additional summary metrics
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Active Strategies</div>
                <div class="metric-value neutral-value">{active_strategies}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col10:
            if last_trade_date:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Last Trade Date</div>
                    <div class="metric-value neutral-value">{last_trade_date.strftime('%Y-%m-%d')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Last Trade Date</div>
                    <div class="metric-value neutral-value">N/A</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col11:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total UIDs</div>
                <div class="metric-value neutral-value">{len(set(trades_df['UID'].dropna()))}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col12:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total IB Commission</div>
                <div class="metric-value neutral-value">${total_ib_commission:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No strategy data available for the selected filters.")


def render():
    st.set_page_config(layout="wide", page_title="Strategy Analytics")
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #4b5563;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    .positive-value {
        color: #059669;
    }
    .negative-value {
        color: #dc2626;
    }
    .neutral-value {
        color: #6b7280;
    }
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.markdown('<h1 class="main-header">📈 Strategy Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<h2 class="section-header">💼 Strategy Performance Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard provides comprehensive analytics for individual strategies and consolidated performance views. 
    All calculations are based on actual trade data and allocation information.
    """)

    trades     = load_trades()
    timeline   = load_timeline()
    strat_df   = load_strategies()
    # Sort strategies in order: TS, UT, SS, Error
    all_strategies = strat_df["Strategy_id"].unique()
    strategies = []
    
    # Add TS strategies first
    strategies.extend(sorted([s for s in all_strategies if s == "TS"]))
    # Add UT strategies second
    strategies.extend(sorted([s for s in all_strategies if s == "UT"]))
    # Add SS strategies third
    strategies.extend(sorted([s for s in all_strategies if s == "SS"]))
    # Add Error last
    strategies.extend(sorted([s for s in all_strategies if s == "ERROR"]))

    view_mode = st.radio(
        "View Mode",
        ["Consolidated View", "Individual Strategy"],
        horizontal=True
    )

    if view_mode == "Individual Strategy":
        strategy = st.selectbox(
            "Select Strategy", 
            strategies,
            help="Select a strategy to view individually"
        )
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
        
        # Primary Period Selection
        choice = st.selectbox("Primary Period", list(presets.keys()), index=0)
        if choice == "Custom Range":
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start Date", value=None)
            end_date   = c2.date_input("End Date",   value=None)
        else:
            days_back = presets[choice]
            if days_back == "all":
                start_date = end_date = None
                st.info("📅 All to Date")
            else:
                days_back = int(days_back)
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.info(f"📅 Primary Period: {choice}: {start_date} to {end_date}")
        
        # Comparison Period Selection
        comparison_enabled = st.checkbox("📊 Enable Period Comparison", value=False)
        
        if comparison_enabled:
            st.markdown("---")
            st.markdown('<h4 class="subsection-header">📊 Comparison Period</h4>', unsafe_allow_html=True)
            
            # Calculate comparison period based on primary period
            if choice != "Custom Range" and choice != "All to Date":
                comparison_choice = st.selectbox(
                    "Comparison Period", 
                    ["Previous Period", "Same Period Last Year", "Custom Comparison"],
                    index=0
                )
                
                if comparison_choice == "Previous Period":
                    # Same duration, but shifted back by the same number of days
                    comp_end_date = start_date - pd.Timedelta(days=1)
                    comp_start_date = comp_end_date - pd.Timedelta(days=days_back)
                    st.info(f"📅 Comparison Period: {comp_start_date} to {comp_end_date}")
                    
                elif comparison_choice == "Same Period Last Year":
                    # Same dates but one year ago
                    comp_start_date = start_date - pd.Timedelta(days=365)
                    comp_end_date = end_date - pd.Timedelta(days=365)
                    st.info(f"📅 Comparison Period: {comp_start_date} to {comp_end_date}")
                    
                else:  # Custom Comparison
                    c3, c4 = st.columns(2)
                    comp_start_date = c3.date_input("Comparison Start Date", value=None)
                    comp_end_date = c4.date_input("Comparison End Date", value=None)
            else:
                # For custom or all-to-date, only allow custom comparison
                c3, c4 = st.columns(2)
                comp_start_date = c3.date_input("Comparison Start Date", value=None)
                comp_end_date = c4.date_input("Comparison End Date", value=None)



        # Filter primary period trades
        df = filter_trades_by_dates(trades, strategy, selected_underlying, start_date, end_date)
        
        # Filter comparison period trades if enabled
        if comparison_enabled and 'comp_start_date' in locals() and comp_start_date and comp_end_date:
            comp_df = filter_trades_by_dates(trades, strategy, selected_underlying, comp_start_date, comp_end_date)
        else:
            comp_df = None

        if df.empty:
            st.warning(f"No trades found for strategy {strategy} with these filters.")
        else:
            render_individual_strategy(
                df, strategy, timeline, df, selected_underlying, comp_df, comparison_enabled
            )
    else:
        # Strategy filter for consolidated view
        strategy_filter = st.multiselect(
            "Filter by Strategy",
            ["All"] + strategies,
            default=["All"],
            help="Select strategies to include in consolidated view (select All or specific ones)"
        )
        
        # Use all strategies if "All" is selected, otherwise use the selected strategies
        if "All" in strategy_filter:
            selected_strategies = strategies
        else:
            selected_strategies = [s for s in strategy_filter if s != "All"]
            if not selected_strategies:  # If nothing selected, show all
                selected_strategies = strategies
        
        # Date range filter for consolidated view
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
                st.info("📅 All to Date")
            else:
                days_back = int(days_back)
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.info(f"📅 {choice}: {start_date} to {end_date}")

        # Comparison Period Selection for consolidated view
        comparison_enabled = st.checkbox("📊 Enable Period Comparison", value=False)
        
        if comparison_enabled:
            st.markdown("---")
            st.markdown('<h4 class="subsection-header">📊 Comparison Period</h4>', unsafe_allow_html=True)
            
            # Calculate comparison period based on primary period
            if choice != "Custom Range" and choice != "All to Date":
                comparison_choice = st.selectbox(
                    "Comparison Period", 
                    ["Previous Period", "Same Period Last Year", "Custom Comparison"],
                    index=0
                )
                
                if comparison_choice == "Previous Period":
                    # Same duration, but shifted back by the same number of days
                    comp_end_date = start_date - pd.Timedelta(days=1)
                    comp_start_date = comp_end_date - pd.Timedelta(days=days_back)
                    st.info(f"📅 Comparison Period: {comp_start_date} to {comp_end_date}")
                    
                elif comparison_choice == "Same Period Last Year":
                    # Same dates but one year ago
                    comp_start_date = start_date - pd.Timedelta(days=365)
                    comp_end_date = end_date - pd.Timedelta(days=365)
                    st.info(f"📅 Comparison Period: {comp_start_date} to {comp_end_date}")
                    
                else:  # Custom Comparison
                    c3, c4 = st.columns(2)
                    comp_start_date = c3.date_input("Comparison Start Date", value=None)
                    comp_end_date = c4.date_input("Comparison End Date", value=None)
            else:
                # For custom or all-to-date, only allow custom comparison
                c3, c4 = st.columns(2)
                comp_start_date = c3.date_input("Comparison Start Date", value=None)
                comp_end_date = c4.date_input("Comparison End Date", value=None)

        # Filter trades based on date range
        filtered_trades = trades.copy()
        if 'start_date' in locals() and start_date and end_date:
            filtered_trades["_dt"] = pd.to_datetime(
                filtered_trades["TradeDate"], format="%d/%m/%y", dayfirst=True
            )
            filtered_trades = filtered_trades[(filtered_trades["_dt"].dt.date >= start_date) &
                                            (filtered_trades["_dt"].dt.date <= end_date)]
            filtered_trades.drop("_dt", axis=1, inplace=True)

        # Filter comparison period trades if enabled
        if comparison_enabled and 'comp_start_date' in locals() and comp_start_date and comp_end_date:
            comp_trades = filter_trades_by_dates(trades, "All", "All", comp_start_date, comp_end_date)
        else:
            comp_trades = None

        render_consolidated_view(trades, selected_strategies, timeline, filtered_trades, comp_trades, comparison_enabled)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 2rem 0;">
        📈 Strategy Analytics Dashboard | Generated on {date} | 
        Data Source: Interactive Brokers Trade Ledger
    </div>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
