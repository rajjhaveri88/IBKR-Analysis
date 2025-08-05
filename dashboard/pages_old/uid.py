# dashboard/pages/uid.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scripts.loaders import (
    load_trades,      # DataFrame of your trades ledger
    load_timeline,    # DataFrame of your raw allocation/margin timeline
    load_uid_margin,  # DataFrame of UID margin data
)



def calculate_uid_nav_with_hedged_margin(uid_name, trades_df, timeline_df, uid_margin_df):
    """
    Returns a DataFrame indexed by Date with columns:
      - nav   : UID NAV (starts at 100, steps on pnl%)
      - units : Total units held (adjusts only when margin changes)
    Uses underlying-specific margin amounts and hedged margin when applicable.
    """
    # 1) Get UID trades
    df_uid = trades_df[trades_df["UID"] == uid_name].copy()
    if df_uid.empty:
        return pd.DataFrame(columns=["nav", "units"])
    
    # 2) Get the strategy and underlying for this UID
    strategy = df_uid["Strategy"].iloc[0] if "Strategy" in df_uid.columns else None
    underlying = df_uid["UnderlyingSymbol"].iloc[0] if "UnderlyingSymbol" in df_uid.columns else None
    
    if not strategy or not underlying:
        return pd.DataFrame(columns=["nav", "units"])
    
    # 3) Get margin data - try underlying-specific first, then fallback to UID margin
    margin_ser = None
    
    # Try to get underlying-specific margin from timeline
    margin_col = f"Margin_{strategy}_Margin_{underlying}"
    hedged_margin_col = f"HedgedMargin_{strategy}_HedgedMargin_{underlying}"
    
    if margin_col in timeline_df.columns:
        timeline_df["Date"] = pd.to_datetime(timeline_df["Date"])
        margin_ser = timeline_df.set_index("Date")[margin_col].sort_index()
        
        # If hedged trades exist, try to use hedged margin
        has_hedged_trades = df_uid["IsHedge"].any() if "IsHedge" in df_uid.columns else False
        if has_hedged_trades and hedged_margin_col in timeline_df.columns:
            hedged_margin_ser = timeline_df.set_index("Date")[hedged_margin_col].sort_index()
            if not hedged_margin_ser.empty and hedged_margin_ser.notna().any():
                margin_ser = hedged_margin_ser
    
    # Fallback to UID margin if no underlying-specific data
    if margin_ser is None or margin_ser.empty:
        uid_margin = uid_margin_df[uid_margin_df["UID"] == uid_name].copy()
        if uid_margin.empty:
            return pd.DataFrame(columns=["nav", "units"])
        
        uid_margin["Date"] = pd.to_datetime(uid_margin["Date"])
        margin_ser = uid_margin.set_index("Date")["Margin"].sort_index()
    
    # 4) Create daily trade count and apply margin multiplication only on trade days
    df_uid["Date"] = pd.to_datetime(
        df_uid["TradeDate"], format="%d/%m/%y", dayfirst=True
    )
    
    # Create trade count series
    trade_count_ser = df_uid.groupby("Date").size()
    
    # Align and multiply margin by trade count only on days with trades
    margin_ser = margin_ser.reindex(trade_count_ser.index.union(margin_ser.index)).ffill()
    trade_count_ser = trade_count_ser.reindex(margin_ser.index).fillna(0)
    margin_ser = margin_ser * trade_count_ser.where(trade_count_ser > 0, 1)
    
    # 5) Carry forward until changed, then drop zeros
    margin_ser = margin_ser.ffill().fillna(0)
    margin_ser = margin_ser[margin_ser > 0]
    
    if margin_ser.empty:
        return pd.DataFrame(columns=["nav", "units"])

    # 6) Build daily P&L series
    pnl_ser = df_uid.groupby("Date")["NetCash"].sum()

    # 7) Merge margin + P&L, flag relevant days
    df = (
        pd.DataFrame({"margin": margin_ser})
          .join(pnl_ser.rename("pnl"), how="left")
          .fillna({"pnl": 0})
          .sort_index()
    )
    df["prev_margin"] = df["margin"].shift(1)
    if not df.empty:
        df.loc[df.index[0], "prev_margin"] = df["margin"].iloc[0]
    df["margin_change"] = df["margin"] != df["prev_margin"]
    df["relevant"]    = (df["pnl"] != 0) | df["margin_change"]
    df = df.loc[df["relevant"]]

    # 8) Loop through relevant days, tracking NAV & units
    nav_list, units_list = [], []
    nav   = 100.0
    
    if df.empty:
        return pd.DataFrame(columns=["nav", "units"])
    
    units = df["margin"].iloc[0] / nav

    for _, row in df.iterrows():
        # apply that day's P&L% on previous capital
        nav *= (1 + row["pnl"] / row["prev_margin"])
        # adjust units only on margin change
        if row["margin_change"]:
            delta = row["margin"] - row["prev_margin"]
            units += delta / nav
        # record
        nav_list.append(nav)
        units_list.append(units)

    nav_df = pd.DataFrame({"nav": nav_list, "units": units_list}, index=df.index)

    # 9) Prepend NAV=100 on day before first active date
    first_date = nav_df.index.min()
    prev_day   = first_date - pd.Timedelta(days=1)
    start_units = nav_df["units"].iloc[0]
    prepend = pd.DataFrame(
        {"nav": [100.0], "units": [start_units]},
        index=[prev_day]
    )
    nav_df = pd.concat([prepend, nav_df]).sort_index()

    return nav_df


def render_individual_uid(df, uid, timeline_df, uid_margin_df, filtered_trades_df):
    """Render individual UID view with professional styling"""
    st.markdown(f'<h3 class="subsection-header">📊 UID: {uid}</h3>', unsafe_allow_html=True)
    
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

    # — NAV & units series (using filtered trades for date range consistency)
    tl2 = timeline_df.copy()
    tl2["Date"] = pd.to_datetime(tl2["Date"], dayfirst=True)
    
    # Filter timeline to match the date range of filtered trades
    if not filtered_trades_df.empty:
        trade_dates = pd.to_datetime(filtered_trades_df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        min_date = trade_dates.min()
        max_date = trade_dates.max()
        tl2 = tl2[(tl2["Date"] >= min_date) & (tl2["Date"] <= max_date)]
    
    nav_units = calculate_uid_nav_with_hedged_margin(
        uid, filtered_trades_df, tl2, uid_margin_df
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

    # Display metrics in styled containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">💵 Net P&L</div>
            <div class="metric-value {'positive-value' if net_pnl > 0 else 'negative-value'}">{net_pnl:,.0f}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From trade NetCash data</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📊 Sharpe Ratio</div>
            <div class="metric-value {'positive-value' if sharpe > 0 else 'negative-value'}">{sharpe:.2f}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Excess return / Volatility × √252</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📈 Profit Factor</div>
            <div class="metric-value">{f'{profit_fac:.2f}' if not np.isnan(profit_fac) else 'N/A'}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total wins / Total losses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📅 Win/Loss Days</div>
            <div class="metric-value neutral-value">{win_days}/{loss_days}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Profitable/unprofitable trading days</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📊 Total Trades</div>
            <div class="metric-value neutral-value">{total}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total number of trades executed</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">🎯 Win/Loss Trades</div>
            <div class="metric-value neutral-value">{win_t}/{loss_t}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Profitable/unprofitable trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📊 Win Rate</div>
            <div class="metric-value">{(win_t/total*100):.1f} %</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Winning trades / Total trades</div>
        </div>
        """, unsafe_allow_html=True)
        
        if has_nav_data and not nav_ser.empty:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Current NAV</div>
                <div class="metric-value">{nav_ser.iloc[-1]:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest NAV value</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Current NAV</div>
                <div class="metric-value neutral-value">N/A</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest NAV value</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">💸 Total Slippage</div>
            <div class="metric-value neutral-value">{total_slippage:,.0f}</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all slippage costs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">📊 Avg Slippage %</div>
            <div class="metric-value neutral-value">{avg_slippage_pct:.2f} %</div>
            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Weighted average slippage percentage</div>
        </div>
        """, unsafe_allow_html=True)
        
        if has_nav_data and not units_ser.empty:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📦 Current Units</div>
                <div class="metric-value neutral-value">{units_ser.iloc[-1]:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest units held</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💳 IB Commission</div>
                <div class="metric-value neutral-value">${total_ib_commission:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total IB commission paid</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📦 Current Units</div>
                <div class="metric-value neutral-value">N/A</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest units held</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💳 IB Commission</div>
                <div class="metric-value neutral-value">${total_ib_commission:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total IB commission paid</div>
            </div>
            """, unsafe_allow_html=True)

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
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">🏆 Max NAV</div>
                    <div class="metric-value">{active_nav.max():.2f}</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Highest NAV achieved</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📉 Curr Drawdown</div>
                    <div class="metric-value negative-value">{(active_nav/active_nav.cummax()-1).iloc[-1]*100:.2f} %</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Current decline from peak</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📉 Max Drawdown</div>
                    <div class="metric-value negative-value">{(active_nav/active_nav.cummax()-1).min()*100:.2f} %</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Largest decline from peak</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📈 CAGR</div>
                    <div class="metric-value {'positive-value' if cagr > 0 else 'negative-value'}">{cagr*100:.2f} %</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Compound annual growth rate</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No NAV data available for this UID.")
    else:
        st.info("No margin data available for NAV calculation.")

    # — charts
    st.markdown('<h3 class="subsection-header">📊 Performance Charts</h3>', unsafe_allow_html=True)
    
    st.markdown('<h4 class="subsection-header">📈 Cumulative P&L</h4>', unsafe_allow_html=True)
    fig = px.line(
        daily_pnl.cumsum(),
        title=f"Cumulative P&L: {uid}",
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
            fig = px.line(
                active_nav,
                title=f"NAV Performance: {uid}",
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
            fig = px.line(
                drawdown,
                title=f"Rolling Drawdown: {uid}",
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
            fig = px.line(
                units_ser,
                title=f"Units Held: {uid}",
                labels={"value": "Units", "index": "Date"}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Performance Metrics for Individual UID
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
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Avg Monthly Return</div>
                            <div class="metric-value neutral-value">{avg_monthly_return:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average of monthly returns from NAV data</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📈 7-Day Rolling Return</div>
                            <div class="metric-value neutral-value">{current_7d_return:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative return</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_r2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📈 30-Day Rolling Return</div>
                            <div class="metric-value neutral-value">{current_30d_return:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative return</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Annualized Volatility</div>
                            <div class="metric-value neutral-value">{annualized_vol:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily volatility × √252</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_r3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Sortino Ratio</div>
                            <div class="metric-value neutral-value">{sortino_ratio:.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Return / Downside deviation</div>
                        </div>
                        """, unsafe_allow_html=True)
                        return_over_dd_display = f"{return_over_dd:.2f}" if not np.isnan(return_over_dd) else "N/A"
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Return/Max DD</div>
                            <div class="metric-value neutral-value">{return_over_dd_display}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">CAGR / Maximum drawdown</div>
                        </div>
                        """, unsafe_allow_html=True)
                
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
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📉 Average Drawdown</div>
                            <div class="metric-value negative-value">{avg_drawdown:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average of all drawdown periods</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">⏱️ Max DD Duration</div>
                            <div class="metric-value neutral-value">{max_dd_duration} days</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Longest drawdown period</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_risk2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">🛡️ VaR (95%)</div>
                            <div class="metric-value negative-value">{var_95:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">95% Value at Risk</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">🛡️ CVaR (95%)</div>
                            <div class="metric-value negative-value">{cvar_95:.2f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Conditional Value at Risk</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_risk3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Kelly Criterion</div>
                            <div class="metric-value neutral-value">{kelly:.3f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Optimal leverage ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Win Rate</div>
                            <div class="metric-value neutral-value">{win_rate*100:.1f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Profitable days / Total days</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📈 Avg Trade Return</div>
                            <div class="metric-value neutral-value">${avg_trade_return:,.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Net P&L / Total trades</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">📊 Expectancy per Trade</div>
                            <div class="metric-value neutral-value">${expectancy:,.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected value per trade</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_eff2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">⚖️ Reward/Risk Ratio</div>
                            <div class="metric-value neutral-value">{rr_ratio_trades:.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Avg Win / Avg Loss</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">🎯 Break-even Win %</div>
                            <div class="metric-value neutral-value">{breakeven_win_pct:.1f}%</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win rate needed to break even</div>
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
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">💸 Avg Loss Amount</div>
                            <div class="metric-value negative-value">${avg_loss_amount:,.2f}</div>
                            <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average loss per losing trade</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('<h4 class="subsection-header">🧮 Cost Metrics</h4>', unsafe_allow_html=True)
                
                    # Calculate cost metrics
                    total_slippage = df["Slippage"].sum() if "Slippage" in df.columns else 0
                    cost_return_ratio = (total_slippage / net_pnl * 100) if net_pnl != 0 else 0
                    avg_cost_per_trade = total_slippage / total_trades if total_trades > 0 else 0
                    
                    # Display Cost Metrics
                    col_cost1, col_cost2, col_cost3 = st.columns(3)
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
                rolling_30d_sharpe = (rolling_7d_avg / rolling_30d_std) * np.sqrt(252)
                # Handle division by zero by filling NaN values with 0
                rolling_7d_sharpe = rolling_7d_sharpe.fillna(0)
                rolling_30d_sharpe = rolling_30d_sharpe.fillna(0)
                
                # Display rolling charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("7-Day Rolling P&L")
                    if len(rolling_7d_pnl.dropna()) > 0:
                        st.line_chart(rolling_7d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling P&L chart")
                    
                    st.subheader("7-Day Rolling Sharpe")
                    if len(rolling_7d_sharpe.dropna()) > 0:
                        st.line_chart(rolling_7d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling Sharpe chart")
                
                with col_chart2:
                    st.subheader("30-Day Rolling P&L")
                    if len(rolling_30d_pnl.dropna()) > 0:
                        st.line_chart(rolling_30d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling P&L chart")
                    
                    st.subheader("30-Day Rolling Sharpe")
                    if len(rolling_30d_sharpe.dropna()) > 0:
                        st.line_chart(rolling_30d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling Sharpe chart")
                
                # Display current rolling metrics
                col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                with col_roll1:
                    current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📈 7-Day P&L</div>
                        <div class="metric-value neutral-value">${current_7d_pnl:,.0f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll2:
                    current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📈 30-Day P&L</div>
                        <div class="metric-value neutral-value">${current_30d_pnl:,.0f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll3:
                    current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📊 7-Day Sharpe</div>
                        <div class="metric-value neutral-value">{current_7d_sharpe:.2f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day rolling Sharpe ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll4:
                    current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📊 30-Day Sharpe</div>
                        <div class="metric-value neutral-value">{current_30d_sharpe:.2f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day rolling Sharpe ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No NAV data to display.")
    else:
        st.info("No NAV charts available - margin data required.")


def render_consolidated_view(full_trades_df, uids, timeline_df, uid_margin_df, filtered_trades_df=None):
    """Render consolidated view with professional styling"""
    st.markdown('<h2 class="section-header">📊 Consolidated UID Performance</h2>', unsafe_allow_html=True)
    
    # Use filtered trades if provided, otherwise use full trades
    trades_df = filtered_trades_df if filtered_trades_df is not None else full_trades_df
    
    rows = []
    
    for uid in uids:
        df = trades_df[trades_df["UID"] == uid].copy()
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
        nav_units = calculate_uid_nav_with_hedged_margin(uid, df, timeline_df, uid_margin_df)
        
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
            "UID": uid,
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
        st.markdown('<h3 class="subsection-header">📋 UID Performance Summary</h3>', unsafe_allow_html=True)
        st.dataframe(dfc, use_container_width=True)

        # — Summary metrics across all UIDs
        st.markdown('<h3 class="subsection-header">📊 Summary Across All UIDs</h3>', unsafe_allow_html=True)
        
        # Calculate totals and averages
        # Calculate total P&L from raw values to avoid rounding errors
        total_pnl = 0
        for uid in uids:
            df = trades_df[trades_df["UID"] == uid].copy()
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
            net_pnl = daily_pnl.sum()
            total_pnl += net_pnl
        
        # Calculate weighted average Sharpe (weighted by number of trades)
        sharpe_values = []
        trade_weights = []
        for row in rows:
            sharpe_val = float(row["Sharpe"])
            trade_count = row["Total Trades"]
            # Filter out extreme individual Sharpe values before calculating weighted average
            if np.isfinite(sharpe_val) and abs(sharpe_val) <= 100:
                sharpe_values.append(sharpe_val)
                trade_weights.append(trade_count)
        
        # Calculate weighted average Sharpe with safety checks
        try:
            if sharpe_values and trade_weights:
                avg_sharpe = np.average(sharpe_values, weights=trade_weights)
                # Final safety check for the calculated average
                if not np.isfinite(avg_sharpe):
                    avg_sharpe = 0
            else:
                avg_sharpe = 0
        except:
            avg_sharpe = 0
        
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
        total_uids = len(uids)
        
        # Calculate active UIDs (those with trades on the last trade date)
        active_uids = 0
        last_trade_date = None
        
        # First, find the last trade date across all UIDs
        for uid in uids:
            uid_trades = trades_df[trades_df["UID"] == uid].copy()
            if not uid_trades.empty:
                uid_trades["Date"] = pd.to_datetime(uid_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                max_date = uid_trades["Date"].max()
                if last_trade_date is None or max_date > last_trade_date:
                    last_trade_date = max_date
        
        # Now count UIDs that had trades on the last trade date
        if last_trade_date is not None:
            for uid in uids:
                uid_trades = trades_df[trades_df["UID"] == uid].copy()
                if not uid_trades.empty:
                    uid_trades["Date"] = pd.to_datetime(uid_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                    if last_trade_date in uid_trades["Date"].values:
                        active_uids += 1
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💵 Total Net P&L</div>
                <div class="metric-value {'positive-value' if total_pnl > 0 else 'negative-value'}">{total_pnl:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all UID P&L</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Average Sharpe</div>
                <div class="metric-value {'positive-value' if avg_sharpe > 0 else 'negative-value'}">{avg_sharpe:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Trade-weighted average Sharpe</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Total Trades</div>
                <div class="metric-value neutral-value">{total_trades}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all UID trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Overall Win Rate</div>
                <div class="metric-value">{overall_win_rate:.1f} %</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total wins / Total trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Win/Loss Trades</div>
                <div class="metric-value neutral-value">{total_win_trades}/{total_loss_trades}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total wins / Total losses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💸 Total Slippage</div>
                <div class="metric-value neutral-value">{total_slippage:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all slippage costs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Avg Slippage %</div>
                <div class="metric-value neutral-value">{avg_slippage_pct:.2f} %</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average slippage percentage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📦 Total UIDs</div>
                <div class="metric-value neutral-value">{total_uids}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Number of unique UIDs</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display additional summary metrics
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Active UIDs</div>
                <div class="metric-value neutral-value">{active_uids}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">UIDs with recent trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col10:
            if last_trade_date:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📅 Last Trade Date</div>
                    <div class="metric-value neutral-value">{last_trade_date.strftime('%Y-%m-%d')}</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Most recent trade date</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📅 Last Trade Date</div>
                    <div class="metric-value neutral-value">N/A</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Most recent trade date</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col11:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Total Strategies</div>
                <div class="metric-value neutral-value">{len(set(trades_df['Strategy'].dropna()))}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Unique strategies used</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col12:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💳 Total IB Commission</div>
                <div class="metric-value neutral-value">${total_ib_commission:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total commission paid</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate Trading Efficiency Metrics for consolidated view
        avg_trade_return = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss amounts across all UIDs
        all_winning_trades = []
        all_losing_trades = []
        
        for uid in uids:
            uid_trades = trades_df[trades_df["UID"] == uid]
            if not uid_trades.empty:
                daily_pnl = uid_trades.groupby("TradeDate")["NetCash"].sum()
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
        
        # Advanced Rolling Metrics for Consolidated View
        with st.expander("📈 Rolling Performance Charts", expanded=False):
            st.markdown('<h4 class="subsection-header">📊 Rolling Metrics Analysis</h4>', unsafe_allow_html=True)
            
            # Calculate rolling metrics for all UIDs combined
            if not trades_df.empty:
                # Create daily PnL series for all UIDs
                daily_pnl_all = trades_df.groupby("TradeDate")["NetCash"].sum()
                daily_pnl_all.index = pd.to_datetime(daily_pnl_all.index, format="%d/%m/%y", dayfirst=True)
                daily_pnl_all = daily_pnl_all.sort_index()
                
                # Calculate rolling metrics
                rolling_7d_pnl = daily_pnl_all.rolling(7).sum()
                rolling_30d_pnl = daily_pnl_all.rolling(30).sum()
                rolling_7d_avg = daily_pnl_all.rolling(7).mean()
                rolling_30d_avg = daily_pnl_all.rolling(30).mean()
                
                # Calculate rolling Sharpe
                rolling_7d_std = daily_pnl_all.rolling(7).std()
                rolling_30d_std = daily_pnl_all.rolling(30).std()
                rolling_7d_sharpe = (rolling_7d_avg / rolling_7d_std) * np.sqrt(252)
                rolling_30d_sharpe = (rolling_7d_avg / rolling_30d_std) * np.sqrt(252)
                # Handle division by zero by filling NaN values with 0
                rolling_7d_sharpe = rolling_7d_sharpe.fillna(0)
                rolling_30d_sharpe = rolling_30d_sharpe.fillna(0)
                
                # Display rolling charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("7-Day Rolling P&L")
                    if len(rolling_7d_pnl.dropna()) > 0:
                        st.line_chart(rolling_7d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling P&L chart")
                    
                    st.subheader("7-Day Rolling Sharpe")
                    if len(rolling_7d_sharpe.dropna()) > 0:
                        st.line_chart(rolling_7d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 7-day rolling Sharpe chart")
                
                with col_chart2:
                    st.subheader("30-Day Rolling P&L")
                    if len(rolling_30d_pnl.dropna()) > 0:
                        st.line_chart(rolling_30d_pnl, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling P&L chart")
                    
                    st.subheader("30-Day Rolling Sharpe")
                    if len(rolling_30d_sharpe.dropna()) > 0:
                        st.line_chart(rolling_30d_sharpe, use_container_width=True)
                    else:
                        st.info("Insufficient data for 30-day rolling Sharpe chart")
                
                # Display current rolling metrics
                col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                with col_roll1:
                    current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📈 7-Day P&L</div>
                        <div class="metric-value neutral-value">${current_7d_pnl:,.0f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll2:
                    current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📈 30-Day P&L</div>
                        <div class="metric-value neutral-value">${current_30d_pnl:,.0f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll3:
                    current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📊 7-Day Sharpe</div>
                        <div class="metric-value neutral-value">{current_7d_sharpe:.2f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day rolling Sharpe ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_roll4:
                    current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty else 0
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">📊 30-Day Sharpe</div>
                        <div class="metric-value neutral-value">{current_30d_sharpe:.2f}</div>
                        <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day rolling Sharpe ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Rolling metrics require trade data.")
    else:
        st.warning("⚠️ No UID data available for the selected filters.")


def render():
    st.set_page_config(layout="wide", page_title="UID Analytics")
    
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
    st.markdown('<h1 class="main-header">📈 UID Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<h2 class="section-header">💼 UID Performance Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard provides comprehensive analytics for individual UIDs and consolidated performance views. 
    All calculations are based on actual trade data and margin information.
    """)

    trades     = load_trades()
    timeline   = load_timeline()
    uid_margin = load_uid_margin()
    
    # Get unique UIDs from trades data
    # Sort UIDs in order: TS, UT, SS, Error
    all_uids = trades["UID"].unique()
    uids = []
    
    # Add TS UIDs first
    uids.extend(sorted([u for u in all_uids if u.startswith("TS-")]))
    # Add UT UIDs second
    uids.extend(sorted([u for u in all_uids if u.startswith("UT-")]))
    # Add SS UIDs third
    uids.extend(sorted([u for u in all_uids if u.startswith("SS-")]))
    # Add Error last
    uids.extend(sorted([u for u in all_uids if u == "Error"]))
    # Include all UIDs including 'Error' to match strategy page

    view_mode = st.radio(
        "View Mode",
        ["Consolidated View", "Individual UID"],
        horizontal=True
    )

    if view_mode == "Individual UID":
        uid = st.selectbox(
            "Select UID", 
            uids,
            help="Select a UID to view individually"
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
                st.info("📅 All to Date")
            else:
                days_back = int(days_back)
                end_date   = pd.Timestamp.now().date()
                start_date = end_date - pd.Timedelta(days=days_back)
                st.info(f"📅 {choice}: {start_date} to {end_date}")

        # filter trades
        df = trades[trades["UID"] == uid].copy()
        if 'start_date' in locals() and start_date and end_date:
            df["_dt"] = pd.to_datetime(
                df["TradeDate"], format="%d/%m/%y", dayfirst=True
            )
            df = df[(df["_dt"].dt.date >= start_date) &
                    (df["_dt"].dt.date <= end_date)]
            df.drop("_dt", axis=1, inplace=True)

        if df.empty:
            st.warning(f"No trades found for UID {uid} with these filters.")
        else:
            render_individual_uid(
                df, uid, timeline, uid_margin, df
            )
    else:
        # UID filter for consolidated view
        uid_filter = st.multiselect(
            "Filter by UID",
            ["All"] + uids,
            default=["All"],
            help="Select UIDs to include in consolidated view (select All or specific ones)"
        )
        
        # Use all UIDs if "All" is selected, otherwise use the selected UIDs
        if "All" in uid_filter:
            selected_uids = uids
        else:
            selected_uids = [u for u in uid_filter if u != "All"]
            if not selected_uids:  # If nothing selected, show all
                selected_uids = uids
        
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

        # Filter trades based on date range
        filtered_trades = trades.copy()
        if 'start_date' in locals() and start_date and end_date:
            filtered_trades["_dt"] = pd.to_datetime(
                filtered_trades["TradeDate"], format="%d/%m/%y", dayfirst=True
            )
            filtered_trades = filtered_trades[(filtered_trades["_dt"].dt.date >= start_date) &
                                            (filtered_trades["_dt"].dt.date <= end_date)]
            filtered_trades.drop("_dt", axis=1, inplace=True)

        render_consolidated_view(trades, selected_uids, timeline, uid_margin, filtered_trades)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 2rem 0;">
        📈 UID Analytics Dashboard | Generated on {date} | 
        Data Source: Interactive Brokers Trade Ledger
    </div>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
