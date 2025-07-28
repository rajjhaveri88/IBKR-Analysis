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
            <div class="metric-label">Net P&L</div>
            <div class="metric-value {'positive-value' if net_pnl > 0 else 'negative-value'}">{net_pnl:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {'positive-value' if sharpe > 0 else 'negative-value'}">{sharpe:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{f'{profit_fac:.2f}' if not np.isnan(profit_fac) else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Win/Loss Days</div>
            <div class="metric-value neutral-value">{win_days}/{loss_days}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value neutral-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Win/Loss Trades</div>
            <div class="metric-value neutral-value">{win_t}/{loss_t}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{(win_t/total*100):.1f} %</div>
        </div>
        """, unsafe_allow_html=True)
        
        if has_nav_data and not nav_ser.empty:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current NAV</div>
                <div class="metric-value">{nav_ser.iloc[-1]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current NAV</div>
                <div class="metric-value neutral-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Slippage</div>
            <div class="metric-value neutral-value">{total_slippage:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Avg Slippage %</div>
            <div class="metric-value neutral-value">{avg_slippage_pct:.2f} %</div>
        </div>
        """, unsafe_allow_html=True)
        
        if has_nav_data and not units_ser.empty:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current Units</div>
                <div class="metric-value neutral-value">{units_ser.iloc[-1]:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value neutral-value">{val_ser.iloc[-1]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current Units</div>
                <div class="metric-value neutral-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value neutral-value">N/A</div>
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
                    <div class="metric-label">Max NAV</div>
                    <div class="metric-value">{active_nav.max():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Curr Drawdown</div>
                    <div class="metric-value negative-value">{(active_nav/active_nav.cummax()-1).iloc[-1]*100:.2f} %</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative-value">{(active_nav/active_nav.cummax()-1).min()*100:.2f} %</div>
                </div>
                """, unsafe_allow_html=True)
            
            with d4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">CAGR</div>
                    <div class="metric-value {'positive-value' if cagr > 0 else 'negative-value'}">{cagr*100:.2f} %</div>
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
            else:
                cagr = np.nan
                current_nav = np.nan
                current_units = np.nan
                portfolio_value = np.nan

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
            "Total Slippage": f"{total_slippage:,.0f}",
            "Avg Slippage %": f"{avg_slippage_pct:.2f} %",
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
        
        # Calculate summary metrics
        total_uids = len(uids)
        
        # Calculate active UIDs (those with trades on the last trade date)
        active_uids = 0
        last_trade_date = None
        
        # First, find the last trade date across all UIDs
        for uid in uids:
            uid_trades = trades_df[trades_df["UID"] == uid]
            if not uid_trades.empty:
                uid_trades["Date"] = pd.to_datetime(uid_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                max_date = uid_trades["Date"].max()
                if last_trade_date is None or max_date > last_trade_date:
                    last_trade_date = max_date
        
        # Now count UIDs that had trades on the last trade date
        if last_trade_date is not None:
            for uid in uids:
                uid_trades = trades_df[trades_df["UID"] == uid]
                if not uid_trades.empty:
                    uid_trades["Date"] = pd.to_datetime(uid_trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
                    if last_trade_date in uid_trades["Date"].values:
                        active_uids += 1
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Net P&L</div>
                <div class="metric-value {'positive-value' if total_pnl > 0 else 'negative-value'}">{total_pnl:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Average Sharpe</div>
                <div class="metric-value {'positive-value' if avg_sharpe > 0 else 'negative-value'}">{avg_sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value neutral-value">{total_trades}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Overall Win Rate</div>
                <div class="metric-value">{overall_win_rate:.1f} %</div>
            </div>
            """, unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Win/Loss Trades</div>
                <div class="metric-value neutral-value">{total_win_trades}/{total_loss_trades}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total Slippage</div>
                <div class="metric-value neutral-value">{total_slippage:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Slippage %</div>
                <div class="metric-value neutral-value">{avg_slippage_pct:.2f} %</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Total UIDs</div>
                <div class="metric-value neutral-value">{total_uids}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display additional summary metrics
        col9, col10, col11 = st.columns(3)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Active UIDs</div>
                <div class="metric-value neutral-value">{active_uids}</div>
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
                <div class="metric-label">Total Strategies</div>
                <div class="metric-value neutral-value">{len(set(trades_df['Strategy'].dropna()))}</div>
            </div>
            """, unsafe_allow_html=True)
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
        border-bottom: 3px solid #3b82f6;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
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
