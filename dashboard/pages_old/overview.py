# dashboard/pages/overview.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scripts.loaders import (
    load_trades,      # DataFrame of your trades ledger
    load_timeline,    # DataFrame of your raw allocation/margin timeline
    load_nav,         # DataFrame of your NAV data
    load_fund_flow,   # DataFrame of your fund flow statement
)


def render():
    st.set_page_config(layout="wide", page_title="IBKR Portfolio Overview")
    
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
    st.markdown('<h1 class="main-header">📊 IBKR Portfolio Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<h2 class="section-header">📈 Executive Summary</h2>', unsafe_allow_html=True)
    st.markdown("""
    This comprehensive dashboard provides real-time insights into your IBKR portfolio performance, 
    including capital allocation, trading metrics, risk analysis, and NAV progression. 
    All metrics are calculated using actual trade data and fund flow statements.
    """)

    # Date Range Filter
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

    # Load data
    timeline = load_timeline()
    trades = load_trades()
    nav_df = load_nav()
    fund_flow = load_fund_flow()

    # Filter data based on date range (using same pattern as Strategy page)
    if 'start_date' in locals() and start_date and end_date:
        # Filter timeline
        timeline["_dt"] = pd.to_datetime(timeline["Date"], dayfirst=True)
        filtered_tl = timeline[(timeline["_dt"].dt.date >= start_date) & 
                              (timeline["_dt"].dt.date <= end_date)].copy()
        filtered_tl.drop("_dt", axis=1, inplace=True)
        
        # Filter trades
        trades["_dt"] = pd.to_datetime(trades["TradeDate"], format="%d/%m/%y", dayfirst=True)
        filtered_trades = trades[(trades["_dt"].dt.date >= start_date) & 
                                (trades["_dt"].dt.date <= end_date)].copy()
        filtered_trades.drop("_dt", axis=1, inplace=True)
        
        # Filter NAV
        nav_df["_dt"] = pd.to_datetime(nav_df["Date"], dayfirst=True)
        filtered_nav = nav_df[(nav_df["_dt"].dt.date >= start_date) & 
                             (nav_df["_dt"].dt.date <= end_date)].copy()
        filtered_nav.drop("_dt", axis=1, inplace=True)
        
        # Filter fund flow
        fund_flow["_dt"] = pd.to_datetime(fund_flow["Date"], dayfirst=True)
        filtered_fund_flow = fund_flow[(fund_flow["_dt"].dt.date >= start_date) & 
                                      (fund_flow["_dt"].dt.date <= end_date)].copy()
        filtered_fund_flow.drop("_dt", axis=1, inplace=True)
    else:
        # No date filter applied
        filtered_tl = timeline
        filtered_trades = trades
        filtered_nav = nav_df
        filtered_fund_flow = fund_flow

    # Capital Allocation Overview
    st.markdown('<h2 class="section-header">💰 Capital Allocation Overview</h2>', unsafe_allow_html=True)
    
    if not filtered_tl.empty:
        # Prepare timeline & compute allocations/spare
        filtered_tl = filtered_tl.copy()
        filtered_tl["Date"] = pd.to_datetime(filtered_tl["Date"], dayfirst=True)
        alloc_cols = [c for c in filtered_tl.columns if c.startswith("Allocation_")]
        
        # Calculate total allocation and spare capital
        filtered_tl["TotalAllocation"] = filtered_tl[alloc_cols].sum(axis=1)
        filtered_tl["SpareCapital"] = filtered_tl["BrokerBalance"] - filtered_tl["TotalAllocation"]
        
        # Capital Allocation Chart
        st.markdown('<h3 class="subsection-header">📊 Capital Allocation Timeline</h3>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # 1) individual allocations as stacked area
        alloc_cols = [c for c in filtered_tl.columns if c.startswith("Allocation_")]
        for col in alloc_cols:
            strategy = col.split("_")[1]
            underlying = col.split("_")[-1]
            fig.add_trace(go.Scatter(
                x=filtered_tl["Date"], y=filtered_tl[col],
                mode="lines",
                name=f"{strategy} - {underlying}",
                stackgroup="one",
                fill="tonexty"
            ))

        # 2) total allocation as solid line
        fig.add_trace(go.Scatter(
            x=filtered_tl["Date"], y=filtered_tl["TotalAllocation"],
            mode="lines",
            name="Total Allocation",
            line=dict(color="lightgrey", width=2)
        ))

        # 3) spare capital as dashed black
        fig.add_trace(go.Scatter(
            x=filtered_tl["Date"], y=filtered_tl["SpareCapital"],
            mode="lines",
            name="Spare Capital",
            line=dict(dash="dash", width=1, color="black")
        ))

        # 4) broker balance as solid blue
        fig.add_trace(go.Scatter(
            x=filtered_tl["Date"], y=filtered_tl["BrokerBalance"],
            mode="lines",
            name="Broker Balance",
            line=dict(color="blue", width=2)
        ))

        # Calculate Y-axis range with leeway
        max_value = max(
            filtered_tl['TotalAllocation'].max(),
            filtered_tl['SpareCapital'].max()
        ) if not filtered_tl.empty else 1000
        
        # Add 20% leeway to the top
        y_max = max_value * 1.2
        
        # Calculate X-axis range with leeway
        if not filtered_tl.empty:
            x_min = filtered_tl['Date'].min()
            x_max = filtered_tl['Date'].max()
            x_range = (x_max - x_min).days
            # Add 10% leeway on each side
            x_leeway = x_range * 0.1
            x_min_with_leeway = x_min - pd.Timedelta(days=x_leeway)
            x_max_with_leeway = x_max + pd.Timedelta(days=x_leeway)
        else:
            x_min_with_leeway = pd.Timestamp.now() - pd.Timedelta(days=30)
            x_max_with_leeway = pd.Timestamp.now()

        fig.update_layout(
            title="Capital Allocation Over Time",
            xaxis_title="Date",
            yaxis_title="Capital (×1,000)",
            xaxis=dict(
                range=[x_min_with_leeway, x_max_with_leeway],
                dtick="D7",  # Show tick every 7 days
                tickangle=-45,  # Angle labels for better readability
                tickmode="auto",
                nticks=15  # Show approximately 15 ticks
            ),
            yaxis=dict(
                range=[0, max(y_max, 1200000)],  # Show up to 1.2M or higher if data exceeds it
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                dtick=max(y_max, 1200000)/6,  # Show 6 grid lines including top
                tickmode="auto"
            ),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            margin=dict(l=40, r=40, t=150, b=60),  # Increased top margin for more space
            hovermode="x unified",
            height=650
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Current Capital Metrics
        st.markdown('<h3 class="subsection-header">💼 Current Capital Status</h3>', unsafe_allow_html=True)
        
        today = filtered_tl.iloc[-1]
        spare = today["SpareCapital"]
        total_alloc = today["TotalAllocation"]
        broker = today["BrokerBalance"]
        
        # Calculate total interest from filtered fund flow
        total_interest = 0
        if not filtered_fund_flow.empty:
            total_interest = filtered_fund_flow["BrokerInterest"].sum()
        
        # Display metrics in a professional grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Spare Capital</div>
                <div class="metric-value">${spare:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Total Allocated</div>
                <div class="metric-value">${total_alloc:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🏦 Broker Balance</div>
                <div class="metric-value">${broker:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💵 Total Interest</div>
                <div class="metric-value">${total_interest:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"*Data as of {today['Date'].strftime('%d‑%b‑%Y')}*")
    else:
        st.warning("⚠️ No timeline data available for the selected date range.")

    # Trading Performance Metrics
    st.markdown('<h2 class="section-header">📊 Trading Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_trades.empty:
        # Calculate trading metrics
        tr = filtered_trades.copy()
        tr["TradeDate_dt"] = pd.to_datetime(tr["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = tr.groupby("TradeDate_dt")["NetCash"].sum().sort_index()

        net_pnl    = daily_pnl.sum()
        rf_daily   = 0.0
        excess     = daily_pnl - rf_daily
        sharpe     = excess.mean() / (daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
        gross_win  = daily_pnl[daily_pnl>0].sum()
        gross_loss = -daily_pnl[daily_pnl<0].sum()
        pf         = gross_win / gross_loss if gross_loss else np.nan
        win_days   = int((daily_pnl>0).sum())
        loss_days  = int((daily_pnl<0).sum())
        total_days = win_days + loss_days
        win_days_pct = (win_days / total_days * 100) if total_days > 0 else 0
        
        # Trade-level metrics (using same logic as strategy/UID pages)
        tg = tr.groupby(["UID", "TradeDate"])
        total_trades, win_trades, loss_trades = 0, 0, 0
        for (_, _), legs in tg:
            cnt = max(1, legs["Quantity"].max())
            total_trades += cnt
            s = legs["NetCash"].sum()
            if s > 0: 
                win_trades += cnt
            elif s < 0: 
                loss_trades += cnt
        win_trades_pct = (win_trades / total_trades * 100) if total_trades > 0 else 0

        # Performance Metrics Grid
        st.markdown('<h3 class="subsection-header">🎯 Key Performance Indicators</h3>', unsafe_allow_html=True)
        
        # Row 1: Core Performance Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pnl_color = "positive-value" if net_pnl >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💵 Net P&L</div>
                <div class="metric-value {pnl_color}">${net_pnl:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe_color = "positive-value" if sharpe >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Sharpe Ratio</div>
                <div class="metric-value {sharpe_color}">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pf_color = "positive-value" if pf > 1 else "negative-value" if pf < 1 else "neutral-value"
            pf_display = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚖️ Profit Factor</div>
                <div class="metric-value {pf_color}">{pf_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Day-based Metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Win/Loss Days</div>
                <div class="metric-value neutral-value">{win_days} / {loss_days}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Win Days %</div>
                <div class="metric-value neutral-value">{win_days_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Win/Loss Trades</div>
                <div class="metric-value neutral-value">{win_trades} / {loss_trades}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 3: Additional Metrics
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Win Trades %</div>
                <div class="metric-value neutral-value">{win_trades_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🔄 Total Trades</div>
                <div class="metric-value neutral-value">{total_trades}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Trading Days</div>
                <div class="metric-value neutral-value">{total_days}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No trading data available for the selected date range.")

    # NAV Performance Analysis
    st.markdown('<h2 class="section-header">📈 NAV Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_nav.empty:
        nav = filtered_nav.copy()
        nav["Date"] = pd.to_datetime(nav["Date"], dayfirst=True)
        nav_ser = nav.set_index("Date")["NAV"].sort_index()
        
        # For filtered date ranges, normalize to start at 100
        if start_date is not None and end_date is not None:
            start_nav_value = nav_ser.iloc[0]
            nav_ser = (nav_ser / start_nav_value) * 100
        
        first_nav = nav_ser.iloc[0]
        current_nav = nav_ser.iloc[-1]
        max_nav = nav_ser.max()
        cummax_nav = nav_ser.cummax()
        dd_ser = nav_ser / cummax_nav - 1
        current_dd = dd_ser.iloc[-1]
        max_dd = dd_ser.min()

        days = (nav_ser.index[-1] - nav_ser.index[0]).days
        cagr = ((current_nav/first_nav) ** (365.25/days) - 1) if days>0 else np.nan

        # NAV Metrics Grid
        st.markdown('<h3 class="subsection-header">📊 NAV Key Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nav_color = "positive-value" if current_nav >= first_nav else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Current NAV</div>
                <div class="metric-value {nav_color}">{current_nav:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🏆 Max NAV</div>
                <div class="metric-value neutral-value">{max_nav:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dd_color = "negative-value" if current_dd < 0 else "positive-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📉 Current Drawdown</div>
                <div class="metric-value {dd_color}">{current_dd*100:,.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            max_dd_color = "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📉 Max Drawdown</div>
                <div class="metric-value {max_dd_color}">{max_dd*100:,.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            cagr_color = "positive-value" if cagr >= 0 else "negative-value"
            cagr_display = f"{cagr*100:,.2f}%" if not np.isnan(cagr) else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 CAGR</div>
                <div class="metric-value {cagr_color}">{cagr_display}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Analysis Period</div>
                <div class="metric-value neutral-value">{days} days</div>
            </div>
            """, unsafe_allow_html=True)

        # NAV Chart
        st.markdown('<h3 class="subsection-header">📈 NAV Performance Chart</h3>', unsafe_allow_html=True)
        
        fig_nav = px.line(nav_ser, title="NAV Performance Over Time")
        fig_nav.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_nav, use_container_width=True)
    else:
        st.warning("⚠️ No NAV data available for the selected date range.")

    # Risk Analysis
    st.markdown('<h2 class="section-header">⚠️ Risk Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_tl.empty:
        st.markdown('<h3 class="subsection-header">📊 Risk Metrics</h3>', unsafe_allow_html=True)
        
        # Rolling Drawdown
        st.markdown('<h4 class="subsection-header">📉 Rolling Drawdown Analysis</h4>', unsafe_allow_html=True)
        broker_balance = filtered_tl.set_index("Date")["BrokerBalance"]
        drawdown = (broker_balance / broker_balance.cummax() - 1) * 100
        
        fig_dd = px.area(drawdown, title="Rolling Drawdown (%)")
        fig_dd.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Margin Headroom
        st.markdown('<h4 class="subsection-header">🛡️ Margin Headroom Analysis</h4>', unsafe_allow_html=True)
        headroom = filtered_tl["SpareCapital"] / filtered_tl["BrokerBalance"] * 100
        
        fig_headroom = px.line(filtered_tl, x="Date", y=headroom, title="Margin Headroom (%)")
        fig_headroom.update_layout(
            xaxis_title="Date",
            yaxis_title="Spare Capital / Broker Balance (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_headroom, use_container_width=True)
    else:
        st.warning("⚠️ No timeline data available for risk analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 2rem 0;">
        📊 IBKR Portfolio Analytics Dashboard | Generated on {date} | 
        Data Source: Interactive Brokers Trade Ledger & Fund Flow Statements
    </div>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
