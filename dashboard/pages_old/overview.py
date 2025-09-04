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

    # Load data
    timeline = load_timeline()
    trades = load_trades()
    nav_df = load_nav()
    fund_flow = load_fund_flow()

    # Helper function to filter data by date range
    def filter_data_by_dates(data, date_col, start_date, end_date):
        """Filter data by date range"""
        if start_date and end_date:
            data_copy = data.copy()
            data_copy["_dt"] = pd.to_datetime(data_copy[date_col], dayfirst=True)
            filtered = data_copy[(data_copy["_dt"].dt.date >= start_date) & 
                                (data_copy["_dt"].dt.date <= end_date)].copy()
            filtered.drop("_dt", axis=1, inplace=True)
            return filtered
        else:
            return data

    # Filter primary period data
    if 'start_date' in locals() and start_date and end_date:
        filtered_tl = filter_data_by_dates(timeline, "Date", start_date, end_date)
        filtered_trades = filter_data_by_dates(trades, "TradeDate", start_date, end_date)
        filtered_nav = filter_data_by_dates(nav_df, "Date", start_date, end_date)
        filtered_fund_flow = filter_data_by_dates(fund_flow, "Date", start_date, end_date)
    else:
        # No date filter applied
        filtered_tl = timeline
        filtered_trades = trades
        filtered_nav = nav_df
        filtered_fund_flow = fund_flow

    # Filter comparison period data if enabled
    if comparison_enabled and 'comp_start_date' in locals() and comp_start_date and comp_end_date:
        comp_tl = filter_data_by_dates(timeline, "Date", comp_start_date, comp_end_date)
        comp_trades = filter_data_by_dates(trades, "TradeDate", comp_start_date, comp_end_date)
        comp_nav = filter_data_by_dates(nav_df, "Date", comp_start_date, comp_end_date)
        comp_fund_flow = filter_data_by_dates(fund_flow, "Date", comp_start_date, comp_end_date)
    else:
        comp_tl = None
        comp_trades = None
        comp_nav = None
        comp_fund_flow = None

    # Helper function to calculate percentage change
    def calculate_pct_change(primary_value, comparison_value):
        """Calculate percentage change between primary and comparison values"""
        if comparison_value == 0:
            return 0 if primary_value == 0 else float('inf')
        return ((primary_value - comparison_value) / abs(comparison_value)) * 100

    # Helper function to render comparison metric
    def render_comparison_metric(label, primary_value, comparison_value, format_type="number", prefix=""):
        """Render a metric with comparison data side-by-side"""
        if comparison_enabled and comparison_value is not None:
            pct_change = calculate_pct_change(primary_value, comparison_value)
            primary_color = "positive-value" if primary_value >= 0 else "negative-value" if primary_value < 0 else "neutral-value"
            comp_color = "positive-value" if comparison_value >= 0 else "negative-value" if comparison_value < 0 else "neutral-value"
            pct_color = "positive-value" if pct_change > 0 else "negative-value" if pct_change < 0 else "neutral-value"
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;">
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value {primary_color}" style="font-size: 1.3rem;">{prefix}{primary_value:,.0f}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">Primary</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div class="metric-value {comp_color}" style="font-size: 1.3rem;">{prefix}{comparison_value:,.0f}</div>
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
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div class="metric-value {primary_color}">{prefix}{primary_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================================
    # SECTION 1: PORTFOLIO OVERVIEW & CAPITAL ALLOCATION
    # ============================================================================
    st.markdown('<h2 class="section-header">💰 Portfolio Overview & Capital Allocation</h2>', unsafe_allow_html=True)
    
    if not filtered_tl.empty:
        # Prepare timeline & compute allocations/spare
        filtered_tl = filtered_tl.copy()
        filtered_tl["Date"] = pd.to_datetime(filtered_tl["Date"], dayfirst=True)
        alloc_cols = [c for c in filtered_tl.columns if c.startswith("Allocation_")]
        
        # Calculate total allocation and spare capital
        filtered_tl["TotalAllocation"] = filtered_tl[alloc_cols].sum(axis=1)
        filtered_tl["SpareCapital"] = filtered_tl["BrokerBalance"] - filtered_tl["TotalAllocation"]
        
        # Current Capital Status
        st.markdown('<h3 class="subsection-header">💼 Current Capital Status</h3>', unsafe_allow_html=True)
        
        today = filtered_tl.iloc[-1]
        spare = today["SpareCapital"]
        total_alloc = today["TotalAllocation"]
        broker = today["BrokerBalance"]
        
        # Calculate total interest from filtered fund flow
        total_interest = 0
        if not filtered_fund_flow.empty:
            total_interest = filtered_fund_flow["BrokerInterest"].sum()
        
        # Calculate comparison values if comparison is enabled
        if comparison_enabled and comp_tl is not None and not comp_tl.empty:
            # Prepare comparison timeline with same calculations as primary
            comp_tl = comp_tl.copy()
            comp_tl["Date"] = pd.to_datetime(comp_tl["Date"], dayfirst=True)
            comp_alloc_cols = [c for c in comp_tl.columns if c.startswith("Allocation_")]
            comp_tl["TotalAllocation"] = comp_tl[comp_alloc_cols].sum(axis=1)
            comp_tl["SpareCapital"] = comp_tl["BrokerBalance"] - comp_tl["TotalAllocation"]
            
            comp_today = comp_tl.iloc[-1]
            comp_spare = comp_today["SpareCapital"]
            comp_total_alloc = comp_today["TotalAllocation"]
            comp_broker = comp_today["BrokerBalance"]
            
            comp_total_interest = 0
            if comp_fund_flow is not None and not comp_fund_flow.empty:
                comp_total_interest = comp_fund_flow["BrokerInterest"].sum()
        else:
            comp_spare = None
            comp_total_alloc = None
            comp_broker = None
            comp_total_interest = None
        
        # Display metrics in 4-column layout with comparison data embedded
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_comparison_metric("💰 Spare Capital", spare, comp_spare, prefix="$")
            if not comparison_enabled or comp_spare is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Broker Balance - Total Allocation</div>', unsafe_allow_html=True)
        
        with col2:
            render_comparison_metric("📈 Total Allocated", total_alloc, comp_total_alloc, prefix="$")
            if not comparison_enabled or comp_total_alloc is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all strategy allocations</div>', unsafe_allow_html=True)
        
        with col3:
            render_comparison_metric("🏦 Broker Balance", broker, comp_broker, prefix="$")
            if not comparison_enabled or comp_broker is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total account balance</div>', unsafe_allow_html=True)
        
        with col4:
            render_comparison_metric("💵 Total Interest", total_interest, comp_total_interest, prefix="$")
            if not comparison_enabled or comp_total_interest is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Cumulative broker interest earned</div>', unsafe_allow_html=True)
        
        st.caption(f"*Data as of {today['Date'].strftime('%d‑%b‑%Y')}*")
        
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
    else:
        st.warning("⚠️ No timeline data available for the selected date range.")

    # ============================================================================
    # SECTION 2: TRADING PERFORMANCE ANALYSIS
    # ============================================================================
    st.markdown('<h2 class="section-header">📊 Trading Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_trades.empty:
        # Calculate trading metrics
        tr = filtered_trades.copy()
        tr["TradeDate_dt"] = pd.to_datetime(tr["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = tr.groupby("TradeDate_dt")["NetCash"].sum().sort_index()

        net_pnl    = daily_pnl.sum()
        
        # Calculate fund flow PnL for comparison
        from scripts.metrics import calculate_fund_flow_pnl
        fund_flow_pnl = calculate_fund_flow_pnl(
            start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
            end_date=end_date.strftime('%Y-%m-%d') if end_date else None
        )
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

        # Calculate comparison metrics if comparison is enabled
        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
            comp_tr = comp_trades.copy()
            comp_tr["TradeDate_dt"] = pd.to_datetime(comp_tr["TradeDate"], format="%d/%m/%y", dayfirst=True)
            comp_daily_pnl = comp_tr.groupby("TradeDate_dt")["NetCash"].sum().sort_index()
            
            comp_net_pnl = comp_daily_pnl.sum()
            comp_excess = comp_daily_pnl - rf_daily
            comp_sharpe = comp_excess.mean() / (comp_daily_pnl.std(ddof=0) or 1e-9) * np.sqrt(252)
            comp_gross_win = comp_daily_pnl[comp_daily_pnl>0].sum()
            comp_gross_loss = -comp_daily_pnl[comp_daily_pnl<0].sum()
            comp_pf = comp_gross_win / comp_gross_loss if comp_gross_loss else np.nan
            comp_win_days = int((comp_daily_pnl>0).sum())
            comp_loss_days = int((comp_daily_pnl<0).sum())
            comp_total_days = comp_win_days + comp_loss_days
            comp_win_days_pct = (comp_win_days / comp_total_days * 100) if comp_total_days > 0 else 0
            
            # Comparison trade-level metrics
            comp_tg = comp_tr.groupby(["UID", "TradeDate"])
            comp_total_trades, comp_win_trades, comp_loss_trades = 0, 0, 0
            for (_, _), legs in comp_tg:
                cnt = max(1, legs["Quantity"].max())
                comp_total_trades += cnt
                s = legs["NetCash"].sum()
                if s > 0: 
                    comp_win_trades += cnt
                elif s < 0: 
                    comp_loss_trades += cnt
            comp_win_trades_pct = (comp_win_trades / comp_total_trades * 100) if comp_total_trades > 0 else 0
        else:
            comp_net_pnl = None
            comp_sharpe = None
            comp_pf = None
            comp_win_days = None
            comp_loss_days = None
            comp_total_days = None
            comp_win_days_pct = None
            comp_total_trades = None
            comp_win_trades = None
            comp_loss_trades = None
            comp_win_trades_pct = None

        # Core Performance Metrics
        st.markdown('<h3 class="subsection-header">🎯 Core Performance Metrics</h3>', unsafe_allow_html=True)
        
        # Row 1: Core Performance Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_comparison_metric("💵 Net P&L", net_pnl, comp_net_pnl, prefix="$")
            if not comparison_enabled or comp_net_pnl is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From trade NetCash data</div>', unsafe_allow_html=True)
        
        with col2:
            render_comparison_metric("📈 Sharpe Ratio", sharpe, comp_sharpe, prefix="")
            if not comparison_enabled or comp_sharpe is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily P&L excess return / volatility × √252</div>', unsafe_allow_html=True)
        
        with col3:
            pf_display = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
            comp_pf_display = f"{comp_pf:.2f}" if comp_pf is not None and not np.isnan(comp_pf) else "N/A"
            render_comparison_metric("⚖️ Profit Factor", pf, comp_pf, prefix="")
            if not comparison_enabled or comp_pf is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross wins / Gross losses</div>', unsafe_allow_html=True)
        
        # Row 2: Day-based Metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
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
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with positive/negative daily P&L</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col5:
            render_comparison_metric("📊 Win Days %", win_days_pct, comp_win_days_pct, prefix="")
            if not comparison_enabled or comp_win_days_pct is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win days / Total trading days</div>', unsafe_allow_html=True)
        
        with col6:
            # For win/loss trades, we need to handle the combined display
            if comparison_enabled and comp_win_trades is not None:
                win_loss_trades_display = f"{win_trades} / {loss_trades}"
                comp_win_loss_trades_display = f"{comp_win_trades} / {comp_loss_trades}"
                pct_change = calculate_pct_change(win_trades, comp_win_trades)
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
                    <div class="metric-value neutral-value">{win_trades} / {loss_trades}</div>
                    <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Individual profitable/unprofitable trades</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Row 3: Additional Metrics
        col7, col8, col9 = st.columns(3)
        
        with col7:
            render_comparison_metric("📈 Win Trades %", win_trades_pct, comp_win_trades_pct, prefix="")
            if not comparison_enabled or comp_win_trades_pct is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win trades / Total trades</div>', unsafe_allow_html=True)
        
        with col8:
            render_comparison_metric("🔄 Total Trades", total_trades, comp_total_trades, prefix="")
            if not comparison_enabled or comp_total_trades is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Count of individual trade executions</div>', unsafe_allow_html=True)
        
        with col9:
            render_comparison_metric("📊 Trading Days", total_days, comp_total_days, prefix="")
            if not comparison_enabled or comp_total_days is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with trading activity</div>', unsafe_allow_html=True)
        
        # Calculate comparison fund flow PnL if comparison is enabled
        if comparison_enabled and comp_fund_flow is not None and not comp_fund_flow.empty:
            from scripts.metrics import calculate_fund_flow_pnl
            comp_fund_flow_pnl = calculate_fund_flow_pnl(
                start_date=comp_start_date.strftime('%Y-%m-%d') if comp_start_date else None,
                end_date=comp_end_date.strftime('%Y-%m-%d') if comp_end_date else None
            )
        else:
            comp_fund_flow_pnl = None

        # PnL Reconciliation Section
        st.markdown('<h3 class="subsection-header">💰 PnL Reconciliation</h3>', unsafe_allow_html=True)
        
        col_pnl1, col_pnl2 = st.columns(2)
        
        with col_pnl1:
            render_comparison_metric("📊 Trade-Based P&L", net_pnl, comp_net_pnl, prefix="$")
            if not comparison_enabled or comp_net_pnl is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From trade NetCash data</div>', unsafe_allow_html=True)
        
        with col_pnl2:
            render_comparison_metric("🏦 Fund Flow P&L", fund_flow_pnl["total_pnl"], comp_fund_flow_pnl["total_pnl"] if comp_fund_flow_pnl else None, prefix="$")
            if not comparison_enabled or comp_fund_flow_pnl is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From cash flow reconciliation</div>', unsafe_allow_html=True)
        
        # Show calculation details in expander
        with st.expander("📋 Fund Flow PnL Calculation Details", expanded=False):
            st.markdown('<h4 class="subsection-header">📊 Calculation Breakdown</h4>', unsafe_allow_html=True)
            st.write("**Formula:** PnL = Ending Cash - Starting Cash - Deposits/Withdrawals - OtherFees - Broker Interest")
            st.write(f"**Calculation:** {fund_flow_pnl['calculation']}")
            
            col_detail1, col_detail2, col_detail3, col_detail4, col_detail5 = st.columns(5)
            with col_detail1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">💰 Starting Cash</div>
                    <div class="metric-value neutral-value">${fund_flow_pnl['starting_cash']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_detail2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">💰 Ending Cash</div>
                    <div class="metric-value neutral-value">${fund_flow_pnl['ending_cash']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_detail3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">💸 Deposits/Withdrawals</div>
                    <div class="metric-value neutral-value">${fund_flow_pnl['total_deposits']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_detail4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">💸 Other Fees</div>
                    <div class="metric-value neutral-value">${fund_flow_pnl['total_fees']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_detail5:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">💵 Broker Interest</div>
                    <div class="metric-value neutral-value">${fund_flow_pnl['total_interest']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show difference if there is one
            difference = net_pnl - fund_flow_pnl["total_pnl"]
            if abs(difference) > 0.01:  # Only show if difference is significant
                st.warning(f"⚠️ **Difference between calculations:** ${difference:,.2f}")
                st.info("This difference could be due to timing differences, fees, or other cash flows not captured in trades.")
            else:
                st.success("✅ **Calculations match!** Both methods show the same PnL.")
        
        # Calculate comparison efficiency metrics if comparison is enabled
        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
            comp_avg_trade_return = comp_net_pnl / comp_total_trades if comp_total_trades > 0 else 0
            comp_avg_win_amount = comp_gross_win / comp_win_days if comp_win_days > 0 else 0
            comp_avg_loss_amount = comp_gross_loss / comp_loss_days if comp_loss_days > 0 else 0
            comp_expectancy = (comp_win_trades_pct/100 * comp_avg_win_amount) - ((100-comp_win_trades_pct)/100 * comp_avg_loss_amount)
        else:
            comp_avg_trade_return = None
            comp_avg_win_amount = None
            comp_avg_loss_amount = None
            comp_expectancy = None

        # Trading Efficiency Metrics
        st.markdown('<h3 class="subsection-header">📊 Trading Efficiency Metrics</h3>', unsafe_allow_html=True)
        
        # Calculate additional efficiency metrics
        avg_trade_return = net_pnl / total_trades if total_trades > 0 else 0
        avg_win_amount = gross_win / win_days if win_days > 0 else 0
        avg_loss_amount = gross_loss / loss_days if loss_days > 0 else 0
        expectancy = (win_trades_pct/100 * avg_win_amount) - ((100-win_trades_pct)/100 * avg_loss_amount)
        
        # Row 1: Efficiency Metrics
        col_eff1, col_eff2, col_eff3 = st.columns(3)
        
        with col_eff1:
            render_comparison_metric("📈 Avg Trade Return", avg_trade_return, comp_avg_trade_return, prefix="$")
            if not comparison_enabled or comp_avg_trade_return is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Net P&L / Total trades</div>', unsafe_allow_html=True)
        
        with col_eff2:
            render_comparison_metric("💰 Avg Win Amount", avg_win_amount, comp_avg_win_amount, prefix="$")
            if not comparison_enabled or comp_avg_win_amount is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross wins / Win days</div>', unsafe_allow_html=True)
        
        with col_eff3:
            render_comparison_metric("💸 Avg Loss Amount", avg_loss_amount, comp_avg_loss_amount, prefix="$")
            if not comparison_enabled or comp_avg_loss_amount is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross losses / Loss days</div>', unsafe_allow_html=True)
        
        # Calculate additional comparison metrics
        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
            comp_rr_ratio = comp_avg_win_amount / comp_avg_loss_amount if comp_avg_loss_amount > 0 else 0
            comp_breakeven_win_pct = 1 / (1 + comp_rr_ratio) * 100 if comp_rr_ratio > 0 else 0
        else:
            comp_rr_ratio = None
            comp_breakeven_win_pct = None

        # Row 2: Additional Efficiency Metrics
        col_eff4, col_eff5, col_eff6 = st.columns(3)
        
        with col_eff4:
            render_comparison_metric("📊 Expectancy per Trade", expectancy, comp_expectancy, prefix="$")
            if not comparison_enabled or comp_expectancy is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">(Win% × Avg Win) - (Loss% × Avg Loss)</div>', unsafe_allow_html=True)
        
        with col_eff5:
            rr_ratio = avg_win_amount / avg_loss_amount if avg_loss_amount > 0 else 0
            render_comparison_metric("⚖️ Reward/Risk Ratio", rr_ratio, comp_rr_ratio, prefix="")
            if not comparison_enabled or comp_rr_ratio is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Avg Win / Avg Loss</div>', unsafe_allow_html=True)
        
        with col_eff6:
            breakeven_win_pct = 1 / (1 + rr_ratio) * 100 if rr_ratio > 0 else 0
            render_comparison_metric("🎯 Break-even Win %", breakeven_win_pct, comp_breakeven_win_pct, prefix="")
            if not comparison_enabled or comp_breakeven_win_pct is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">1 / (1 + Reward/Risk ratio)</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No trading data available for the selected date range.")

    # ============================================================================
    # SECTION 3: NAV PERFORMANCE ANALYSIS
    # ============================================================================
    st.markdown('<h2 class="section-header">📈 NAV Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_nav.empty:
        nav = filtered_nav.copy()
        nav["Date"] = pd.to_datetime(nav["Date"], dayfirst=True)
        nav_ser = nav.set_index("Date")["NAV"].sort_index()
        
        # For filtered date ranges, add baseline NAV=100 on day before filter starts (like Strategy/UID pages)
        if start_date is not None and end_date is not None:
            # Find the most recent NAV before the filter starts (handles weekends/missing data)
            nav_df["Date"] = pd.to_datetime(nav_df["Date"], dayfirst=True)
            # Convert start_date to datetime for proper comparison
            start_datetime = pd.to_datetime(start_date)
            baseline_nav_data = nav_df[nav_df["Date"] < start_datetime]
            
            if not baseline_nav_data.empty:
                # Get the most recent NAV before the filter starts
                baseline_nav_data = baseline_nav_data.sort_values("Date")
                baseline_date = baseline_nav_data.iloc[-1]["Date"]
                baseline_nav_value = baseline_nav_data.iloc[-1]["NAV"]
                
                # Normalize using the baseline NAV
                nav_ser = (nav_ser / baseline_nav_value) * 100
                # Add baseline NAV=100 on the baseline date
                baseline_series = pd.Series([100.0], index=[baseline_date])
                nav_ser = pd.concat([baseline_series, nav_ser]).sort_index()
            else:
                # Fallback to original logic if no baseline found
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

        # Calculate comparison NAV metrics if comparison is enabled
        if comparison_enabled and comp_nav is not None and not comp_nav.empty:
            comp_nav_ser = comp_nav.set_index("Date")["NAV"].sort_index()
            comp_first_nav = comp_nav_ser.iloc[0]
            comp_current_nav = comp_nav_ser.iloc[-1]
            comp_max_nav = comp_nav_ser.max()
            comp_cummax_nav = comp_nav_ser.cummax()
            comp_dd_ser = comp_nav_ser / comp_cummax_nav - 1
            comp_current_dd = comp_dd_ser.iloc[-1]
            comp_max_dd = comp_dd_ser.min()
        else:
            comp_current_nav = None
            comp_max_nav = None
            comp_current_dd = None
            comp_max_dd = None

        # NAV Key Metrics
        st.markdown('<h3 class="subsection-header">📊 NAV Key Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_comparison_metric("📈 Current NAV", current_nav, comp_current_nav, prefix="")
            if not comparison_enabled or comp_current_nav is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest NAV value from portfolio data</div>', unsafe_allow_html=True)
        
        with col2:
            render_comparison_metric("🏆 Max NAV", max_nav, comp_max_nav, prefix="")
            if not comparison_enabled or comp_max_nav is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Highest NAV value achieved</div>', unsafe_allow_html=True)
        
        with col3:
            render_comparison_metric("📉 Current Drawdown", current_dd*100, comp_current_dd*100 if comp_current_dd is not None else None, prefix="")
            if not comparison_enabled or comp_current_dd is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">(Current NAV / Max NAV) - 1</div>', unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            render_comparison_metric("📉 Max Drawdown", max_dd*100, comp_max_dd*100 if comp_max_dd is not None else None, prefix="")
            if not comparison_enabled or comp_max_dd is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Largest peak-to-trough decline</div>', unsafe_allow_html=True)
        
        with col5:
            # Calculate comparison CAGR if available
            if comparison_enabled and comp_nav is not None and not comp_nav.empty:
                comp_days = (comp_nav_ser.index[-1] - comp_nav_ser.index[0]).days
                comp_cagr = ((comp_current_nav/comp_first_nav) ** (365.25/comp_days) - 1) if comp_days>0 else np.nan
            else:
                comp_cagr = None
            
            cagr_display = f"{cagr*100:,.2f}%" if not np.isnan(cagr) else "N/A"
            comp_cagr_display = f"{comp_cagr*100:,.2f}%" if comp_cagr is not None and not np.isnan(comp_cagr) else "N/A"
            render_comparison_metric("📊 CAGR", cagr*100 if not np.isnan(cagr) else 0, comp_cagr*100 if comp_cagr is not None and not np.isnan(comp_cagr) else 0, prefix="")
            if not comparison_enabled or comp_cagr is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Compound annual growth rate</div>', unsafe_allow_html=True)
        
        with col6:
            render_comparison_metric("📅 Analysis Period", days, comp_days if comparison_enabled and comp_nav is not None and not comp_nav.empty else None, prefix="")
            if not comparison_enabled or comp_nav is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total days in analysis period</div>', unsafe_allow_html=True)

        # Return-Based Metrics
        st.markdown('<h3 class="subsection-header">📈 Return-Based Metrics</h3>', unsafe_allow_html=True)
        
        # Calculate additional return metrics using existing logic patterns
        daily_returns = nav_ser.pct_change().dropna()
        
        # Monthly Returns
        monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        avg_monthly_return = monthly_returns.mean() * 100
        
        # Rolling Returns (7d and 30d)
        rolling_7d = daily_returns.rolling(7).apply(lambda x: (1 + x).prod() - 1)
        rolling_30d = daily_returns.rolling(30).apply(lambda x: (1 + x).prod() - 1)
        current_7d_return = rolling_7d.iloc[-1] * 100 if not rolling_7d.empty else 0
        current_30d_return = rolling_30d.iloc[-1] * 100 if not rolling_30d.empty else 0
        
        # Annualized Volatility
        annualized_vol = daily_returns.std() * np.sqrt(252) * 100
        
        # Sortino Ratio (using existing logic pattern)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
        sortino_ratio = (daily_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
        
        # Return Over Max Drawdown
        return_over_dd = (cagr / abs(max_dd)) if max_dd != 0 else np.nan
        
        # Calculate comparison return metrics if comparison is enabled
        if comparison_enabled and comp_nav is not None and not comp_nav.empty:
            comp_daily_returns = comp_nav_ser.pct_change().dropna()
            comp_monthly_returns = comp_daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            comp_avg_monthly_return = comp_monthly_returns.mean() * 100
            comp_rolling_7d = comp_daily_returns.rolling(7).apply(lambda x: (1 + x).prod() - 1)
            comp_rolling_30d = comp_daily_returns.rolling(30).apply(lambda x: (1 + x).prod() - 1)
            comp_current_7d_return = comp_rolling_7d.iloc[-1] * 100 if not comp_rolling_7d.empty else 0
            comp_current_30d_return = comp_rolling_30d.iloc[-1] * 100 if not comp_rolling_30d.empty else 0
        else:
            comp_avg_monthly_return = None
            comp_current_7d_return = None
            comp_current_30d_return = None

        # Row 1: Return Metrics
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            render_comparison_metric("📊 Avg Monthly Return", avg_monthly_return, comp_avg_monthly_return, prefix="")
            if not comparison_enabled or comp_avg_monthly_return is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average monthly NAV return</div>', unsafe_allow_html=True)
        
        with col_r2:
            render_comparison_metric("📈 7-Day Rolling Return", current_7d_return, comp_current_7d_return, prefix="")
            if not comparison_enabled or comp_current_7d_return is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative NAV return</div>', unsafe_allow_html=True)
        
        with col_r3:
            render_comparison_metric("📈 30-Day Rolling Return", current_30d_return, comp_current_30d_return, prefix="")
            if not comparison_enabled or comp_current_30d_return is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative NAV return</div>', unsafe_allow_html=True)
        
        # Calculate additional comparison metrics
        if comparison_enabled and comp_nav is not None and not comp_nav.empty:
            comp_annualized_vol = comp_daily_returns.std() * np.sqrt(252) * 100
            comp_downside_returns = comp_daily_returns[comp_daily_returns < 0]
            comp_downside_vol = comp_downside_returns.std() * np.sqrt(252) if len(comp_downside_returns) > 0 else 1e-9
            comp_sortino_ratio = (comp_daily_returns.mean() * 252) / comp_downside_vol if comp_downside_vol > 0 else 0
            comp_return_over_dd = (comp_cagr / abs(comp_max_dd)) if comp_max_dd != 0 else np.nan
        else:
            comp_annualized_vol = None
            comp_sortino_ratio = None
            comp_return_over_dd = None

        # Row 2: Additional Return Metrics
        col_r4, col_r5, col_r6 = st.columns(3)
        
        with col_r4:
            render_comparison_metric("📊 Annualized Volatility", annualized_vol, comp_annualized_vol, prefix="")
            if not comparison_enabled or comp_annualized_vol is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily NAV return std × √252</div>', unsafe_allow_html=True)
        
        with col_r5:
            render_comparison_metric("📈 Sortino Ratio", sortino_ratio, comp_sortino_ratio, prefix="")
            if not comparison_enabled or comp_sortino_ratio is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Return / Downside volatility</div>', unsafe_allow_html=True)
        
        with col_r6:
            render_comparison_metric("📊 Return/Max DD", return_over_dd if not np.isnan(return_over_dd) else 0, comp_return_over_dd if comp_return_over_dd is not None and not np.isnan(comp_return_over_dd) else 0, prefix="")
            if not comparison_enabled or comp_return_over_dd is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">CAGR / Max drawdown</div>', unsafe_allow_html=True)

        # NAV Performance Chart
        st.markdown('<h3 class="subsection-header">📈 NAV Performance Chart</h3>', unsafe_allow_html=True)
        
        # Create NAV chart with comparison if enabled
        if comparison_enabled and comp_nav is not None and not comp_nav.empty:
            # Calculate comparison NAV series
            comp_nav_ser = comp_nav.set_index("Date")["NAV"].sort_index()
            
            # Normalize comparison NAV to start at 100 for fair comparison
            if not comp_nav_ser.empty:
                comp_start_nav = comp_nav_ser.iloc[0]
                comp_nav_ser = (comp_nav_ser / comp_start_nav) * 100
            
            # Create figure with both series
            fig_nav = go.Figure()
            
            # Primary period NAV
            fig_nav.add_trace(go.Scatter(
                x=nav_ser.index,
                y=nav_ser.values,
                mode='lines',
                name='Primary Period',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Comparison period NAV
            fig_nav.add_trace(go.Scatter(
                x=comp_nav_ser.index,
                y=comp_nav_ser.values,
                mode='lines',
                name='Comparison Period',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig_nav.update_layout(
                title="NAV Performance Comparison",
                xaxis_title="Date",
                yaxis_title="NAV (Normalized to 100)",
                height=500,
                showlegend=True
            )
        else:
            # Original single period chart
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

    # ============================================================================
    # SECTION 4: ADVANCED PERFORMANCE METRICS
    # ============================================================================
    if not filtered_nav.empty or not filtered_trades.empty:
        with st.expander("📊 Advanced Performance Metrics", expanded=False):
            st.markdown('<h4 class="subsection-header">🧮 Cost Analysis</h4>', unsafe_allow_html=True)
            
            # Cost Metrics
            st.markdown('<h5 class="subsection-header">💰 Transaction Costs</h5>', unsafe_allow_html=True)
            
            if not filtered_trades.empty:
                # Calculate cost metrics using existing logic patterns
                tr = filtered_trades.copy()
                
                # Total Transaction Costs (using slippage data)
                total_slippage = tr["Slippage"].sum() if "Slippage" in tr.columns else 0
                
                # Cost-to-Return Ratio
                cost_return_ratio = (total_slippage / net_pnl * 100) if net_pnl != 0 else 0
                
                # Average Cost per Trade
                avg_cost_per_trade = total_slippage / total_trades if total_trades > 0 else 0
                
                # Calculate comparison cost metrics if comparison is enabled
                if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                    comp_total_slippage = comp_trades["Slippage"].sum() if "Slippage" in comp_trades.columns else 0
                    comp_cost_return_ratio = (comp_total_slippage / comp_net_pnl * 100) if comp_net_pnl != 0 else 0
                    comp_avg_cost_per_trade = comp_total_slippage / comp_total_trades if comp_total_trades > 0 else 0
                else:
                    comp_total_slippage = None
                    comp_cost_return_ratio = None
                    comp_avg_cost_per_trade = None

                # Display Cost Metrics
                col_cost1, col_cost2, col_cost3 = st.columns(3)
                with col_cost1:
                    render_comparison_metric("💰 Total Slippage", total_slippage, comp_total_slippage, prefix="$")
                    if not comparison_enabled or comp_total_slippage is None:
                        st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all trade slippage costs</div>', unsafe_allow_html=True)
                with col_cost2:
                    render_comparison_metric("📊 Cost/Return Ratio", cost_return_ratio, comp_cost_return_ratio, prefix="")
                    if not comparison_enabled or comp_cost_return_ratio is None:
                        st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Slippage / Net P&L × 100</div>', unsafe_allow_html=True)
                with col_cost3:
                    render_comparison_metric("💸 Avg Cost per Trade", avg_cost_per_trade, comp_avg_cost_per_trade, prefix="$")
                    if not comparison_enabled or comp_avg_cost_per_trade is None:
                        st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total slippage / Total trades</div>', unsafe_allow_html=True)
            else:
                st.info("Cost metrics require trade data.")
            
            # Rolling Performance Charts
            st.markdown('<h4 class="subsection-header">📈 Rolling Performance Analysis</h4>', unsafe_allow_html=True)
            
            if not filtered_trades.empty:
                try:
                    # Calculate rolling metrics using daily PnL
                    daily_pnl_series = filtered_trades.groupby("TradeDate")["NetCash"].sum()
                    daily_pnl_series.index = pd.to_datetime(daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                    daily_pnl_series = daily_pnl_series.sort_index()
                    
                    # Only show charts if we have enough data
                    if len(daily_pnl_series) > 7:
                        # Calculate rolling metrics
                        rolling_7d_pnl = daily_pnl_series.rolling(7).sum()
                        rolling_30d_pnl = daily_pnl_series.rolling(30).sum()
                        rolling_7d_avg = daily_pnl_series.rolling(7).mean()
                        rolling_30d_avg = daily_pnl_series.rolling(30).mean()
                        
                        # Calculate rolling Sharpe
                        rolling_7d_std = daily_pnl_series.rolling(7).std()
                        rolling_30d_std = daily_pnl_series.rolling(30).std()
                        rolling_7d_sharpe = (rolling_7d_avg / rolling_7d_std) * np.sqrt(252)
                        rolling_30d_sharpe = (rolling_30d_avg / rolling_30d_std) * np.sqrt(252)
                        # Handle division by zero by filling NaN values with 0
                        rolling_7d_sharpe = rolling_7d_sharpe.fillna(0)
                        rolling_30d_sharpe = rolling_30d_sharpe.fillna(0)
                        
                        # Display rolling charts
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.markdown('<h5 class="subsection-header">📈 7-Day Rolling P&L</h5>', unsafe_allow_html=True)
                            if len(rolling_7d_pnl.dropna()) > 0:
                                # Create 7-day rolling P&L chart with comparison if enabled
                                if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                                    # Calculate comparison rolling P&L
                                    comp_daily_pnl_series = comp_trades.groupby("TradeDate")["NetCash"].sum()
                                    comp_daily_pnl_series.index = pd.to_datetime(comp_daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                                    comp_daily_pnl_series = comp_daily_pnl_series.sort_index()
                                    comp_rolling_7d_pnl = comp_daily_pnl_series.rolling(7).sum()
                                    
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
                            
                            st.markdown('<h5 class="subsection-header">📈 7-Day Rolling Sharpe</h5>', unsafe_allow_html=True)
                            if len(rolling_7d_sharpe.dropna()) > 0:
                                # Create 7-day rolling Sharpe chart with comparison if enabled
                                if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                                    # Calculate comparison rolling Sharpe
                                    comp_daily_pnl_series = comp_trades.groupby("TradeDate")["NetCash"].sum()
                                    comp_daily_pnl_series.index = pd.to_datetime(comp_daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                                    comp_daily_pnl_series = comp_daily_pnl_series.sort_index()
                                    comp_rolling_7d_avg = comp_daily_pnl_series.rolling(7).mean()
                                    comp_rolling_7d_std = comp_daily_pnl_series.rolling(7).std()
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
                            st.markdown('<h5 class="subsection-header">📈 30-Day Rolling P&L</h5>', unsafe_allow_html=True)
                            if len(rolling_30d_pnl.dropna()) > 0:
                                # Create 30-day rolling P&L chart with comparison if enabled
                                if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                                    # Calculate comparison rolling P&L
                                    comp_daily_pnl_series = comp_trades.groupby("TradeDate")["NetCash"].sum()
                                    comp_daily_pnl_series.index = pd.to_datetime(comp_daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                                    comp_daily_pnl_series = comp_daily_pnl_series.sort_index()
                                    comp_rolling_30d_pnl = comp_daily_pnl_series.rolling(30).sum()
                                    
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
                            
                            st.markdown('<h5 class="subsection-header">📈 30-Day Rolling Sharpe</h5>', unsafe_allow_html=True)
                            if len(rolling_30d_sharpe.dropna()) > 0:
                                # Create 30-day rolling Sharpe chart with comparison if enabled
                                if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                                    # Calculate comparison rolling Sharpe
                                    comp_daily_pnl_series = comp_trades.groupby("TradeDate")["NetCash"].sum()
                                    comp_daily_pnl_series.index = pd.to_datetime(comp_daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                                    comp_daily_pnl_series = comp_daily_pnl_series.sort_index()
                                    comp_rolling_30d_avg = comp_daily_pnl_series.rolling(30).mean()
                                    comp_rolling_30d_std = comp_daily_pnl_series.rolling(30).std()
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
                        
                        # Calculate comparison rolling metrics if enabled
                        if comparison_enabled and comp_trades is not None and not comp_trades.empty:
                            comp_daily_pnl_series = comp_trades.groupby("TradeDate")["NetCash"].sum()
                            comp_daily_pnl_series.index = pd.to_datetime(comp_daily_pnl_series.index, format="%d/%m/%y", dayfirst=True)
                            comp_daily_pnl_series = comp_daily_pnl_series.sort_index()
                            
                            if len(comp_daily_pnl_series) > 7:
                                comp_rolling_7d_pnl = comp_daily_pnl_series.rolling(7).sum()
                                comp_rolling_30d_pnl = comp_daily_pnl_series.rolling(30).sum()
                                comp_rolling_7d_avg = comp_daily_pnl_series.rolling(7).mean()
                                comp_rolling_30d_avg = comp_daily_pnl_series.rolling(30).mean()
                                comp_rolling_7d_std = comp_daily_pnl_series.rolling(7).std()
                                comp_rolling_30d_std = comp_daily_pnl_series.rolling(30).std()
                                comp_rolling_7d_sharpe = (comp_rolling_7d_avg / comp_rolling_7d_std) * np.sqrt(252)
                                comp_rolling_30d_sharpe = (comp_rolling_30d_avg / comp_rolling_30d_std) * np.sqrt(252)
                                comp_rolling_7d_sharpe = comp_rolling_7d_sharpe.fillna(0)
                                comp_rolling_30d_sharpe = comp_rolling_30d_sharpe.fillna(0)
                                
                                comp_current_7d_pnl = comp_rolling_7d_pnl.iloc[-1] if not comp_rolling_7d_pnl.empty else 0
                                comp_current_30d_pnl = comp_rolling_30d_pnl.iloc[-1] if not comp_rolling_30d_pnl.empty else 0
                                comp_current_7d_sharpe = comp_rolling_7d_sharpe.iloc[-1] if not comp_rolling_7d_sharpe.empty else 0
                                comp_current_30d_sharpe = comp_rolling_30d_sharpe.iloc[-1] if not comp_rolling_30d_sharpe.empty else 0
                            else:
                                comp_current_7d_pnl = None
                                comp_current_30d_pnl = None
                                comp_current_7d_sharpe = None
                                comp_current_30d_sharpe = None
                        else:
                            comp_current_7d_pnl = None
                            comp_current_30d_pnl = None
                            comp_current_7d_sharpe = None
                            comp_current_30d_sharpe = None

                        # Display current rolling metrics
                        st.markdown('<h5 class="subsection-header">📊 Current Rolling Metrics</h5>', unsafe_allow_html=True)
                        col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                        with col_roll1:
                            current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty else 0
                            render_comparison_metric("📈 7-Day P&L", current_7d_pnl, comp_current_7d_pnl, prefix="$")
                        with col_roll2:
                            current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty else 0
                            render_comparison_metric("📈 30-Day P&L", current_30d_pnl, comp_current_30d_pnl, prefix="$")
                        with col_roll3:
                            current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty else 0
                            render_comparison_metric("📊 7-Day Sharpe", current_7d_sharpe, comp_current_7d_sharpe, prefix="")
                        with col_roll4:
                            current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty else 0
                            render_comparison_metric("📊 30-Day Sharpe", current_30d_sharpe, comp_current_30d_sharpe, prefix="")
                    else:
                        st.info("Insufficient data for rolling performance charts in selected period.")
                except Exception as e:
                    st.info("Insufficient data for rolling performance charts in selected period.")
            else:
                st.info("Rolling performance charts require trade data.")
    else:
        st.warning("⚠️ No NAV data available for the selected date range.")

    # ============================================================================
    # SECTION 5: RISK ANALYSIS
    # ============================================================================
    st.markdown('<h2 class="section-header">🛡️ Risk Analysis</h2>', unsafe_allow_html=True)
    
    if not filtered_tl.empty and not filtered_nav.empty:
        # Calculate risk metrics using existing logic patterns
        daily_returns = nav_ser.pct_change().dropna()
        
        # Average Drawdown
        drawdown_episodes = dd_ser[dd_ser < 0]
        avg_drawdown = drawdown_episodes.mean() * 100 if len(drawdown_episodes) > 0 else 0
        
        # Drawdown Duration (simplified calculation)
        peak_indices = (dd_ser == 0).astype(int)
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
        
        # Calculate comparison risk metrics if comparison is enabled
        if comparison_enabled and comp_nav is not None and not comp_nav.empty:
            comp_daily_returns = comp_nav_ser.pct_change().dropna()
            comp_drawdown_episodes = comp_dd_ser[comp_dd_ser < 0]
            comp_avg_drawdown = comp_drawdown_episodes.mean() * 100 if len(comp_drawdown_episodes) > 0 else 0
            comp_peak_indices = (comp_dd_ser == 0).astype(int)
            comp_recovery_periods = comp_peak_indices.groupby(comp_peak_indices.cumsum()).cumcount()
            comp_max_dd_duration = comp_recovery_periods.max()
            comp_var_95 = np.percentile(comp_daily_returns, 5) * 100
            comp_cvar_95 = comp_daily_returns[comp_daily_returns <= np.percentile(comp_daily_returns, 5)].mean() * 100
            comp_win_rate = len(comp_daily_returns[comp_daily_returns > 0]) / len(comp_daily_returns)
            comp_avg_win = comp_daily_returns[comp_daily_returns > 0].mean()
            comp_avg_loss = abs(comp_daily_returns[comp_daily_returns < 0].mean())
            comp_rr_ratio = comp_avg_win / comp_avg_loss if comp_avg_loss != 0 else 0
            comp_kelly = comp_win_rate - ((1 - comp_win_rate) / comp_rr_ratio) if comp_rr_ratio > 0 else 0
        else:
            comp_avg_drawdown = None
            comp_max_dd_duration = None
            comp_var_95 = None
            comp_cvar_95 = None
            comp_win_rate = None
            comp_kelly = None
        
        # Risk Metrics
        st.markdown('<h3 class="subsection-header">📊 Risk Metrics</h3>', unsafe_allow_html=True)
        
        # Row 1: Core Risk Metrics
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            render_comparison_metric("📉 Average Drawdown", avg_drawdown, comp_avg_drawdown, prefix="")
            if not comparison_enabled or comp_avg_drawdown is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Mean of all drawdown periods</div>', unsafe_allow_html=True)
        
        with col_risk2:
            render_comparison_metric("⏱️ Max DD Duration", max_dd_duration, comp_max_dd_duration, prefix="")
            if not comparison_enabled or comp_max_dd_duration is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Longest recovery period</div>', unsafe_allow_html=True)
        
        with col_risk3:
            render_comparison_metric("📊 Win Rate", win_rate*100, comp_win_rate*100 if comp_win_rate is not None else None, prefix="")
            if not comparison_enabled or comp_win_rate is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with positive NAV returns</div>', unsafe_allow_html=True)
        
        # Row 2: Advanced Risk Metrics
        col_risk4, col_risk5, col_risk6 = st.columns(3)
        
        with col_risk4:
            render_comparison_metric("⚠️ VaR (95%)", var_95, comp_var_95, prefix="")
            if not comparison_enabled or comp_var_95 is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">95th percentile of daily returns</div>', unsafe_allow_html=True)
        
        with col_risk5:
            render_comparison_metric("⚠️ CVaR (95%)", cvar_95, comp_cvar_95, prefix="")
            if not comparison_enabled or comp_cvar_95 is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected loss beyond VaR</div>', unsafe_allow_html=True)
        
        with col_risk6:
            render_comparison_metric("🎯 Kelly Criterion", kelly, comp_kelly, prefix="")
            if not comparison_enabled or comp_kelly is None:
                st.markdown('<div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Optimal leverage ratio</div>', unsafe_allow_html=True)
        
        # Rolling Drawdown Analysis
        st.markdown('<h3 class="subsection-header">📉 Rolling Drawdown Analysis</h3>', unsafe_allow_html=True)
        
        # Check if we have enough data for charts
        if len(filtered_tl) > 1:
            try:
                broker_balance = filtered_tl.set_index("Date")["BrokerBalance"]
                drawdown = (broker_balance / broker_balance.cummax() - 1) * 100
                
                # Create drawdown chart with comparison if enabled
                if comparison_enabled and comp_tl is not None and not comp_tl.empty:
                    # Calculate comparison drawdown
                    comp_broker_balance = comp_tl.set_index("Date")["BrokerBalance"]
                    comp_drawdown = (comp_broker_balance / comp_broker_balance.cummax() - 1) * 100
                    
                    # Create figure with both series
                    fig_dd = go.Figure()
                    
                    # Primary period drawdown
                    fig_dd.add_trace(go.Scatter(
                        x=drawdown.index,
                        y=drawdown.values,
                        mode='lines',
                        name=f'Primary Period',
                        line=dict(color='red', width=2),
                        fill='tonexty'
                    ))
                    
                    # Comparison period drawdown
                    fig_dd.add_trace(go.Scatter(
                        x=comp_drawdown.index,
                        y=comp_drawdown.values,
                        mode='lines',
                        name=f'Comparison Period',
                        line=dict(color='orange', width=2, dash='dash'),
                        fill='tonexty'
                    ))
                    
                    fig_dd.update_layout(
                        title="Rolling Drawdown (%)",
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
                        fig_dd.update_layout(yaxis=dict(range=y_range))
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                else:
                    # Original single period chart
                    fig_dd = px.area(drawdown, title="Rolling Drawdown (%)")
                    fig_dd.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
            except Exception as e:
                st.info("Insufficient data for drawdown chart in selected period.")

            # Margin Headroom Analysis
            st.markdown('<h3 class="subsection-header">🛡️ Margin Headroom Analysis</h3>', unsafe_allow_html=True)
            
            try:
                headroom = filtered_tl["SpareCapital"] / filtered_tl["BrokerBalance"] * 100
                
                # Create margin headroom chart with comparison if enabled
                if comparison_enabled and comp_tl is not None and not comp_tl.empty:
                    # Calculate comparison margin headroom
                    comp_headroom = comp_tl["SpareCapital"] / comp_tl["BrokerBalance"] * 100
                    
                    # Create figure with both series
                    fig_headroom = go.Figure()
                    
                    # Primary period headroom
                    fig_headroom.add_trace(go.Scatter(
                        x=filtered_tl["Date"],
                        y=headroom,
                        mode='lines',
                        name=f'Primary Period',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Comparison period headroom
                    fig_headroom.add_trace(go.Scatter(
                        x=comp_tl["Date"],
                        y=comp_headroom,
                        mode='lines',
                        name=f'Comparison Period',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    fig_headroom.update_layout(
                        title="Margin Headroom (%)",
                        xaxis_title="Date",
                        yaxis_title="Spare Capital / Broker Balance (%)",
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
                    all_data = pd.concat([headroom, comp_headroom])
                    if pd.notna(all_data.min()) and pd.notna(all_data.max()):
                        y_range = [all_data.min() * 0.95, all_data.max() * 1.05]
                        fig_headroom.update_layout(yaxis=dict(range=y_range))
                    
                    st.plotly_chart(fig_headroom, use_container_width=True)
                else:
                    # Original single period chart
                    fig_headroom = px.line(filtered_tl, x="Date", y=headroom, title="Margin Headroom (%)")
                    fig_headroom.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Spare Capital / Broker Balance (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_headroom, use_container_width=True)
            except Exception as e:
                st.info("Insufficient data for margin headroom chart in selected period.")
        else:
            st.info("Insufficient data for charts in selected period.")
    else:
        st.warning("⚠️ No timeline or NAV data available for risk analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 2rem 0;">
        📊 IBKR Portfolio Analytics Dashboard | Generated on {date} | 
        Data Source: Interactive Brokers Trade Ledger & Fund Flow Statements
    </div>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
