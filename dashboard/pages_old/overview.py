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
        
        # Display metrics in a professional grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💰 Spare Capital</div>
                <div class="metric-value">${spare:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Broker Balance - Total Allocation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Total Allocated</div>
                <div class="metric-value">${total_alloc:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Sum of all strategy allocations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🏦 Broker Balance</div>
                <div class="metric-value">${broker:,.0f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total account balance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💵 Total Interest</div>
                <div class="metric-value">${total_interest:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Cumulative broker interest earned</div>
            </div>
            """, unsafe_allow_html=True)
        
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

        # Core Performance Metrics
        st.markdown('<h3 class="subsection-header">🎯 Core Performance Metrics</h3>', unsafe_allow_html=True)
        
        # Row 1: Core Performance Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pnl_color = "positive-value" if net_pnl >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💵 Net P&L</div>
                <div class="metric-value {pnl_color}">${net_pnl:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">From trade NetCash data</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe_color = "positive-value" if sharpe >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Sharpe Ratio</div>
                <div class="metric-value {sharpe_color}">{sharpe:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily P&L excess return / volatility × √252</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pf_color = "positive-value" if pf > 1 else "negative-value" if pf < 1 else "neutral-value"
            pf_display = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚖️ Profit Factor</div>
                <div class="metric-value {pf_color}">{pf_display}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross wins / Gross losses</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Day-based Metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Win/Loss Days</div>
                <div class="metric-value neutral-value">{win_days} / {loss_days}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with positive/negative daily P&L</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Win Days %</div>
                <div class="metric-value neutral-value">{win_days_pct:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win days / Total trading days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
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
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Win Trades %</div>
                <div class="metric-value neutral-value">{win_trades_pct:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Win trades / Total trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🔄 Total Trades</div>
                <div class="metric-value neutral-value">{total_trades}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Count of individual trade executions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col9:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Trading Days</div>
                <div class="metric-value neutral-value">{total_days}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with trading activity</div>
            </div>
            """, unsafe_allow_html=True)
        
        # PnL Reconciliation Section
        st.markdown('<h3 class="subsection-header">💰 PnL Reconciliation</h3>', unsafe_allow_html=True)
        
        col_pnl1, col_pnl2 = st.columns(2)
        
        with col_pnl1:
            pnl_color = "positive-value" if net_pnl >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Trade-Based P&L</div>
                <div class="metric-value {pnl_color}">${net_pnl:,.2f}</div>
                <div class="metric-caption">From trade NetCash data</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_pnl2:
            ff_pnl_color = "positive-value" if fund_flow_pnl["total_pnl"] >= 0 else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🏦 Fund Flow P&L</div>
                <div class="metric-value {ff_pnl_color}">${fund_flow_pnl["total_pnl"]:,.2f}</div>
                <div class="metric-caption">From cash flow reconciliation</div>
            </div>
            """, unsafe_allow_html=True)
        
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
                <div class="metric-label">💰 Avg Win Amount</div>
                <div class="metric-value positive-value">${avg_win_amount:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross wins / Win days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">💸 Avg Loss Amount</div>
                <div class="metric-value negative-value">${avg_loss_amount:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Gross losses / Loss days</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Additional Efficiency Metrics
        col_eff4, col_eff5, col_eff6 = st.columns(3)
        
        with col_eff4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Expectancy per Trade</div>
                <div class="metric-value neutral-value">${expectancy:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">(Win% × Avg Win) - (Loss% × Avg Loss)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff5:
            rr_ratio = avg_win_amount / avg_loss_amount if avg_loss_amount > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚖️ Reward/Risk Ratio</div>
                <div class="metric-value neutral-value">{rr_ratio:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Avg Win / Avg Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eff6:
            breakeven_win_pct = 1 / (1 + rr_ratio) * 100 if rr_ratio > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Break-even Win %</div>
                <div class="metric-value neutral-value">{breakeven_win_pct:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">1 / (1 + Reward/Risk ratio)</div>
            </div>
            """, unsafe_allow_html=True)
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

        # NAV Key Metrics
        st.markdown('<h3 class="subsection-header">📊 NAV Key Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nav_color = "positive-value" if current_nav >= first_nav else "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Current NAV</div>
                <div class="metric-value {nav_color}">{current_nav:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Latest NAV value from portfolio data</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🏆 Max NAV</div>
                <div class="metric-value neutral-value">{max_nav:,.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Highest NAV value achieved</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dd_color = "negative-value" if current_dd < 0 else "positive-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📉 Current Drawdown</div>
                <div class="metric-value {dd_color}">{current_dd*100:,.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">(Current NAV / Max NAV) - 1</div>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            max_dd_color = "negative-value"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📉 Max Drawdown</div>
                <div class="metric-value {max_dd_color}">{max_dd*100:,.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Largest peak-to-trough decline</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            cagr_color = "positive-value" if cagr >= 0 else "negative-value"
            cagr_display = f"{cagr*100:,.2f}%" if not np.isnan(cagr) else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 CAGR</div>
                <div class="metric-value {cagr_color}">{cagr_display}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Compound annual growth rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📅 Analysis Period</div>
                <div class="metric-value neutral-value">{days} days</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Total days in analysis period</div>
            </div>
            """, unsafe_allow_html=True)

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
        
        # Row 1: Return Metrics
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Avg Monthly Return</div>
                <div class="metric-value neutral-value">{avg_monthly_return:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Average monthly NAV return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 7-Day Rolling Return</div>
                <div class="metric-value neutral-value">{current_7d_return:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">7-day cumulative NAV return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 30-Day Rolling Return</div>
                <div class="metric-value neutral-value">{current_30d_return:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">30-day cumulative NAV return</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Additional Return Metrics
        col_r4, col_r5, col_r6 = st.columns(3)
        
        with col_r4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Annualized Volatility</div>
                <div class="metric-value neutral-value">{annualized_vol:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Daily NAV return std × √252</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📈 Sortino Ratio</div>
                <div class="metric-value neutral-value">{sortino_ratio:.2f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Return / Downside volatility</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r6:
            rodd_display = f"{return_over_dd:.2f}" if not np.isnan(return_over_dd) else "N/A"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Return/Max DD</div>
                <div class="metric-value neutral-value">{rodd_display}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">CAGR / Max drawdown</div>
            </div>
            """, unsafe_allow_html=True)

        # NAV Performance Chart
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
                
                # Display Cost Metrics
                col_cost1, col_cost2, col_cost3 = st.columns(3)
                with col_cost1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">💰 Total Slippage</div>
                        <div class="metric-value neutral-value">${total_slippage:,.2f}</div>
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
                                st.line_chart(rolling_7d_pnl, use_container_width=True)
                            else:
                                st.info("Insufficient data for 7-day rolling P&L chart")
                            
                            st.markdown('<h5 class="subsection-header">📈 7-Day Rolling Sharpe</h5>', unsafe_allow_html=True)
                            if len(rolling_7d_sharpe.dropna()) > 0:
                                st.line_chart(rolling_7d_sharpe, use_container_width=True)
                            else:
                                st.info("Insufficient data for 7-day rolling Sharpe chart")
                        
                        with col_chart2:
                            st.markdown('<h5 class="subsection-header">📈 30-Day Rolling P&L</h5>', unsafe_allow_html=True)
                            if len(rolling_30d_pnl.dropna()) > 0:
                                st.line_chart(rolling_30d_pnl, use_container_width=True)
                            else:
                                st.info("Insufficient data for 30-day rolling P&L chart")
                            
                            st.markdown('<h5 class="subsection-header">📈 30-Day Rolling Sharpe</h5>', unsafe_allow_html=True)
                            if len(rolling_30d_sharpe.dropna()) > 0:
                                st.line_chart(rolling_30d_sharpe, use_container_width=True)
                            else:
                                st.info("Insufficient data for 30-day rolling Sharpe chart")
                        
                        # Display current rolling metrics
                        st.markdown('<h5 class="subsection-header">📊 Current Rolling Metrics</h5>', unsafe_allow_html=True)
                        col_roll1, col_roll2, col_roll3, col_roll4 = st.columns(4)
                        with col_roll1:
                            current_7d_pnl = rolling_7d_pnl.iloc[-1] if not rolling_7d_pnl.empty else 0
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">📈 7-Day P&L</div>
                                <div class="metric-value neutral-value">${current_7d_pnl:,.0f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_roll2:
                            current_30d_pnl = rolling_30d_pnl.iloc[-1] if not rolling_30d_pnl.empty else 0
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">📈 30-Day P&L</div>
                                <div class="metric-value neutral-value">${current_30d_pnl:,.0f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_roll3:
                            current_7d_sharpe = rolling_7d_sharpe.iloc[-1] if not rolling_7d_sharpe.empty else 0
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">📊 7-Day Sharpe</div>
                                <div class="metric-value neutral-value">{current_7d_sharpe:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_roll4:
                            current_30d_sharpe = rolling_30d_sharpe.iloc[-1] if not rolling_30d_sharpe.empty else 0
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">📊 30-Day Sharpe</div>
                                <div class="metric-value neutral-value">{current_30d_sharpe:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
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
        
        # Risk Metrics
        st.markdown('<h3 class="subsection-header">📊 Risk Metrics</h3>', unsafe_allow_html=True)
        
        # Row 1: Core Risk Metrics
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📉 Average Drawdown</div>
                <div class="metric-value negative-value">{avg_drawdown:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Mean of all drawdown periods</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⏱️ Max DD Duration</div>
                <div class="metric-value neutral-value">{max_dd_duration} days</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Longest recovery period</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">📊 Win Rate</div>
                <div class="metric-value neutral-value">{win_rate*100:.1f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Days with positive NAV returns</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Advanced Risk Metrics
        col_risk4, col_risk5, col_risk6 = st.columns(3)
        
        with col_risk4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚠️ VaR (95%)</div>
                <div class="metric-value negative-value">{var_95:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">95th percentile of daily returns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk5:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">⚠️ CVaR (95%)</div>
                <div class="metric-value negative-value">{cvar_95:.2f}%</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Expected loss beyond VaR</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk6:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">🎯 Kelly Criterion</div>
                <div class="metric-value neutral-value">{kelly:.3f}</div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 0.25rem;">Optimal leverage ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Rolling Drawdown Analysis
        st.markdown('<h3 class="subsection-header">📉 Rolling Drawdown Analysis</h3>', unsafe_allow_html=True)
        
        # Check if we have enough data for charts
        if len(filtered_tl) > 1:
            try:
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
            except Exception as e:
                st.info("Insufficient data for drawdown chart in selected period.")

            # Margin Headroom Analysis
            st.markdown('<h3 class="subsection-header">🛡️ Margin Headroom Analysis</h3>', unsafe_allow_html=True)
            
            try:
                headroom = filtered_tl["SpareCapital"] / filtered_tl["BrokerBalance"] * 100
                
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
