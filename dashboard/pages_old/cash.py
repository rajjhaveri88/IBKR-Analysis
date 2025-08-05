import streamlit as st, pandas as pd, plotly.express as px
import numpy as np
from scripts.loaders import load_fund_flow

def render() -> None:
    st.title("💰 Cash Movements")
    
    # Load fund flow data
    fund_flow = load_fund_flow()
    cash = fund_flow[["ToDate", "Deposit/Withdrawals", "BrokerInterest", "OtherFees"]].copy()
    cash["NetFlow"] = cash["Deposit/Withdrawals"] + cash["BrokerInterest"] + cash["OtherFees"]
    
    # Basic cash flow chart
    fig = px.bar(cash, x="ToDate", y="NetFlow", title="Cash Flows")
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Cash Flow Metrics
    with st.expander("📊 Advanced Cash Flow Metrics", expanded=False):
        st.markdown('<h4 class="subsection-header">💰 Cash Flow Summary</h4>', unsafe_allow_html=True)
        
        # Calculate cash flow metrics
        total_deposits = cash["Deposit/Withdrawals"].sum()
        total_interest = cash["BrokerInterest"].sum()
        total_fees = cash["OtherFees"].sum()
        total_net_flow = cash["NetFlow"].sum()
        
        # Monthly cash flow analysis
        cash["ToDate"] = pd.to_datetime(cash["ToDate"], dayfirst=True)
        monthly_flows = cash.groupby(cash["ToDate"].dt.to_period('M'))["NetFlow"].sum()
        avg_monthly_flow = monthly_flows.mean()
        
        # Cash flow volatility
        cash_flow_volatility = cash["NetFlow"].std()
        
        # Largest cash movements
        largest_deposit = cash["Deposit/Withdrawals"].max()
        largest_withdrawal = cash["Deposit/Withdrawals"].min()
        largest_interest = cash["BrokerInterest"].max()
        largest_fee = cash["OtherFees"].max()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deposits", f"${total_deposits:,.2f}")
            st.metric("Total Interest", f"${total_interest:,.2f}")
        with col2:
            st.metric("Total Fees", f"${total_fees:,.2f}")
            st.metric("Net Cash Flow", f"${total_net_flow:,.2f}")
        with col3:
            st.metric("Avg Monthly Flow", f"${avg_monthly_flow:,.2f}")
            st.metric("Flow Volatility", f"${cash_flow_volatility:,.2f}")
        with col4:
            st.metric("Largest Deposit", f"${largest_deposit:,.2f}")
            st.metric("Largest Withdrawal", f"${largest_withdrawal:,.2f}")
        
        st.markdown('<h4 class="subsection-header">📈 Cash Flow Trends</h4>', unsafe_allow_html=True)
        
        # Monthly trends chart
        monthly_trends = cash.groupby(cash["ToDate"].dt.to_period('M')).agg({
            "Deposit/Withdrawals": "sum",
            "BrokerInterest": "sum", 
            "OtherFees": "sum",
            "NetFlow": "sum"
        }).reset_index()
        monthly_trends["ToDate"] = monthly_trends["ToDate"].astype(str)
        
        fig_trends = px.line(monthly_trends, x="ToDate", y=["Deposit/Withdrawals", "BrokerInterest", "OtherFees", "NetFlow"],
                            title="Monthly Cash Flow Trends")
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Rolling cash flow analysis
        cash_sorted = cash.sort_values("ToDate")
        rolling_30d_flow = cash_sorted["NetFlow"].rolling(30).sum()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("30-Day Rolling Net Flow")
            if len(rolling_30d_flow.dropna()) > 0:
                st.line_chart(rolling_30d_flow, use_container_width=True)
            else:
                st.info("Insufficient data for 30-day rolling flow chart")
        
        with col_chart2:
            st.subheader("Cash Flow Distribution")
            fig_dist = px.histogram(cash, x="NetFlow", title="Distribution of Daily Cash Flows")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Cash flow efficiency metrics
        st.markdown('<h4 class="subsection-header">📊 Cash Flow Efficiency</h4>', unsafe_allow_html=True)
        
        # Calculate efficiency metrics
        positive_flows = cash[cash["NetFlow"] > 0]["NetFlow"].sum()
        negative_flows = abs(cash[cash["NetFlow"] < 0]["NetFlow"].sum())
        flow_efficiency = positive_flows / (positive_flows + negative_flows) * 100 if (positive_flows + negative_flows) > 0 else 0
        
        # Interest efficiency
        interest_efficiency = (total_interest / total_deposits * 100) if total_deposits > 0 else 0
        
        # Fee impact
        fee_impact = (total_fees / total_net_flow * 100) if total_net_flow != 0 else 0
        
        col_eff1, col_eff2, col_eff3 = st.columns(3)
        with col_eff1:
            st.metric("Flow Efficiency", f"{flow_efficiency:.1f}%")
        with col_eff2:
            st.metric("Interest Efficiency", f"{interest_efficiency:.2f}%")
        with col_eff3:
            st.metric("Fee Impact", f"{fee_impact:.2f}%")
    
    # Display raw data
    st.dataframe(cash) 