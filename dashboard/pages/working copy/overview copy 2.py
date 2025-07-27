# dashboard/pages/overview.py             ✓ 2025‑07‑25
# ─────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

from scripts.loaders import (
    load_timeline,
    load_fund_flow,
)
from scripts.metrics import account_kpis

# ── page title & layout ─────────────────────────────────
st.title("📈 Account Overview")

# ── load data ───────────────────────────────────────────
tl  = load_timeline()            # processed timeline.parquet
ff  = load_fund_flow()           # raw Fund Flow Statement.csv
kpi = account_kpis(ff, tl)       # Series (one row)

if tl.empty or kpi.empty:
    st.warning("No timeline or fund‑flow data available.")
    st.stop()

# today's row
today = tl.iloc[-1]
date_label = pd.to_datetime(today["Date"]).strftime("%d‑%b‑%Y")

# Spare = BrokerBalance ‑ Σ Allocation_*
alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
total_allocated = today[alloc_cols].sum() if alloc_cols else 0.0
spare_today     = today["BrokerBalance"] - total_allocated

# ── first metric row ───────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### Spare Today")
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{spare_today:,.0f}</h2>", unsafe_allow_html=True)

with c2:
    st.markdown("#### Total Allocated")
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{total_allocated:,.0f}</h2>", unsafe_allow_html=True)

with c3:
    st.markdown("#### Broker Balance")
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{today['BrokerBalance']:,.0f}</h2>", unsafe_allow_html=True)

st.caption(f"Date  {date_label}")

# ── second metric row (performance) ─────────────────────
c4, c5, c6 = st.columns(3)

with c4:
    st.markdown("#### XIRR")
    xirr = kpi["XIRR"]
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{xirr*100:,.2f} %</h2>"
                if pd.notna(xirr) else "N/A", unsafe_allow_html=True)

with c5:
    st.markdown("#### Max DD")
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{kpi['MaxDD_pct']:,.2f} %</h2>"
                if pd.notna(kpi['MaxDD_pct']) else "N/A", unsafe_allow_html=True)

with c6:
    st.markdown("#### Net PnL")
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{kpi['NetPnL']:,.0f}</h2>", unsafe_allow_html=True)

# ── third metric row (ratios) ───────────────────────────
c7, c8, c9 = st.columns(3)

with c7:
    st.markdown("#### Sharpe")
    sharpe = kpi["Sharpe"]
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{sharpe:.2f}</h2>"
                if pd.notna(sharpe) else "N/A", unsafe_allow_html=True)

with c8:
    st.markdown("#### Profit Factor")
    pf = kpi["ProfitFactor"]
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{pf:.2f}</h2>"
                if pd.notna(pf) else "N/A", unsafe_allow_html=True)

with c9:
    st.markdown("#### Win / Loss Days")
    wins = int(kpi["WinDays"]); losses = int(kpi["LossDays"])
    st.markdown(f"<h2 style='margin-top:-0.5rem'>{wins} / {losses}</h2>", unsafe_allow_html=True)

# ── optional: raw KPI expander ──────────────────────────
with st.expander("Show raw KPI series"):
    st.write(kpi)
