# scripts/metrics.py   ✓ 2025‑07‑25
# ──────────────────────────────────────────────────────────────
"""
KPI / analytics helpers for the IBKR‑Analysis dashboard.

• account_kpis(): overall NAV‑based metrics with cash‑flow IRR
• strategy_kpis(): per‑strategy KPIs (Trades, WinRate, ProfitFactor,
                    NetPnL, AvgSlip, Sharpe, XIRR)
"""

from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.loaders import (
    load_nav,
    load_fund_flow,
    load_trades,
    load_strategies,
    load_timeline,
)

# ── math helpers ────────────────────────────────────────────
def _years_between(d1: datetime, d2: datetime) -> float:
    return (d2 - d1).days / 365.25

def _xirr(cashflows: list[tuple[datetime, float]], guess: float = 0.1) -> float:
    """Newton–Raphson IRR solver for irregular cash‑flows."""
    if len(cashflows) < 2:
        return np.nan
    t0 = cashflows[0][0]
    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** _years_between(t0, dt) for dt, cf in cashflows)
    rate = guess
    for _ in range(100):
        f = npv(rate)
        df = sum(
            -_years_between(t0, dt) * cf / (1 + rate) ** (_years_between(t0, dt) + 1)
            for dt, cf in cashflows
        )
        if df == 0:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < 1e-7:
            return new_rate
        rate = new_rate
    return np.nan

def _annualised(series: pd.Series) -> float:
    """CAGR from first to last point, requiring ≥30 days."""
    if series.empty or series.iloc[0] == 0:
        return np.nan
    days = (series.index[-1] - series.index[0]).days
    if days < 30:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1

def _sharpe(daily_ret: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe (√252) from daily returns."""
    sd = daily_ret.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (daily_ret.mean() - rf) / sd * np.sqrt(252)

# ╔═════════ ACCOUNT‑LEVEL KPIs (NAV‑based) ═══════════════════╗
def account_kpis(
    fund_flow: pd.DataFrame | None = None,
    timeline:  pd.DataFrame | None = None,
) -> pd.Series:
    # Load data
    ff     = fund_flow   if fund_flow   is not None else load_fund_flow()
    nav_df = load_nav()
    if nav_df.empty or ff.empty:
        return pd.Series(dtype=float)

    # Prepare NAV series
    nav_df["Date"] = pd.to_datetime(nav_df["Date"], dayfirst=True)
    nav = nav_df.set_index("Date")["NAV"].sort_index()

    # Performance metrics
    daily_ret = nav.pct_change().dropna()
    net_pnl   = nav.iloc[-1] - nav.iloc[0]
    cagr_val  = _annualised(nav)

    # Max draw‑down on NAV
    cum_max = nav.cummax()
    max_dd  = ((nav - cum_max) / cum_max).min()

    # Win/Loss days & Profit Factor
    wins   = daily_ret[daily_ret > 0]
    losses = daily_ret[daily_ret < 0].abs()
    pf     = wins.sum() / losses.sum() if not losses.empty else np.nan

    # Build cash‑flows for IRR: deposits = negative, withdrawals = positive
    ff["Date"] = pd.to_datetime(ff["Date"], dayfirst=True)
    cf = [
        (row.Date.to_pydatetime(), -row.Deposit_Withdrawals)
        for row in ff.itertuples()
        if getattr(row, "Deposit_Withdrawals", 0) != 0
    ]
    # final portfolio value as inflow
    cf.append((nav.index[-1].to_pydatetime(), nav.iloc[-1]))
    xirr_val = _xirr(cf)

    return pd.Series({
        "StartBal":     nav.iloc[0],
        "EndBal":       nav.iloc[-1],
        "NetPnL":       net_pnl,
        "CAGR":         cagr_val,
        "XIRR":         xirr_val,
        "MaxDD":        max_dd,
        "MaxDD_pct":    max_dd * 100,
        "Sharpe":       _sharpe(daily_ret),
        "WinDays":      wins.count(),
        "LossDays":     losses.count(),
        "ProfitFactor": pf,
    })

# ╔═════════ STRATEGY‑LEVEL KPIs ══════════════════════════════╗
def strategy_kpis(
    trades:   pd.DataFrame | None = None,
    timeline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Load data
    tr = trades   if trades   is not None else load_trades()
    tl = timeline if timeline is not None else load_timeline()
    if tr.empty:
        return pd.DataFrame()

    # Map UID → Strategy (UID_base is already in the processed data as UID)
    strat_map = load_strategies()[["UID", "Strategy"]]
    tr = tr.merge(strat_map, on="UID", how="left")
    tr["Strategy"].fillna(tr["UID"], inplace=True)

    # Build StratLabel
    if "UnderlyingSymbol" in tr.columns:
        tr["StratLabel"] = tr["Strategy"] + "-" + tr["UnderlyingSymbol"].astype(str)
    else:
        tr["StratLabel"] = tr["Strategy"]

    # Aggregate basic stats
    g   = tr.groupby("StratLabel")
    kpi = pd.DataFrame(index=g.size().index)
    kpi["Trades"]   = g.size()
    kpi["Wins"]     = g.apply(lambda df: (df["NetCash"] > 0).sum())
    kpi["Losses"]   = g.apply(lambda df: (df["NetCash"] < 0).sum())
    kpi["WinRate"]  = kpi["Wins"] / kpi["Trades"]
    gross_win  = g.apply(lambda df: df.loc[df["NetCash"] > 0, "NetCash"].sum())
    gross_loss = g.apply(lambda df: -df.loc[df["NetCash"] < 0, "NetCash"].sum())
    kpi["ProfitFactor"] = gross_win / gross_loss.replace({0: np.nan})
    kpi["NetPnL"]  = g["NetCash"].sum()
    kpi["AvgSlip"] = g["slippage_abs"].mean()

    # Sharpe & XIRR per strategy
    if not tl.empty and "TradeDate" in tr.columns:
        tr["TradeDate"] = pd.to_datetime(tr["TradeDate"], dayfirst=True)
        daily_pnl = tr.groupby(["StratLabel", "TradeDate"])["NetCash"].sum()

        sharpe_vals, xirr_vals = [], []
        for label, pnl in daily_pnl.groupby(level=0):
            pnl = pnl.droplevel(0).sort_index()

            mean_abs = pnl.abs().mean()
            denom = mean_abs if mean_abs >= 1e-9 else 1e-9
            dr = pnl / denom

            sharpe_vals.append(_sharpe(dr))
            xirr_vals.append(_annualised(pnl.cumsum()))

        kpi["Sharpe"] = sharpe_vals
        kpi["XIRR"]   = xirr_vals

    kpi.fillna(0, inplace=True)
    kpi.sort_values("NetPnL", ascending=False, inplace=True)
    return kpi.reset_index(names=["Strategy"])

# ╔═════════ STREAMLIT HELPERS ═══════════════════════════════╗
def get_account_kpi_series() -> pd.Series:
    return account_kpis()

def get_strategy_kpi_table() -> pd.DataFrame:
    return strategy_kpis()

# ╔═════════ UID KPIs ════════════════════════════════════════╗
def uid_kpis(trades, margin_daily):
    # guarantee column exists – prevents crashes
    if "slippage_abs" not in trades.columns:
        trades["slippage_abs"] = 0.0

    pnl_daily = (trades.groupby(["UID", "TradeDate"])["NetCash"]
                        .sum()
                        .reset_index())

    slip_avg = (trades[trades["slippage_abs"] > 0]
                .groupby("UID")["slippage_abs"].mean())

    rows = []
    for uid, mg in margin_daily.groupby("UID"):
        marg = mg.set_index("Date")["MarginRequired"]
        pnl_s = pnl_daily[pnl_daily["UID"] == uid].set_index("TradeDate")["NetCash"]
        pnl_s = pnl_s.reindex(marg.index, fill_value=0)

        flows = [(d.to_pydatetime(), -v) for d, v in marg.items()]
        flows.append((marg.index[-1].to_pydatetime(), pnl_s.sum()))
        xirr = _xirr(flows)

        rows.append(dict(UID=uid, PnL=pnl_s.sum(), XIRR=xirr,
                         Expectancy=pnl_s.mean(),
                         AvgSlippage=slip_avg.get(uid, np.nan)))
    return pd.DataFrame(rows).round(4)

# ╔═════════ CLI TEST ════════════════════════════════════════╗
if __name__ == "__main__":
    pd.set_option("display.float_format", "{:,.2f}".format)
    print("=== Account KPIs ===")
    print(account_kpis())
    print("\n=== Strategy KPIs (top 5) ===")
    print(strategy_kpis().head())
