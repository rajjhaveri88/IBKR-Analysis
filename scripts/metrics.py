from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd

from scripts.loaders import load_nav, load_fund_flow, load_trades, load_uid_margin

def _years_between(d1: datetime, d2: datetime) -> float:
    return (d2 - d1).days / 365.25

def _xirr(cashflows: list[tuple[datetime, float]], guess: float = 0.1) -> float:
    """Newton–Raphson IRR."""
    if len(cashflows) < 2:
        return np.nan
    t0 = cashflows[0][0]
    def npv(r): return sum(cf / (1+r)**_years_between(t0, dt) for dt, cf in cashflows)
    rate = guess
    for _ in range(100):
        f  = npv(rate)
        df = sum(
            -_years_between(t0, dt) * cf / (1+rate)**(_years_between(t0,dt)+1)
            for dt,cf in cashflows
        )
        if df == 0:
            break
        new = rate - f/df
        if abs(new-rate) < 1e-7:
            return new
        rate = new
    return np.nan

def _annualised(series: pd.Series) -> float:
    if series.empty or series.iloc[0] == 0:
        return np.nan
    days = (series.index[-1]-series.index[0]).days
    if days < 30:
        return np.nan
    return (series.iloc[-1]/series.iloc[0])**(365.25/days) - 1

def _sharpe(daily_ret: pd.Series, rf: float=0.0) -> float:
    sd = daily_ret.std(ddof=0)
    return ((daily_ret.mean()-rf)/sd)*np.sqrt(252) if sd>0 else np.nan

# ╔═ ACCOUNT‑LEVEL KPIs ════════════════════════════════════
def account_kpis(
    fund_flow: pd.DataFrame|None=None,
    timeline:  pd.DataFrame|None=None,
) -> pd.Series:
    ff     = fund_flow if fund_flow is not None else load_fund_flow()
    nav_df = load_nav()
    if nav_df.empty or ff.empty:
        return pd.Series(dtype=float)

    nav_df["Date"] = pd.to_datetime(nav_df["Date"], dayfirst=True)
    nav = nav_df.set_index("Date")["NAV"].sort_index()

    daily_ret = nav.pct_change().dropna()
    net_pnl   = nav.iloc[-1] - nav.iloc[0]
    cagr_val  = _annualised(nav)

    cummax = nav.cummax()
    max_dd = ((nav - cummax)/cummax).min()

    wins   = daily_ret[daily_ret>0]
    losses = daily_ret[daily_ret<0].abs()
    pf     = wins.sum()/losses.sum() if not losses.empty else np.nan

    ff["Date"] = pd.to_datetime(ff["Date"], dayfirst=True)
    cf = [
        (row.Date.to_pydatetime(), -row.Deposit_Withdrawals)
        for row in ff.itertuples()
        if getattr(row,"Deposit_Withdrawals",0) != 0
    ]
    cf.append((nav.index[-1].to_pydatetime(), nav.iloc[-1]))
    xirr_val = _xirr(cf)

    return pd.Series({
        "StartBal":     nav.iloc[0],
        "EndBal":       nav.iloc[-1],
        "NetPnL":       net_pnl,
        "CAGR":         cagr_val,
        "XIRR":         xirr_val,
        "MaxDD":        max_dd,
        "MaxDD_pct":    max_dd*100,
        "Sharpe":       _sharpe(daily_ret),
        "WinDays":      wins.count(),
        "LossDays":     losses.count(),
        "ProfitFactor": pf,
    })

# ╔═ STRATEGY‑LEVEL KPIs ═══════════════════════════════════
def strategy_kpis(
    trades:   pd.DataFrame|None=None,
    timeline: pd.DataFrame|None=None,
) -> pd.DataFrame:
    tr = trades if trades is not None else load_trades()
    if tr.empty:
        return pd.DataFrame()

    # Check if Strategy column exists, if not create it from UID
    if "Strategy" not in tr.columns:
        tr["Strategy"] = tr["UID"]
    else:
        tr["Strategy"] = tr["Strategy"].fillna(tr["UID"])
    if "UnderlyingSymbol" in tr.columns:
        tr["StratLabel"] = tr["Strategy"] + "-" + tr["UnderlyingSymbol"]
    else:
        tr["StratLabel"] = tr["Strategy"]

    g   = tr.groupby("StratLabel")
    kpi = pd.DataFrame(index=g.size().index)
    kpi["Trades"]       = g.size()
    kpi["Wins"]         = g.apply(lambda df: (df.NetCash>0).sum())
    kpi["Losses"]       = g.apply(lambda df: (df.NetCash<0).sum())
    kpi["WinRate"]      = kpi["Wins"]/kpi["Trades"]
    gross_win  = g.apply(lambda df: df.loc[df.NetCash>0,"NetCash"].sum())
    gross_loss = g.apply(lambda df: -df.loc[df.NetCash<0,"NetCash"].sum())
    kpi["ProfitFactor"] = gross_win/gross_loss.replace({0:np.nan})
    kpi["NetPnL"]       = g.NetCash.sum()
    kpi["AvgSlip"]      = g["Slippage"].mean()

    # optional Sharpe & XIRR per strategy
    tr["TradeDate"] = pd.to_datetime(tr["TradeDate"], dayfirst=True)
    daily_pnl = tr.groupby(["StratLabel","TradeDate"])["NetCash"].sum()
    sharpe_vals, xirr_vals = [], []
    for label, pnl in daily_pnl.groupby(level=0):
        p   = pnl.droplevel(0).sort_index()
        mean_abs = p.abs().mean()
        if mean_abs > 0:
            dr = p / mean_abs
        else:
            dr = p
        sharpe_vals.append(_sharpe(dr))
        xirr_vals.append(_annualised(p.cumsum()))
    kpi["Sharpe"] = sharpe_vals
    kpi["XIRR"]   = xirr_vals

    return kpi.reset_index(names=["Strategy"]).fillna(0)

# ╔═ UID‑LEVEL KPIs ══════════════════════════════════════════
def uid_kpis(
    margin_daily: pd.DataFrame|None=None
) -> pd.DataFrame:
    mg = margin_daily if margin_daily is not None else load_uid_margin()
    if mg.empty:
        return pd.DataFrame()
    # mg has columns Date, UID, Margin
    df = mg.groupby("UID")["Margin"].agg(
        AvgMargin=lambda s: s.mean(),
        MaxMargin=lambda s: s.max()
    ).reset_index()
    return df

# ── helpers ───────────────────────────────────────────────────
def get_account_kpi_series(): return account_kpis()
def get_strategy_kpi_table(): return strategy_kpis()
def get_uid_kpi_table(margin_df=None):      return uid_kpis(margin_df)
