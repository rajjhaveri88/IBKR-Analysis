# scripts/metrics.py
# ──────────────────────────────────────────────────────────────
"""
KPIs for Overview (account), Strategy, and UID levels.
Slippage column is auto‑added (zeros) if missing – no crashes.
"""
from __future__ import annotations
import pandas as pd, numpy as np
from scipy.optimize import newton

TRADING_DAYS = 252  # trading days / year


# ╭─ IRR helpers ───────────────────────────────────────────────╮
def _xnpv(r, vals, dates):
    d0 = dates[0]
    return sum(v / (1 + r) ** ((d - d0).days / 365.0)
               for v, d in zip(vals, dates))

def _xirr(flows):
    if len(flows) < 2:
        return None
    dates = pd.to_datetime([f["date"] for f in flows])
    vals  = [f["value"] for f in flows]
    try:
        return float(newton(lambda r: _xnpv(r, vals, dates), 0.1))
    except Exception:
        return None

def _sharpe_daily(ret):
    if ret.empty or ret.std(ddof=0) == 0:
        return None
    return np.sqrt(TRADING_DAYS) * ret.mean() / ret.std(ddof=0)


# ╭─ Account KPIs ──────────────────────────────────────────────╮
def account_kpis(fund, tl):
    flows = [{"date": d, "value": v}
             for d, v in (fund
                          .assign(Net=lambda d: d["Deposit/Withdrawals"] + d["BrokerInterest"])
                          [["Date", "Net"]].values)]
    flows.append({"date": tl["Date"].iloc[-1], "value": tl["BrokerBalance"].iloc[-1]})
    xirr = _xirr(flows)

    broker = tl.set_index("Date")["BrokerBalance"]
    ret = broker.pct_change().dropna()
    sharpe  = _sharpe_daily(ret)
    downside = ret[ret < 0]
    sortino  = (_sharpe_daily(ret) * ret.std(ddof=0) / downside.std(ddof=0)
                if not downside.empty and downside.std(ddof=0) else None)
    max_dd = (broker / broker.cummax() - 1).min() * 100

    return dict(XIRR=xirr, Sharpe=sharpe, Sortino=sortino, MaxDD_pct=max_dd)


# ╭─ Strategy KPIs (capital = Allocation) ──────────────────────╮
def strategy_kpis(trades, tl):
    trades = trades.copy()
    trades["UID_base"] = trades["UID"].str.replace(r"-H$", "", regex=True)

    from scripts.loaders import load_strategies
    strat_map = (load_strategies()[["UID", "Strategy"]]
                 .rename(columns={"UID": "UID_base"}))
    trades = trades.merge(strat_map, on="UID_base", how="left").drop(columns="UID_base")
    trades = trades.dropna(subset=["Strategy"])

    pnl_daily = (trades.groupby(["Strategy", "TradeDate"])["NetCash"]
                        .sum()
                        .reset_index())

    alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
    alloc_long = (tl.melt(id_vars="Date", value_vars=alloc_cols,
                          var_name="col", value_name="Allocation")
                    .assign(Strategy=lambda d: d["col"].str.split("_").str[1])
                    .drop(columns="col"))

    rows = []
    for sid, grp in pnl_daily.groupby("Strategy"):
        alloc_s = (alloc_long[alloc_long["Strategy"] == sid]
                   .set_index("Date")["Allocation"])
        if alloc_s.empty:
            continue
        pnl_s = grp.set_index("TradeDate")["NetCash"].reindex(alloc_s.index, fill_value=0)

        flows = [{"date": d, "value": -a} for d, a in alloc_s.items()]
        flows.append({"date": alloc_s.index[-1], "value": pnl_s.sum()})
        xirr = _xirr(flows)

        cap = alloc_s.replace(0, np.nan).ffill().bfill()
        sharpe = _sharpe_daily((pnl_s / cap.replace(0, np.nan)).dropna())

        equity = pnl_s.cumsum()
        max_dd = abs((equity - equity.cummax()).min())
        days = (equity.index[-1] - equity.index[0]).days
        cagr = ((equity.iloc[-1] / cap.iloc[0]) ** (365 / days) - 1
                if days and cap.iloc[0] else None)
        calmar = cagr / (max_dd / cap.iloc[0]) if (cagr and max_dd) else None

        wins = grp["NetCash"].gt(0).sum()
        losses = grp["NetCash"].lt(0).sum()
        pf = (grp[grp["NetCash"] > 0]["NetCash"].sum() /
              abs(grp[grp["NetCash"] < 0]["NetCash"].sum())
              ) if losses else None
        win_r = wins / (wins + losses) if (wins + losses) else None

        rows.append(dict(Strategy=sid, PnL=pnl_s.sum(), XIRR=xirr,
                         Sharpe=sharpe, Calmar=calmar,
                         WinRate=win_r, ProfitFactor=pf))
    return pd.DataFrame(rows).round(4)


# ╭─ UID KPIs (capital = MarginRequired) ───────────────────────╮
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

        flows = [{"date": d, "value": -v} for d, v in marg.items()]
        flows.append({"date": marg.index[-1], "value": pnl_s.sum()})
        xirr = _xirr(flows)

        rows.append(dict(UID=uid, PnL=pnl_s.sum(), XIRR=xirr,
                         Expectancy=pnl_s.mean(),
                         AvgSlippage=slip_avg.get(uid, np.nan)))
    return pd.DataFrame(rows).round(4)
