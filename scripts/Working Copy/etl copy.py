# scripts/etl.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from loaders import (
    load_fund_flow,
    load_allocation,
    load_trades,
    load_strategies,
)
from cash_engine import build_cash_timeline

# ── output folder ──────────────────────────────────────────────
RAW_DIR  = Path("Data")
PROC_DIR = RAW_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ╭──────────────────────────────────────────────────────────╮
# │  Slippage helper                                         │
# ╰──────────────────────────────────────────────────────────╯
def _compute_slippage(trades: pd.DataFrame,
                      strat_info: pd.DataFrame) -> pd.Series:
    """
    slippage_abs = max(0, ExitPx – (EntryPx × (1 + Stop_Loss_Pct)))
    Only for losing legs; profitable or un‑touched legs = 0.
    """
    t = trades.copy()

    # base UID (strip "-H") for merge key
    t["UID_base"] = t["UID"].str.replace(r"-H$", "", regex=True)

    # bring Stop_Loss_Pct onto every trade row
    info = (strat_info[["UID", "Stop_Loss_Pct"]]
            .rename(columns={"UID": "UID_base"}))
    t = t.merge(info, on="UID_base", how="left")
    t["Stop_Loss_Pct"].fillna(0.0, inplace=True)

    stop_px = t["EntryPx"] * (1 + t["Stop_Loss_Pct"])
    slip = np.where(
        (t["ExitPx"] > stop_px) & (t["NetCash"] < 0),   # losing + beyond stop
        t["ExitPx"] - stop_px,
        0.0
    )
    return pd.Series(slip, index=t.index, name="slippage_abs")


# ╭──────────────────────────────────────────────────────────╮
# │  Build UID‑margin daily table                            │
# ╰──────────────────────────────────────────────────────────╯
def _build_uid_margin(allocation: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame [Date, UID, MarginRequired] forward‑filled daily.
    UID column is StrategyUID, hedged legs share the same UID.
    """
    alloc = allocation.copy()
    alloc["UID"] = alloc["UID"].str.replace(r"-H$", "", regex=True)
    margin = (alloc.groupby(["Date", "UID"])["Margin"]
              .sum()
              .reset_index())

    # build daily range per UID & ffill
    all_days = pd.date_range(alloc["Date"].min(), alloc["Date"].max(), freq="D")
    uid_margin = []
    for uid, grp in margin.groupby("UID"):
        m = grp.set_index("Date").reindex(all_days).sort_index()["Margin"]
        m = m.ffill().fillna(0)
        uid_margin.append(
            pd.DataFrame({"Date": all_days, "UID": uid, "MarginRequired": m})
        )
    return pd.concat(uid_margin, ignore_index=True)


# ╭──────────────────────────────────────────────────────────╮
# │  ETL pipeline                                            │
# ╰──────────────────────────────────────────────────────────╯
def run_etl() -> None:
    # 1) raw loads
    fund        = load_fund_flow()
    allocation  = load_allocation()
    trades      = load_trades()
    strategies  = load_strategies()

    # 2) cash timeline parquet
    timeline = build_cash_timeline(fund, allocation)
    timeline.to_parquet(PROC_DIR / "timeline.parquet", index=False)

    # 3) slippage & save legs.parquet
    trades["slippage_abs"] = _compute_slippage(trades, strategies)
    trades.to_parquet(PROC_DIR / "legs.parquet", index=False)

    # 4) realised PnL per UID × day
    runs = (trades.groupby(["UID", "TradeDate"])["NetCash"]
                   .sum()
                   .reset_index())
    runs.to_parquet(PROC_DIR / "runs.parquet", index=False)

    # 5) UID‑margin daily parquet
    uid_margin = _build_uid_margin(allocation)
    uid_margin.to_parquet(PROC_DIR / "uid_margin.parquet", index=False)

    # 6) dump strategies for joins
    strategies.to_parquet(PROC_DIR / "strategies.parquet", index=False)

    # completion log
    print("✅ ETL complete:")
    for f in ("timeline", "legs", "runs", "uid_margin", "strategies"):
        p = PROC_DIR / f"{f}.parquet"
        print(f"   • {p.name:15} {p.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    run_etl()
