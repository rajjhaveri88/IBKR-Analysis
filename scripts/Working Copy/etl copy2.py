# scripts/etl.py   ✓ 2025‑07‑25
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.loaders import (
    _p,
    load_raw_trades,
    load_raw_allocation,
    load_raw_strategies,
    load_raw_fund_flow,
)

# 1) legs.parquet ─────────────────────────────────────────────
def build_legs():
    t = load_raw_trades()
    if t.empty:
        print("⚠️  Trades.csv missing or empty.")
        return
    t["DateTime"]  = pd.to_datetime(t["DateTime"])
    t["TradeDate"] = pd.to_datetime(t["TradeDate"], dayfirst=True).dt.date

    legs = (
        t.sort_values(["UID", "TradeDate", "DateTime"])
         .groupby(["UID", "TradeDate"])
         .agg(
            EntryPx=("TradePrice","first"),
            ExitPx=("TradePrice","last"),
            NetCash=("NetCash","sum"),
            slippage_abs=("TradePrice", lambda s: abs(s.iloc[0] - s.iloc[-1])),
         )
         .reset_index()
    )
    legs["UID_base"] = legs["UID"].str.replace(r"-H$", "", regex=True)
    legs.to_parquet(_p("legs"), index=False)
    print("✅ legs.parquet written")

# 2) timeline.parquet & uid_margin.parquet ───────────────────
def build_timeline_and_margin():
    alloc = load_raw_allocation()
    fund  = load_raw_fund_flow()
    if fund.empty:
        print("⚠️  Fund Flow CSV missing or empty.")
        return

    tl = fund[["Date", "EndingCash"]].rename(columns={"EndingCash":"BrokerBalance"})

    if not alloc.empty:
        piv = (
            alloc.pivot_table(
                index="Date",
                columns=["Strategy","UnderlyingSymbol"],
                values="Allocation",
                aggfunc="sum",
            )
            .add_prefix("Allocation_")
        )
        piv.columns = ["_".join(col) for col in piv.columns]
        tl = tl.merge(piv.reset_index(), on="Date", how="left")

        if {"Date","UID","Margin"}.issubset(alloc.columns):
            alloc.groupby(["Date","UID"])["Margin"].sum().reset_index() \
                 .to_parquet(_p("uid_margin"), index=False)

    tl.to_parquet(_p("timeline"), index=False)
    print("✅ timeline.parquet written")

# 3) strategies.parquet ─────────────────────────────────────
def build_strategies():
    df = load_raw_strategies()
    df.to_parquet(_p("strategies"), index=False)
    print("✅ strategies.parquet written")

# 4) nav.parquet (daily NAV curve) ───────────────────────────
def build_nav_series():
    tl = pd.read_parquet(_p("timeline")) if _p("timeline").exists() else pd.DataFrame()
    ff = load_raw_fund_flow()
    if tl.empty or ff.empty:
        print("⚠️  Cannot build NAV: missing timeline or fund_flow.")
        return

    # business‑day index
    dr    = pd.date_range(tl["Date"].min(), tl["Date"].max(), freq="B")
    bb    = tl.set_index("Date")["BrokerBalance"].reindex(dr).ffill()
    flows = ff.set_index("Date")["Deposit/Withdrawals"].reindex(dr).fillna(0.0)

    rows = []
    units_prev = bb.iloc[0] / 100.0
    nav_prev   = 100.0

    for dt in dr:
        bal  = float(bb.loc[dt])
        flow = float(flows.loc[dt])
        units = units_prev + flow / nav_prev
        nav   = bal / units if units else np.nan
        rows.append({"Date": dt, "NAV": nav, "Units": units, "BrokerBalance": bal})
        units_prev, nav_prev = units, nav

    pd.DataFrame(rows).to_parquet(_p("nav"), index=False)
    print("✅ nav.parquet written")

# ── orchestrator ─────────────────────────────────────────────
def main():
    Path(_p("").parent).mkdir(parents=True, exist_ok=True)
    build_legs()
    build_timeline_and_margin()
    build_strategies()
    build_nav_series()
    print("\n🎉  ETL complete.")

if __name__ == "__main__":
    main()
