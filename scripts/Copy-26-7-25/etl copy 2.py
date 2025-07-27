# scripts/etl.py   ✓ revised for grouping logic 2025‑07‑26
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

def build_legs() -> None:
    df = load_raw_trades()
    if df.empty:
        print("⚠️  Trades.csv missing or empty.")
        return

    # 1) normalize & drop duplicate cols
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # 2) parse dates
    df["TradeDate"] = pd.to_datetime(
        df["TradeDate"], format="%d/%m/%y", dayfirst=True
    ).dt.strftime("%d/%m/%y")
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)

    # 3) explode multi‑UID rows
    df["UID_list"] = df["UID"].astype(str).str.split(",").apply(lambda L: [u.strip() for u in L])
    df = df.explode("UID_list")
    df = df.drop(columns=["UID"]).rename(columns={"UID_list": "UID"})
    df["UID"] = df["UID"].astype(str)
    df["UID_base"] = df["UID"].str.replace(r"-H$", "", regex=True)

    # 4) load mappings
    strat_raw = load_raw_strategies()
    strat_map = strat_raw.set_index("UID")["Strategy_id"].to_dict()
    stop_map  = {}
    if "Stop_Loss_Pct" in strat_raw.columns:
        stop_map = (
            strat_raw.set_index("UID")["Stop_Loss_Pct"].astype(float).div(100.0).to_dict()
        )

    rows = []
    grp_cols = ["UID", "TradeDate", "Description", "UnderlyingSymbol", "Strike", "Put/Call"]
    df = df.sort_values(grp_cols + ["DateTime"])

    for key, g in df.groupby(grp_cols):
        if g["Quantity"].sum() != 0:
            continue
        entry = g.iloc[0]
        exit_ = g.iloc[-1]

        entry_px = entry["TradePrice"]
        exit_px  = exit_["TradePrice"]
        qty       = g[g.Quantity > 0]["Quantity"].sum() or abs(g[g.Quantity < 0]["Quantity"].sum())
        proceeds  = g["Proceeds"].sum()
        comm      = g["IBCommission"].sum()
        netcash   = g["NetCash"].sum()

        sl_pct    = stop_map.get(entry["UID"], 0.0)
        expected  = entry_px * (1 + sl_pct)
        raw_slip  = max(exit_px - expected, 0.0)
        slippage  = raw_slip * entry["Multiplier"]
        dist_exp  = expected - entry_px
        slip_pct  = (raw_slip / dist_exp * 100) if dist_exp else 0.0

        rows.append({
            "TradeDate":        entry["TradeDate"],
            "Description":      entry["Description"],
            "UnderlyingSymbol": entry["UnderlyingSymbol"],
            "Strike":           entry["Strike"],
            "Put/Call":         entry["Put/Call"],
            "DateTime_entry":   entry["DateTime"],
            "DateTime_exit":    exit_["DateTime"],
            "Quantity":         int(qty),
            "TradePrice":       entry_px,
            "ClosePrice":       exit_px,
            "Multiplier":       entry["Multiplier"],
            "Proceeds":         proceeds,
            "IBCommission":     comm,
            "NetCash":          netcash,
            "OrderType_entry":  entry["OrderType"],
            "OrderType_exit":   exit_["OrderType"],
            "UID":              entry["UID"],
            "Strategy":         strat_map.get(entry["UID"], entry["UID"]),
            "Legs":             "1/1",
            "Slippage":         slippage,
            "Slippage_pct":     slip_pct,
        })

    pd.DataFrame(rows).to_parquet(_p("legs"), index=False)
    print("✅ legs.parquet written")


def build_timeline_and_margin() -> None:
    alloc = load_raw_allocation()
    fund  = load_raw_fund_flow()
    if fund.empty:
        print("⚠️  Fund Flow CSV missing or empty.")
        return

    tl = fund[["Date", "EndingCash"]].rename(columns={"EndingCash": "BrokerBalance"})
    if not alloc.empty:
        piv = (
            alloc.pivot_table(
                index="Date",
                columns=["Strategy", "UnderlyingSymbol"],
                values="Allocation",
                aggfunc="sum",
            )
            .add_prefix("Allocation_")
        )
        piv.columns = ["_".join(c) for c in piv.columns]
        tl = tl.merge(piv.reset_index(), on="Date", how="left")

        margin_df = (
            alloc.groupby(["Date", "Strategy"])["Margin"]
            .sum().reset_index()
            .rename(columns={"Strategy": "UID"})
        )
        margin_df.to_parquet(_p("uid_margin"), index=False)

    tl.to_parquet(_p("timeline"), index=False)
    print("✅ timeline.parquet written")


def build_strategies() -> None:
    df = load_raw_strategies()
    df.to_parquet(_p("strategies"), index=False)
    print("✅ strategies.parquet written")


def build_nav_series() -> None:
    tl = pd.read_parquet(_p("timeline")) if _p("timeline").exists() else pd.DataFrame()
    ff = load_raw_fund_flow()
    if tl.empty or ff.empty:
        print("⚠️  Cannot build NAV: missing timeline or fund_flow.")
        return

    tl["Date"] = pd.to_datetime(tl["Date"], dayfirst=True)
    ff["Date"] = pd.to_datetime(ff["Date"], dayfirst=True)
    flows      = ff.set_index("Date")["Deposit/Withdrawals"]

    dr         = pd.date_range(tl["Date"].min(), tl["Date"].max(), freq="B")
    bb         = tl.set_index("Date")["BrokerBalance"].reindex(dr).ffill()
    flows_full = flows.reindex(dr).fillna(0.0)

    rows, units_prev, nav_prev = [], bb.iloc[0] / 100.0, 100.0
    for dt in dr:
        bal   = float(bb.loc[dt])
        flow  = float(flows_full.loc[dt])
        units = units_prev + flow / nav_prev
        nav   = bal / units if units else np.nan
        rows.append(
            {"Date": dt, "NAV": nav, "Units": units, "BrokerBalance": bal}
        )
        units_prev, nav_prev = units, nav

    pd.DataFrame(rows).to_parquet(_p("nav"), index=False)
    print("✅ nav.parquet written")


def main() -> None:
    Path(_p("").parent).mkdir(parents=True, exist_ok=True)
    build_legs()
    build_timeline_and_margin()
    build_strategies()
    build_nav_series()
    print("\n🎉 ETL complete.")

if __name__ == "__main__":
    main()
