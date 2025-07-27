# scripts/etl.py   ✓ revised 2025‑07‑26
# ──────────────────────────────────────────────────────────────
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

# ── 1) legs.parquet ───────────────────────────────────────────
def build_legs() -> None:
    df = load_raw_trades()
    if df.empty:
        print("⚠️  Trades.csv missing or empty.")
        return

    # ─ normalize & drop any duplicate columns ─────────────────
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # ─ parse dates ─────────────────────────────────────────────
    df["TradeDate_dt"] = pd.to_datetime(
        df["TradeDate"], format="%d/%m/%y", dayfirst=True
    )
    df["TradeDate"] = df["TradeDate_dt"].dt.strftime("%d/%m/%y")
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)

    # ─ split multi‑UID rows ────────────────────────────────────
    df["UID_list"] = (
        df["UID"]
        .str.split(",")
        .apply(lambda L: [u.strip() for u in L])
    )
    df["N_UID"] = df["UID_list"].apply(len)
    df = df.explode("UID_list")
    df["UID"] = df["UID_list"]  # overwrite original
    for col in ("Quantity", "Proceeds", "IBCommission", "NetCash"):
        df[col] = df[col].astype(float) / df["N_UID"]
    df["Quantity"] = df["Quantity"].round().astype(int)
    df = df.drop(columns=["UID_list", "N_UID", "TradeDate_dt"])
    df["UID_base"] = df["UID"].str.replace(r"-H$", "", regex=True)

    # ─ load Strategies.csv for Strategy_id & Stop_Loss_Pct ────
    strat_raw = load_raw_strategies()
    strat_map = strat_raw.set_index("UID")["Strategy_id"].to_dict()
    if "Stop_Loss_Pct" in strat_raw.columns:
        stop_map = (
            strat_raw.set_index("UID")["Stop_Loss_Pct"]
            .astype(float) / 100.0
        ).to_dict()
    else:
        stop_map = {}

    # ─ build one row per round‑trip leg ────────────────────────
    rows = []
    grp_cols = ["UID_base", "TradeDate", "UnderlyingSymbol", "Strike", "Put/Call"]
    df = df.sort_values(grp_cols + ["DateTime"])

    for key, g in df.groupby(grp_cols):
        uid_base, td, sym, strike, pc = key
        n = len(g)
        if n < 2 or n % 2 != 0:
            continue  # skip incomplete
        total_legs = n // 2

        for i in range(total_legs):
            part = g.iloc[2 * i : 2 * i + 2]
            entry = part.iloc[0]
            exit_ = part.iloc[1]

            entry_px = entry["TradePrice"]
            exit_px = exit_["TradePrice"]

            # closed quantity
            qty = int(part[part.Quantity > 0]["Quantity"].sum())
            if qty == 0:
                qty = abs(int(part[part.Quantity < 0]["Quantity"].sum()))

            # slippage vs Stop_Loss
            sl_pct = stop_map.get(entry["UID"], 0.0)
            expected_exit = entry_px * (1 + sl_pct)
            raw_slip = max(exit_px - expected_exit, 0.0)
            slippage = raw_slip * entry["Multiplier"]
            dist_expected = expected_exit - entry_px
            slippage_pct = (raw_slip / dist_expected * 100) if dist_expected else 0.0

            rows.append({
                "TradeDate":        entry["TradeDate"],      # "DD/MM/YY"
                "Description":      entry["Description"],
                "UnderlyingSymbol": sym,
                "DateTime_entry":   entry["DateTime"],
                "DateTime_exit":    exit_["DateTime"],
                "Put/Call":         pc,
                "Quantity":         qty,
                "TradePrice":       entry_px,
                "ClosePrice":       exit_px,
                "Multiplier":       entry["Multiplier"],
                "Proceeds":         part["Proceeds"].sum(),
                "IBCommission":     part["IBCommission"].sum(),
                "NetCash":          part["NetCash"].sum(),
                "OrderType_entry":  entry["OrderType"],
                "OrderType_exit":   exit_["OrderType"],
                "UID":              entry["UID"],
                "Strategy":         strat_map.get(entry["UID"], uid_base),
                "Legs":             f"{i+1}/{total_legs}",
                "Slippage":         slippage,
                "Slippage_pct":     slippage_pct,
            })

    pd.DataFrame(rows).to_parquet(_p("legs"), index=False)
    print("✅ legs.parquet written")


# ── 2) timeline.parquet & uid_margin.parquet ─────────────────
def build_timeline_and_margin() -> None:
    alloc = load_raw_allocation()
    fund  = load_raw_fund_flow()
    if fund.empty:
        print("⚠️  Fund Flow CSV missing or empty.")
        return

    # build daily timeline
    tl = fund[["Date", "EndingCash"]].rename(
        columns={"EndingCash": "BrokerBalance"}
    )

    if not alloc.empty:
        # pivot allocations
        piv = (
            alloc
            .pivot_table(
                index="Date",
                columns=["Strategy", "UnderlyingSymbol"],
                values="Allocation",
                aggfunc="sum",
            )
            .add_prefix("Allocation_")
        )
        piv.columns = ["_".join(c) for c in piv.columns]
        tl = tl.merge(piv.reset_index(), on="Date", how="left")

        # now build UID‐margin from Strategy
        margin_df = (
            alloc.groupby(["Date", "Strategy"])["Margin"]
            .sum()
            .reset_index()
            .rename(columns={"Strategy": "UID"})
        )
        margin_df.to_parquet(_p("uid_margin"), index=False)

    tl.to_parquet(_p("timeline"), index=False)
    print("✅ timeline.parquet written")


# ── 3) strategies.parquet ────────────────────────────────────
def build_strategies() -> None:
    df = load_raw_strategies()
    df.to_parquet(_p("strategies"), index=False)
    print("✅ strategies.parquet written")


# ── 4) nav.parquet (daily NAV) ───────────────────────────────
def build_nav_series() -> None:
    tl = pd.read_parquet(_p("timeline")) if _p("timeline").exists() else pd.DataFrame()
    ff = load_raw_fund_flow()
    if tl.empty or ff.empty:
        print("⚠️  Cannot build NAV: missing timeline or fund_flow.")
        return

    tl["Date"] = pd.to_datetime(tl["Date"], dayfirst=True)
    ff["Date"] = pd.to_datetime(ff["Date"], dayfirst=True)
    flows = ff.set_index("Date")["Deposit/Withdrawals"]

    dr         = pd.date_range(tl["Date"].min(), tl["Date"].max(), freq="B")
    bb         = tl.set_index("Date")["BrokerBalance"].reindex(dr).ffill()
    flows_full = flows.reindex(dr).fillna(0.0)

    rows, units_prev, nav_prev = [], bb.iloc[0] / 100.0, 100.0
    for dt in dr:
        bal   = float(bb.loc[dt])
        flow  = float(flows_full.loc[dt])
        units = units_prev + flow / nav_prev
        nav   = bal / units if units else np.nan
        rows.append({
            "Date":          dt,
            "NAV":           nav,
            "Units":         units,
            "BrokerBalance": bal,
        })
        units_prev, nav_prev = units, nav

    pd.DataFrame(rows).to_parquet(_p("nav"), index=False)
    print("✅ nav.parquet written")


# ── Orchestrator ─────────────────────────────────────────────
def main() -> None:
    Path(_p("").parent).mkdir(parents=True, exist_ok=True)
    build_legs()
    build_timeline_and_margin()
    build_strategies()
    build_nav_series()
    print("\n🎉 ETL complete.")

if __name__ == "__main__":
    main()
