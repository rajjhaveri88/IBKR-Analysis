# scripts/etl.py   ✓ grouping‑by‑leg logic 2025‑07‑26
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

    # 1) Normalize & drop any exact duplicate columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # 2) Parse TradeDate (DD/MM/YY) and full DateTime
    df["TradeDate"] = pd.to_datetime(
        df["TradeDate"], format="%d/%m/%y", dayfirst=True
    ).dt.strftime("%d/%m/%y")
    df["DateTime"]  = pd.to_datetime(df["DateTime"], dayfirst=True)

    # 3) Load strategy & stop‑loss mappings
    strat_raw = load_raw_strategies()
    strat_map = strat_raw.set_index("UID")["Strategy_id"].to_dict()
    stop_map  = {}
    if "Stop_Loss_Pct" in strat_raw.columns:
        stop_map = (
            strat_raw
            .set_index("UID")["Stop_Loss_Pct"]
            .astype(float)
            .div(100.0)
            .to_dict()
        )

    # 4) Prepare for grouping
    grp_cols = [
        "TradeDate",
        "Description",
        "UnderlyingSymbol",
        "Strike",
        "Put/Call",
        "UID",
    ]
    df = df.sort_values(grp_cols + ["DateTime"])

    rows = []
    for key, g in df.groupby(grp_cols):
        # only keep complete legs whose quantities net to zero
        if g["Quantity"].sum() != 0:
            continue

        trade_date, desc, sym, strike, pc, uid = key

        # entry trades are the negative‑quantity rows
        entries = g[g.Quantity < 0]
        # exit trades are the positive‑quantity rows
        exits   = g[g.Quantity > 0]

        # times
        dt_entry = entries["DateTime"].min()
        dt_exit  = exits["DateTime"].max()

        # weighted average entry price
        eq = entries["Quantity"].abs()
        if eq.sum() > 0:
            px_entry = (entries["TradePrice"] * eq).sum() / eq.sum()
        else:
            px_entry = entries["TradePrice"].iloc[0]

        # weighted average exit price
        exit_qty = exits["Quantity"]
        px_exit  = (exits["TradePrice"] * exit_qty).sum() / exit_qty.sum()

        # total contracts closed
        qty = int(exit_qty.sum())

        # sums of P&L and fees
        proceeds    = g["Proceeds"].sum()
        ib_comm     = g["IBCommission"].sum()
        net_cash    = g["NetCash"].sum()

        # slippage vs stop‑loss
        sl_pct       = stop_map.get(uid, 0.0)
        exp_price    = px_entry * (1 + sl_pct)
        raw_slip     = max(px_exit - exp_price, 0.0)
        slip_amt     = raw_slip * entries["Multiplier"].iloc[0]
        dist_exp     = exp_price - px_entry
        slip_pct     = (raw_slip / dist_exp * 100) if dist_exp else 0.0

        rows.append({
            "TradeDate":        trade_date,
            "Description":      desc,
            "UnderlyingSymbol": sym,
            "Strike":           strike,
            "Put/Call":         pc,
            "DateTime_entry":   dt_entry,
            "DateTime_exit":    dt_exit,
            "Quantity":         qty,
            "TradePrice":       px_entry,
            "ClosePrice":       px_exit,
            "Multiplier":       entries["Multiplier"].iloc[0],
            "Proceeds":         proceeds,
            "IBCommission":     ib_comm,
            "NetCash":          net_cash,
            "OrderType_entry":  entries["OrderType"].iloc[0],
            "OrderType_exit":   exits["OrderType"].iloc[-1],
            "UID":              uid,
            "Strategy":         strat_map.get(uid, uid),
            "Legs":             "1/1",
            "Slippage":         slip_amt,
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
        piv.columns = ["_".join(c) for c in piv.columns]
        tl = tl.merge(piv.reset_index(), on="Date", how="left")

        margin_df = (
            alloc.groupby(["Date","Strategy"])["Margin"]
                 .sum()
                 .reset_index()
                 .rename(columns={"Strategy":"UID"})
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

    rows, units_prev, nav_prev = [], bb.iloc[0]/100.0, 100.0
    for dt in dr:
        bal   = float(bb.loc[dt])
        flow  = float(flows_full.loc[dt])
        units = units_prev + flow/nav_prev
        nav   = bal/units if units else np.nan
        rows.append({
            "Date":          dt,
            "NAV":           nav,
            "Units":         units,
            "BrokerBalance": bal,
        })
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
