# scripts/etl.py   ✓ hedged‑UID as base + IsHedge flag 2025‑07‑26
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

    # 1) Cleanup
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # 2) Dates
    df["TradeDate"] = (
        pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
          .dt.strftime("%d/%m/%y")
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)

    # 2.5) Handle multi-UID rows by splitting them into separate rows
    # Find rows where UID contains commas (multiple UIDs)
    multi_uid_mask = df["UID"].str.contains(",", na=False)
    if multi_uid_mask.any():
        # Split multi-UID rows
        multi_uid_rows = df[multi_uid_mask].copy()
        single_uid_rows = df[~multi_uid_mask].copy()
        
        # Split each multi-UID row into multiple rows
        split_rows = []
        for _, row in multi_uid_rows.iterrows():
            uids = [uid.strip() for uid in row["UID"].split(",")]
            # Distribute quantity equally among UIDs
            qty_per_uid = row["Quantity"] / len(uids)
            proceeds_per_uid = row["Proceeds"] / len(uids)
            ib_comm_per_uid = row["IBCommission"] / len(uids)
            net_cash_per_uid = row["NetCash"] / len(uids)
            
            for uid in uids:
                new_row = row.copy()
                new_row["UID"] = uid
                new_row["Quantity"] = qty_per_uid
                new_row["Proceeds"] = proceeds_per_uid
                new_row["IBCommission"] = ib_comm_per_uid
                new_row["NetCash"] = net_cash_per_uid
                split_rows.append(new_row)
        
        # Combine back
        df = pd.concat([single_uid_rows, pd.DataFrame(split_rows)], ignore_index=True)

    # 3) Compute UID_base by stripping "-H"
    df["UID_base"] = df["UID"].str.replace(r"-H$", "", regex=True)
    # And flag which rows were hedges
    df["IsHedge"]  = df["UID"].str.endswith("-H")

    # 4) Load strategy & stop‑loss maps (keyed by base UID)
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

    # 5) First, group by option contract to get individual legs
    leg_cols = [
        "TradeDate",
        "Description",
        "UnderlyingSymbol",
        "Strike",
        "Put/Call",
        "UID_base",
    ]
    df = df.sort_values(leg_cols + ["DateTime"])

    individual_legs = []
    for key, g in df.groupby(leg_cols):
        # only keep complete legs whose quantities net to zero
        if g["Quantity"].sum() != 0:
            continue

        trade_date, desc, sym, strike, pc, uid_base = key

        # entry trades are the negative‑quantity rows
        entries = g[g.Quantity < 0]
        # exit trades are the positive‑quantity rows
        exits   = g[g.Quantity > 0]

        # times
        dt_entry = entries["DateTime"].min()
        dt_exit  = exits["DateTime"].max()

        # weighted average entry price
        entry_qty = entries["Quantity"].abs()
        if entry_qty.sum() > 0:
            px_entry = (entries["TradePrice"] * entry_qty).sum() / entry_qty.sum()
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

        # slippage vs stop‑loss (use base UID for stop-loss lookup)
        sl_pct       = stop_map.get(uid_base, 0.0)
        exp_price    = px_entry * (1 + sl_pct)
        
        # Check for positive slippage conditions (better than expected exit)
        positive_slippage = False
        if sl_pct > 0:  # Only if stop-loss is defined
            # Conditions for positive slippage:
            # 1) Exit price > Entry price (profitable trade)
            # 2) Exit time < 15:45 PM (early exit)
            # 3) Actual exit <= Expected stop-loss (better than planned)
            if (px_exit > px_entry and 
                dt_exit.time() < pd.Timestamp("15:45").time() and 
                px_exit <= exp_price):
                positive_slippage = True
        
        # Calculate slippage
        if positive_slippage:
            # Positive slippage: exit is better than expected (negative value)
            raw_slip = px_exit - exp_price  # This will be negative
        else:
            # Negative slippage: exit is worse than expected (positive value)
            raw_slip = max(px_exit - exp_price, 0.0)
        
        slippage_amt = raw_slip * entries["Multiplier"].iloc[0]
        planned_loss = abs(px_entry - exp_price)  # absolute planned loss amount
        slippage_pct = (raw_slip / planned_loss * 100) if planned_loss else 0.0

        individual_legs.append({
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
            "UID":              uid_base,
            "IsHedge":          entries["IsHedge"].iloc[0],
            "Strategy":         strat_map.get(uid_base, uid_base),
            "Slippage":         slippage_amt,
            "Slippage_pct":     slippage_pct,
        })

    # 6) Now group individual legs by UID and TradeDate to create strategy legs
    legs_df = pd.DataFrame(individual_legs)
    if legs_df.empty:
        pd.DataFrame().to_parquet(_p("legs"), index=False)
        print("✅ legs.parquet written")
        return

    # Group by UID and TradeDate to identify strategies
    strategy_groups = legs_df.groupby(["UID", "TradeDate"])
    
    rows = []
    for (uid, trade_date), strategy_legs in strategy_groups:
        total_legs = len(strategy_legs)
        
        # Sort by description to ensure consistent ordering
        strategy_legs = strategy_legs.sort_values("Description")
        
        for leg_idx, (_, leg) in enumerate(strategy_legs.iterrows(), 1):
            leg_dict = leg.to_dict()
            leg_dict["Legs"] = f"{leg_idx}/{total_legs}"
            rows.append(leg_dict)

    pd.DataFrame(rows).to_parquet(_p("legs"), index=False)
    print("✅ legs.parquet written")


def build_timeline_and_margin() -> None:
    alloc = load_raw_allocation()
    fund  = load_raw_fund_flow()
    if fund.empty:
        print("⚠️  Fund Flow CSV missing or empty.")
        return

    tl = fund[["Date","EndingCash"]].rename(columns={"EndingCash":"BrokerBalance"})

    if not alloc.empty:
        # pivot allocations as before
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
        
        # Forward-fill allocation values to persist until next change
        piv = piv.ffill()
        
        # Merge with timeline and then forward-fill the merged data
        tl = tl.merge(piv.reset_index(), on="Date", how="left")
        
        # Forward-fill allocation columns after merge to handle missing dates
        alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
        if alloc_cols:
            tl[alloc_cols] = tl[alloc_cols].ffill()
            
            # Reset strategies to 0 when they're removed (not present in allocation data)
            # For each allocation column, find the last date it was present and reset to 0 on the next allocation date
            for col in alloc_cols:
                # Extract strategy and underlying from column name
                # Format: Allocation_TS_Allocation_NDX -> TS, NDX
                # Remove "Allocation_" prefix and split by "_Allocation_"
                if col.startswith("Allocation_") and "_Allocation_" in col:
                    # Remove the first "Allocation_" and split by "_Allocation_"
                    parts = col.replace("Allocation_", "", 1).split("_Allocation_")
                    if len(parts) == 2:
                        strategy, underlying = parts[0], parts[1]
                        
                        # Get the last date this strategy+underlying combination was present
                        strategy_data = alloc[(alloc["Strategy"] == strategy) & (alloc["UnderlyingSymbol"] == underlying)]
                        if not strategy_data.empty:
                            # Parse the date correctly (dayfirst=True for DD/MM/YY format)
                            last_date_str = strategy_data["Date"].max()
                            last_date = pd.to_datetime(last_date_str, format="%d/%m/%y", dayfirst=True)
                            
                            # Find the next allocation date after the last date this strategy was present
                            all_dates = pd.to_datetime(alloc["Date"], format="%d/%m/%y", dayfirst=True)
                            next_allocation_date = all_dates[all_dates > last_date].min()
                            
                            if pd.notna(next_allocation_date):
                                # Reset to 0 from the next allocation date onwards
                                tl.loc[tl["Date"] >= next_allocation_date, col] = 0

        # group margin by Strategy (which corresponds to UID_base)
        margin_df = (
            alloc.groupby(["Date","Strategy"])["Margin"]
                 .sum().reset_index()
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

    nav_rows, units_prev, nav_prev = [], bb.iloc[0]/100.0, 100.0
    for dt in dr:
        bal   = float(bb.loc[dt])
        flow  = float(flows_full.loc[dt])
        units = units_prev + flow/nav_prev
        nav   = bal/units if units else np.nan
        nav_rows.append({
            "Date":          dt,
            "NAV":           nav,
            "Units":         units,
            "BrokerBalance": bal,
        })
        units_prev, nav_prev = units, nav

    pd.DataFrame(nav_rows).to_parquet(_p("nav"), index=False)
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
