import pandas as pd
import numpy as np
from scripts.loaders import load_trades, load_timeline, load_strategies, load_uid_margin

def debug_dashboard_pnl():
    """Debug script that mimics the exact dashboard logic to find the 2-dollar difference"""
    
    print("=== Loading Data ===")
    trades = load_trades()
    timeline = load_timeline()
    strat_df = load_strategies()
    uid_margin = load_uid_margin()
    
    # Get unique strategies and UIDs (mimicking dashboard logic)
    strategies = strat_df["Strategy_id"].unique()
    uids = trades["UID"].unique()
    
    # Sort strategies in order: TS, UT, SS, Error (mimicking dashboard logic)
    strategy_order = []
    for prefix in ["TS", "UT", "SS"]:
        strategy_order.extend([s for s in strategies if s.startswith(prefix)])
    strategy_order.extend([s for s in strategies if s == "ERROR"])
    strategies = strategy_order
    
    # Sort UIDs in order: TS, UT, SS, Error (mimicking dashboard logic)
    uid_order = []
    for prefix in ["TS", "UT", "SS"]:
        uid_order.extend([u for u in uids if u.startswith(prefix)])
    uid_order.extend([u for u in uids if u == "Error"])
    uids = uid_order
    
    print(f"Strategies (ordered): {strategies}")
    print(f"UIDs (ordered): {uids}")
    
    # Mimic the exact dashboard logic for strategy consolidated view
    print("\n=== Strategy Dashboard Logic ===")
    strategy_rows = []
    
    for strategy in strategies:
        df = trades[trades["Strategy"] == strategy].copy()
        if df.empty:
            continue

        # — P&L & risk metrics (same as dashboard)
        df = df.copy()
        df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl = daily_pnl.sum()
        
        strategy_rows.append({
            "Strategy": strategy,
            "Net P&L": f"{net_pnl:,.0f}",
            "Raw P&L": net_pnl
        })
        print(f"{strategy}: {net_pnl:,.2f} -> {f'{net_pnl:,.0f}'}")
    
    # Calculate total P&L from formatted strings (mimicking dashboard)
    strategy_total = sum(float(row["Net P&L"].replace(",", "")) for row in strategy_rows)
    print(f"Strategy Total (from formatted strings): {strategy_total:,.0f}")
    
    # Mimic the exact dashboard logic for UID consolidated view
    print("\n=== UID Dashboard Logic ===")
    uid_rows = []
    
    for uid in uids:
        df = trades[trades["UID"] == uid].copy()
        if df.empty:
            continue

        # — P&L & risk metrics (same as dashboard)
        df = df.copy()
        df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl = daily_pnl.sum()
        
        uid_rows.append({
            "UID": uid,
            "Net P&L": f"{net_pnl:,.0f}",
            "Raw P&L": net_pnl
        })
        print(f"{uid}: {net_pnl:,.2f} -> {f'{net_pnl:,.0f}'}")
    
    # Calculate total P&L from formatted strings (mimicking dashboard)
    uid_total = sum(float(row["Net P&L"].replace(",", "")) for row in uid_rows)
    print(f"UID Total (from formatted strings): {uid_total:,.0f}")
    
    # Calculate difference
    difference = abs(strategy_total - uid_total)
    print(f"\n=== Dashboard Difference Analysis ===")
    print(f"Strategy Total: {strategy_total:,.0f}")
    print(f"UID Total: {uid_total:,.0f}")
    print(f"Difference: {difference:,.0f}")
    
    if abs(difference - 2.00) < 0.01:
        print("*** FOUND THE 2-DOLLAR DIFFERENCE! ***")
        
        # Check if it's a rounding issue
        print("\n=== Rounding Analysis ===")
        strategy_raw_total = sum(row["Raw P&L"] for row in strategy_rows)
        uid_raw_total = sum(row["Raw P&L"] for row in uid_rows)
        
        print(f"Strategy Raw Total: {strategy_raw_total:,.2f}")
        print(f"UID Raw Total: {uid_raw_total:,.2f}")
        print(f"Raw Difference: {abs(strategy_raw_total - uid_raw_total):,.2f}")
        
        # Check individual rounding differences
        print("\n=== Individual Rounding Differences ===")
        for i, (strategy_row, uid_row) in enumerate(zip(strategy_rows, uid_rows)):
            strategy_raw = strategy_row["Raw P&L"]
            uid_raw = uid_row["Raw P&L"]
            strategy_rounded = round(strategy_raw, 0)
            uid_rounded = round(uid_raw, 0)
            
            if strategy_rounded != uid_rounded:
                print(f"{strategy_row['Strategy']} vs {uid_row['UID']}:")
                print(f"  Strategy: {strategy_raw:,.2f} -> {strategy_rounded:,.0f}")
                print(f"  UID: {uid_raw:,.2f} -> {uid_rounded:,.0f}")
                print(f"  Difference: {abs(strategy_rounded - uid_rounded):,.0f}")

if __name__ == "__main__":
    debug_dashboard_pnl() 