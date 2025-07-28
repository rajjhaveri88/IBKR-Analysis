import pandas as pd
import numpy as np
from scripts.loaders import load_trades, load_timeline, load_strategies, load_uid_margin

def debug_pnl_difference():
    """Debug script to identify the 2-dollar P&L difference between Strategy and UID pages"""
    
    print("=== Loading Data ===")
    trades = load_trades()
    timeline = load_timeline()
    strat_df = load_strategies()
    uid_margin = load_uid_margin()
    
    print(f"Total trades loaded: {len(trades)}")
    
    # Get unique strategies and UIDs
    strategies = strat_df["Strategy_id"].unique()
    uids = trades["UID"].unique()
    
    print(f"Strategies: {len(strategies)}")
    print(f"UIDs: {len(uids)}")
    
    # Check for duplicates and their impact
    print(f"\n=== Duplicate Analysis ===")
    duplicate_trades = trades.duplicated(subset=["TradeDate", "UID", "Strategy", "NetCash"], keep=False)
    print(f"Duplicate trades found: {duplicate_trades.sum()}")
    
    if duplicate_trades.any():
        duplicates = trades[duplicate_trades].copy()
        duplicates = duplicates.sort_values(["TradeDate", "UID", "Strategy"])
        print("Sample duplicates:")
        print(duplicates[["TradeDate", "UID", "Strategy", "NetCash"]].head(10))
        
        # Check if duplicates are causing the difference
        unique_trades = trades.drop_duplicates(subset=["TradeDate", "UID", "Strategy", "NetCash"])
        print(f"Trades after removing duplicates: {len(unique_trades)}")
        
        # Recalculate with unique trades
        strategy_total_unique = 0
        for strategy in strategies:
            df = unique_trades[unique_trades["Strategy"] == strategy].copy()
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
            net_pnl = daily_pnl.sum()
            strategy_total_unique += net_pnl
        
        uid_total_unique = 0
        for uid in uids:
            df = unique_trades[unique_trades["UID"] == uid].copy()
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
            daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
            net_pnl = daily_pnl.sum()
            uid_total_unique += net_pnl
        
        print(f"Strategy Total (with duplicates): {strategy_total_unique:,.2f}")
        print(f"UID Total (with duplicates): {uid_total_unique:,.2f}")
        print(f"Difference (with duplicates): {abs(strategy_total_unique - uid_total_unique):,.2f}")
    
    # Calculate P&L for each strategy (with duplicates)
    print("\n=== Strategy P&L Calculations (with duplicates) ===")
    strategy_pnl = {}
    strategy_total = 0
    
    for strategy in strategies:
        df = trades[trades["Strategy"] == strategy].copy()
        if df.empty:
            continue
            
        df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl = daily_pnl.sum()
        strategy_pnl[strategy] = net_pnl
        strategy_total += net_pnl
        print(f"{strategy}: {net_pnl:,.2f}")
    
    print(f"Strategy Total P&L: {strategy_total:,.2f}")
    
    # Calculate P&L for each UID (with duplicates)
    print("\n=== UID P&L Calculations (with duplicates) ===")
    uid_pnl = {}
    uid_total = 0
    
    for uid in uids:
        df = trades[trades["UID"] == uid].copy()
        if df.empty:
            continue
            
        df["Date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%y", dayfirst=True)
        daily_pnl = df.groupby("Date")["NetCash"].sum().sort_index()
        net_pnl = daily_pnl.sum()
        uid_pnl[uid] = net_pnl
        uid_total += net_pnl
        print(f"{uid}: {net_pnl:,.2f}")
    
    print(f"UID Total P&L: {uid_total:,.2f}")
    
    # Calculate difference
    difference = abs(strategy_total - uid_total)
    print(f"\n=== Difference Analysis ===")
    print(f"Strategy Total: {strategy_total:,.2f}")
    print(f"UID Total: {uid_total:,.2f}")
    print(f"Difference: {difference:,.2f}")
    
    # Check if the difference is exactly 2.00
    if abs(difference - 2.00) < 0.01:
        print("*** FOUND THE 2-DOLLAR DIFFERENCE! ***")
        
        # Let's check if there's a specific trade or set of trades causing this
        print("\n=== Detailed Analysis ===")
        
        # Check if there are any trades that appear in one calculation but not the other
        strategy_trades = set()
        uid_trades = set()
        
        for strategy in strategies:
            df = trades[trades["Strategy"] == strategy]
            for _, row in df.iterrows():
                strategy_trades.add((row["TradeDate"], row["UID"], row["Strategy"], row["NetCash"]))
        
        for uid in uids:
            df = trades[trades["UID"] == uid]
            for _, row in df.iterrows():
                uid_trades.add((row["TradeDate"], row["UID"], row["Strategy"], row["NetCash"]))
        
        print(f"Strategy trades: {len(strategy_trades)}")
        print(f"UID trades: {len(uid_trades)}")
        
        if strategy_trades != uid_trades:
            print("*** TRADE SETS ARE DIFFERENT! ***")
            strategy_only = strategy_trades - uid_trades
            uid_only = uid_trades - strategy_trades
            print(f"Trades only in strategy calculation: {len(strategy_only)}")
            print(f"Trades only in UID calculation: {len(uid_only)}")
            
            if strategy_only:
                print("Trades only in strategy:")
                for trade in list(strategy_only)[:5]:
                    print(f"  {trade}")
            
            if uid_only:
                print("Trades only in UID:")
                for trade in list(uid_only)[:5]:
                    print(f"  {trade}")
    
    # Check for any rounding differences
    print(f"\n=== Rounding Check ===")
    strategy_rounded = round(strategy_total, 0)
    uid_rounded = round(uid_total, 0)
    print(f"Strategy Total (rounded): {strategy_rounded:,.0f}")
    print(f"UID Total (rounded): {uid_rounded:,.0f}")
    print(f"Rounded Difference: {abs(strategy_rounded - uid_rounded):,.0f}")

if __name__ == "__main__":
    debug_pnl_difference() 