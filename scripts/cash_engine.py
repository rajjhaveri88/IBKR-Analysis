# scripts/cash_engine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List


def _find_date_col(df: pd.DataFrame,
                   candidates: List[str] | None = None) -> str:
    """Return column that parses as dates on ≥80 % of rows."""
    if candidates is None:
        candidates = ["Date", "ActivityDate", "FromDate", "ToDate"]

    for col in candidates:
        if col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[:5], dayfirst=True)
                return col
            except Exception:
                pass

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if parsed.notna().mean() >= 0.8:
            return col
    raise KeyError("No date‑like column found in Funds Flow Statement")


def build_cash_timeline(fund_flow: pd.DataFrame,
                        allocation: pd.DataFrame) -> pd.DataFrame:
    """
    Daily timeline enforcing reset‑to‑zero allocation snapshots.
    """
    fund_flow  = fund_flow.copy()
    allocation = allocation.copy()
    allocation.columns = allocation.columns.str.strip()

    # Parse dates
    dcol = _find_date_col(fund_flow)
    fund_flow["Date"]  = pd.to_datetime(fund_flow[dcol], dayfirst=True)
    allocation["Date"] = pd.to_datetime(allocation["Date"], dayfirst=True)

    # Calendar
    all_days = pd.date_range(fund_flow["Date"].min(),
                             fund_flow["Date"].max(), freq="D")
    cal = pd.DataFrame(index=all_days)

    # Broker balance
    broker = (
        fund_flow[["Date", "EndingCash"]]
        .set_index("Date")
        .rename(columns={"EndingCash": "BrokerBalance"})
        .reindex(all_days)
        .ffill()
    )
    cal = cal.join(broker)

    # Clean numeric
    allocation["Allocation"] = (
        allocation["Allocation"]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
        .replace({"": "0", ".": "0"})
        .astype(float)
    )

    # Snapshot matrix
    allocation["col"] = allocation["Strategy"] + "_" + allocation["UnderlyingSymbol"]
    snap = (
        allocation.groupby(["Date", "col"])["Allocation"]
        .sum()
        .unstack(fill_value=np.nan)
        .reindex(all_days)
    )

    # Set NaN -> 0 on snapshot rows only, then global forward‑fill
    is_snapshot = snap.notna().any(axis=1)
    snap.loc[is_snapshot] = snap.loc[is_snapshot].fillna(0)
    snap = snap.ffill().fillna(0)

    # Prefix & join
    snap = snap.add_prefix("Allocation_")
    cal = cal.join(snap)

    # Totals & spare
    alloc_cols = [c for c in cal.columns if c.startswith("Allocation_")]
    cal["TotalAllocation"] = cal[alloc_cols].sum(axis=1)
    cal["Spare"]           = cal["BrokerBalance"] - cal["TotalAllocation"]

    cal = cal.reset_index().rename(columns={"index": "Date"})
    return cal[["Date", "BrokerBalance", "TotalAllocation", "Spare", *alloc_cols]]
