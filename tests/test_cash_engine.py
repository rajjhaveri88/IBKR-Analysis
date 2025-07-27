from scripts.cash_engine import build_cash_timeline
import pandas as pd

def test_zero_reset():
    fund = pd.DataFrame({"Date": ["01/07/25"], "EndingCash": [1000], "ToDate": ["01/07/25"]})
    alloc = pd.DataFrame({
        "Date": ["01/07/25"],
        "Strategy": ["TS"],
        "UnderlyingSymbol": ["SPX"],
        "Allocation": [500],
        "Margin": [100],
        "Hedged Margin": [0],
    })
    tl = build_cash_timeline(fund, alloc)
    assert tl["TotalAllocation"].iloc[0] == 500 