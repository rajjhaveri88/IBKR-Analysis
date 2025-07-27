from scripts.metrics import sharpe
import pandas as pd
def test_sharpe():
    s = pd.Series([0.01, 0.02, -0.005, 0.015])
    assert sharpe(s) > 0 