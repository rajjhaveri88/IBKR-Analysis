# scripts/loaders.py
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("Data") if Path("Data").is_dir() else Path(".")
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _p(name: str) -> Path:
    return PROC_DIR / f"{name}.parquet"

def _find(*names: str) -> Path | None:
    for nm in names:
        fp = DATA_DIR / nm
        if fp.exists():
            return fp
    return None

# ─ Raw CSV loaders ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_trades() -> pd.DataFrame:
    fp = _find("Trades.csv")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_raw_fund_flow() -> pd.DataFrame:
    fp = _find("Fund Flow Statement.csv", "Funds Flow Statement.csv")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    # DAY‑FIRST parse for DD/MM/YY
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_raw_allocation() -> pd.DataFrame:
    fp = _find("Allocation and Margin.csv", "Data - Allocation and Margin.csv")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_raw_strategies() -> pd.DataFrame:
    fp = _find("Strategies.csv", "Data - Strategies.csv")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip().str.replace("Strategy_id", "Strategy")
    return df

# ─ Processed parquet loaders ───────────────────────────────────
@st.cache_data(show_spinner=False)
def load_trades() -> pd.DataFrame:
    pp = _p("legs")
    df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    if not df.empty:
        # TradeDate: keep as dd/mm/yy string format
        # DateTime_entry/exit: full timestamp
        df["DateTime_entry"]  = pd.to_datetime(df["DateTime_entry"], dayfirst=True)
        df["DateTime_exit"]   = pd.to_datetime(df["DateTime_exit"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_timeline() -> pd.DataFrame:
    pp = _p("timeline")
    df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_nav() -> pd.DataFrame:
    pp = _p("nav")
    df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_fund_flow() -> pd.DataFrame:
    return load_raw_fund_flow()

@st.cache_data(show_spinner=False)
def load_allocation() -> pd.DataFrame:
    return load_raw_allocation()

@st.cache_data(show_spinner=False)
def load_strategies() -> pd.DataFrame:
    pp = _p("strategies")
    return pd.read_parquet(pp) if pp.exists() else load_raw_strategies()

@st.cache_data(show_spinner=False)
def load_uid_margin() -> pd.DataFrame:
    pp = _p("uid_margin")
    df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df
