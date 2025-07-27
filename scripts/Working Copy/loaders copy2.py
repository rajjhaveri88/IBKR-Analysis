# scripts/loaders.py   ✓ 2025‑07‑25
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("Data") if Path("Data").is_dir() else Path(".")
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _p(name: str) -> Path:
    return PROC_DIR / f"{name}.parquet"

def _clean(df: pd.DataFrame, rename: dict | None = None) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if rename:
        df = df.rename(columns=rename)
    return df

def _find(*names: str) -> Path | None:
    for nm in names:
        fp = DATA_DIR / nm
        if fp.exists():
            return fp
    return None

# ─── RAW CSV LOADERS ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_trades() -> pd.DataFrame:
    fp = _find("Trades.csv")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "")
    # no Date field here, ETL handles TradeDate/DateTime
    for c in ("TradePrice", "NetCash"):
        if c in df.columns:
            df[c] = (
                df[c]
                  .astype(str)
                  .str.replace(r"[₹$,]", "", regex=True)
                  .replace("", "0")
                  .astype(float)
            )
    return df

@st.cache_data(show_spinner=False)
def load_raw_allocation() -> pd.DataFrame:
    fp = _find("Allocation and Margin.csv", "Data - Allocation and Margin.csv")
    if not fp:
        return pd.DataFrame()
    df = _clean(pd.read_csv(fp))
    # ⬇️ parse Date here so ETL merges align
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_raw_fund_flow() -> pd.DataFrame:
    fp = _find("Fund Flow Statement.csv", "Funds Flow Statement.csv")
    if not fp:
        return pd.DataFrame()
    df = _clean(pd.read_csv(fp))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

@st.cache_data(show_spinner=False)
def load_raw_strategies() -> pd.DataFrame:
    fp = _find("Strategies.csv", "Data - Strategies.csv")
    if not fp:
        return pd.DataFrame()
    return _clean(
        pd.read_csv(fp),
        rename={"Strategy_id": "Strategy"},
    )

# ─── PROCESSED PARQUET LOADERS ─────────────────────────────────
@st.cache_data(show_spinner=False)
def load_trades() -> pd.DataFrame:
    pp = _p("legs")
    return pd.read_parquet(pp) if pp.exists() else pd.DataFrame()

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
def load_strategies() -> pd.DataFrame:
    pp = _p("strategies")
    return pd.read_parquet(pp) if pp.exists() else load_raw_strategies()
