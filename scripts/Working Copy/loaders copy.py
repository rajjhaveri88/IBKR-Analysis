# scripts/loaders.py   ✓ 2025‑07‑25
# ──────────────────────────────────────────────────────────────
"""
Centralised loaders for raw CSVs and processed parquet files.

Key features
------------
▪ Finds CSVs either in repo‑root (e.g. "Data - Allocation and Margin.csv")
  **or** in a dedicated ./Data/ folder.
▪ Strips whitespace from every header.
▪ Normalises column aliases (e.g. Strategy_id ➜ Strategy).
▪ Caches results with Streamlit so dashboard reloads stay snappy.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── folder helpers ─────────────────────────────────────────────
# Prefer ./Data if it exists; otherwise default to project root.
DATA_DIR = Path("Data") if Path("Data").is_dir() else Path(".")
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _p(name: str) -> Path:
    """Return a processed‑parquet path, e.g. timeline.parquet."""
    return PROC_DIR / f"{name}.parquet"

def _clean(df: pd.DataFrame, rename_map: dict | None = None) -> pd.DataFrame:
    """Strip spaces from headers and apply optional renames."""
    df.columns = df.columns.str.strip()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _find(fname_variants: list[str]) -> Path | None:
    """Return the first existing path among a list of filename variants."""
    for fn in fname_variants:
        fp = DATA_DIR / fn
        if fp.exists():
            return fp
    return None

# ╔═════════ RAW CSV LOADERS ═════════╗
@st.cache_data(show_spinner=False)
def load_raw_trades() -> pd.DataFrame:
    fp = _find(["Trades.csv"])
    return _clean(pd.read_csv(fp)) if fp else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_raw_allocation() -> pd.DataFrame:
    fp = _find([
        "Allocation and Margin.csv",
        "Data - Allocation and Margin.csv"
    ])
    if not fp:
        return pd.DataFrame()

    df = _clean(pd.read_csv(fp))

    # ensure UID exists (older dumps used StrategyUID)
    if "UID" not in df.columns and "StrategyUID" in df.columns:
        df = df.rename(columns={"StrategyUID": "UID"})

    return df

@st.cache_data(show_spinner=False)
def load_raw_fund_flow() -> pd.DataFrame:
    fp = _find([
        "Funds Flow Statement.csv",
        "Fund Flow Statement.csv",
        "Data - Fund Flow Statement.csv"
    ])
    return _clean(pd.read_csv(fp)) if fp else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_raw_strategies() -> pd.DataFrame:
    fp = _find(["Strategies.csv", "Data - Strategies.csv"])
    if not fp:
        return pd.DataFrame()

    # normalise Strategy column right here
    return _clean(pd.read_csv(fp), rename_map={"Strategy_id": "Strategy"})

# ╔═════════ PROCESSED PARQUET LOADERS ═════════╗
@st.cache_data(show_spinner=False)
def load_trades() -> pd.DataFrame:
    pp = _p("legs")
    if pp.exists():
        return pd.read_parquet(pp)

    # fallback — raw CSV
    df = load_raw_trades()
    if "slippage_abs" not in df.columns:
        df["slippage_abs"] = 0.0
    return df

@st.cache_data(show_spinner=False)
def load_timeline() -> pd.DataFrame:
    pp = _p("timeline")
    if not pp.exists():
        raise FileNotFoundError("timeline.parquet not found – run scripts/etl.py first.")
    return pd.read_parquet(pp)

@st.cache_data(show_spinner=False)
def load_uid_margin() -> pd.DataFrame:
    return pd.read_parquet(_p("uid_margin")) if _p("uid_margin").exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_strategies() -> pd.DataFrame:
    """
    Strategy look‑up with canonical columns.
    Always enforces `Strategy_id ➜ Strategy` even if processed parquet exists.
    """
    pp = _p("strategies")
    df = pd.read_parquet(pp) if pp.exists() else load_raw_strategies()
    return _clean(df, rename_map={"Strategy_id": "Strategy"})

@st.cache_data(show_spinner=False)
def load_fund_flow() -> pd.DataFrame:
    """
    Broker cash‑balance timeline.
    Prefers processed parquet; falls back to raw CSV if needed.
    """
    pp = _p("fund_flow")
    return pd.read_parquet(pp) if pp.exists() else load_raw_fund_flow()

# ╔═════════ HYBRID ALLOCATION LOADER (for cash_engine) ═════════╗
@st.cache_data(show_spinner=False)
def load_allocation() -> pd.DataFrame:
    """
    Returns a long‑form allocation dataframe with:
    ['Date', 'Strategy', 'UnderlyingSymbol', 'Allocation', 'Margin', 'UID']
    """
    raw = load_raw_allocation()
    if not raw.empty and "Allocation" in raw.columns:
        return raw

    # fallback – melt out of timeline parquet columns Allocation_*
    tl = load_timeline()
    alloc_cols = [c for c in tl.columns if c.startswith("Allocation_")]
    if not alloc_cols:
        return pd.DataFrame()

    return (tl.melt(id_vars="Date", value_vars=alloc_cols,
                    var_name="col", value_name="Allocation")
              .assign(Strategy=lambda d: d["col"].str.split("_").str[1],
                      UnderlyingSymbol=lambda d: d["col"].str.split("_").str[2],
                      Margin=0,
                      UID="NA")
              .drop(columns="col"))
