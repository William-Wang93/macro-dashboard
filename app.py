"""
Macro Dashboard — Streamlit MVP

Features
- Pulls live macro time series from FRED (and optional Yahoo Finance)
- Interactive Plotly charts with recession shading
- Sidebar module picker + date range
- Lightweight caching
- AI helper stub to turn natural-language prompts into chart configs (plug in your LLM key)

Quick setup
1) python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
2) pip install -r requirements.txt  (see inline list below)
3) Export FRED_API_KEY in your shell or create a .env with FRED_API_KEY=...
4) streamlit run app.py

requirements.txt
---------------
streamlit
pandas
numpy
plotly
python-dotenv
fredapi
yfinance
requests-cache

Rename this file to app.py when you download it.
"""

import os
from datetime import date, timedelta
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from fredapi import Fred
import yfinance as yf
import requests_cache

# ------------------------
# Config & helpers
# ------------------------
load_dotenv()
st.set_page_config(page_title="Macro Dashboard — MVP", page_icon="📊", layout="wide")

# Cache HTTP requests for non-FRED calls (yfinance uses yfinance backend)
requests_cache.install_cache("macro_cache", expire_after=60 * 15)  # 15 min

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("FRED_API_KEY not found. Set it in your environment or a .env file.")
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# Recession series (0/1) for shading
USREC_SERIES = "USREC"  # NBER recessions (monthly)

# Common series map for convenience
SERIES = {
    # Prices / Inflation
    "CPI (All Items)": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "PCE (Headline)": "PCEPI",
    "Core PCE": "PCEPILFE",
    "10Y Breakeven (T10YIE)": "T10YIE",

    # Labor
    "Unemployment Rate": "UNRATE",
    "Nonfarm Payrolls": "PAYEMS",
    "Initial Jobless Claims": "ICSA",

    # Rates & Curve
    "2Y Treasury": "DGS2",
    "10Y Treasury": "DGS10",

    # Housing
    "Case-Shiller 20-City (NSA)": "SPCS20RSA",
    "Case-Shiller National (SA)": "CSUSHPINSA",
    "30Y Mortgage Rate": "MORTGAGE30US",
    "Median New Home Price": "MSPUS",

    # Growth
    "Industrial Production Index": "INDPRO",
    "Retail Sales (Nominal)": "RSAFS",
}

MARKETS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Bitcoin": "BTC-USD",
    "WTI Crude": "CL=F",
    "Gold": "GC=F",
}

# ------------------------
# Data loaders
# ------------------------
@st.cache_data(show_spinner=False, ttl=60 * 15)
def fred_series(series_id: str) -> pd.Series:
    if fred is None:
        return pd.Series(dtype=float)
    s = fred.get_series(series_id)
    s.name = series_id
    # Try to coerce to datetime index (FRED returns PeriodIndex sometimes)
    s.index = pd.to_datetime(s.index)
    return s.dropna()

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fred_dataframe(series_ids: List[str]) -> pd.DataFrame:
    frames = []
    for sid in series_ids:
        s = fred_series(sid)
        if not s.empty:
            frames.append(s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    return df

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fred_recession_flags() -> pd.Series:
    s = fred_series(USREC_SERIES)
    return s

@st.cache_data(show_spinner=False, ttl=60 * 15)
def yahoo_series(ticker: str) -> pd.Series:
    data = yf.download(ticker, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    s = data["Close"].copy()
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s

@st.cache_data(show_spinner=False, ttl=60 * 15)
def yahoo_dataframe(tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        s = yahoo_series(t)
        if not s.empty:
            frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)

# ------------------------
# Visualization
# ------------------------

def add_recession_shades(fig: go.Figure, rec: pd.Series):
    if rec is None or rec.empty:
        return fig
    rec = rec.copy().astype(int)
    rec.index = pd.to_datetime(rec.index)
    in_rec = False
    start = None
    for dt, val in rec.items():
        if val == 1 and not in_rec:
            in_rec = True
            start = dt
        if val == 0 and in_rec:
            in_rec = False
            end = dt
            fig.add_vrect(x0=start, x1=end, opacity=0.1, line_width=0)
    # If currently in a recession per series
    if in_rec and start is not None:
        fig.add_vrect(x0=start, x1=rec.index.max(), opacity=0.1, line_width=0)
    return fig


def line_chart(df: pd.DataFrame, title: str, yaxis_title: Optional[str] = None) -> go.Figure:
    fig = px.line(df, x=df.index, y=df.columns, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
    return fig

# ------------------------
# AI Helper (stub)
# ------------------------

def llm_suggest_chart(user_prompt: str, available: Dict[str, str]) -> Dict:
    """Stub function to demonstrate AI-driven chart suggestions.
    Replace body with your favorite LLM call to turn natural-language prompts
    into a dict like {"series": ["CPIAUCSL", "UNRATE"], "transform": "yoy"}.
    """
    # Extremely naive heuristic example:
    prompt = user_prompt.lower()
    selection = []
    if "inflation" in prompt or "cpi" in prompt:
        selection.append("CPIAUCSL")
    if "unemployment" in prompt or "labor" in prompt:
        selection.append("UNRATE")
    if "yield" in prompt or "curve" in prompt:
        selection.extend(["DGS10", "DGS2"])
    if not selection:
        selection = ["CPIAUCSL"]
    return {"series": selection, "transform": None}


def transform_df(df: pd.DataFrame, transform: Optional[str]) -> pd.DataFrame:
    if df.empty or not transform:
        return df
    x = df.copy()
    if transform == "yoy":
        return x.pct_change(12) * 100
    if transform == "mom":
        return x.pct_change(1) * 100
    if transform == "diff":
        return x.diff()
    return df

# ------------------------
# UI
# ------------------------

st.title("📊 Macro Dashboard — MVP")
st.caption("Interactive macro trends with live data (FRED + Markets).")

min_date = date(1960, 1, 1)
max_date = date.today()
def_start = max_date - timedelta(days=365 * 5)

with st.sidebar:
    st.header("Controls")
    start = st.date_input("Start date", value=def_start, min_value=min_date, max_value=max_date)
    end = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    module = st.selectbox(
        "Module",
        ["Overview", "Inflation", "Labor", "Rates & Curve", "Housing", "Markets", "Affordability (beta)", "Ask the data (AI)"]
    )

# Recession flags
rec = fred_recession_flags() if fred else pd.Series(dtype=int)

# Small helper to pull last value with safe formatting

def last_val(series_id: str, transform: Optional[str] = None):
    s = fred_series(series_id)
    if s.empty:
        return None
    if transform == "yoy":
        s = s.pct_change(12) * 100
    return float(s.dropna().iloc[-1])

# KPI cards at the top of Overview

def kpi_cards():
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        v = last_val("CPIAUCSL", "yoy")
        st.metric("Inflation YoY (CPI)", f"{v:.1f}%" if v is not None else "—")
    with c2:
        v = last_val("UNRATE")
        st.metric("Unemployment", f"{v:.1f}%" if v is not None else "—")
    with c3:
        v10 = last_val("DGS10")
        st.metric("10Y Treasury", f"{v10:.2f}%" if v10 is not None else "—")
    with c4:
        yc = fred_dataframe(["DGS10", "DGS2"]) if fred else pd.DataFrame()
        sp = (yc["DGS10"] - yc["DGS2"]).dropna().iloc[-1] if not yc.empty else None
        st.metric("Curve (10Y−2Y)", f"{sp:.2f} pp" if sp is not None else "—")
    with c5:
        mk = yahoo_series(MARKETS["S&P 500"]) if MARKETS else pd.Series(dtype=float)
        if not mk.empty:
                try:
                    # ensure it's properly indexed and numeric
                    mk = mk.dropna()
                    if len(mk) >= 2:
                        start = mk[mk.index.year == mk.index.max().year].iloc[0]
                        end = mk.iloc[-1]
                        ytd = (end / start - 1) * 100.0
                        ytd = float(ytd)
                        st.metric("S&P YTD", f"{ytd:.1f}%")
                    else:
                        st.metric("S&P YTD", "-")
                except Exception as e:
                    st.metric("S&P YTD", "-")
        else:
            st.metric("S&P YTD", "-")



# --------------- Overview ---------------
if module == "Overview":
    st.subheader("Key Indicators")
    kpi_cards()
    ids = ["CPIAUCSL", "UNRATE", "PAYEMS", "DGS10", "DGS2", "T10YIE"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
        fig = line_chart(df.loc[mask], title="Headline view")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True)

    # Yield curve spread
    st.markdown("**Yield Curve (10Y − 2Y)**")
    yc = fred_dataframe(["DGS10", "DGS2"])
    st.subheader("Key Indicators")
    ids = ["CPIAUCSL", "UNRATE", "PAYEMS", "DGS10", "DGS2", "T10YIE"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
        fig = line_chart(df.loc[mask], title="Headline view")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True, key="overview_headline")

    # Yield curve spread
    st.markdown("**Yield Curve (10Y − 2Y)**")
    yc = fred_dataframe(["DGS10", "DGS2"])
    if not yc.empty:
        spread = (yc["DGS10"] - yc["DGS2"]).to_frame("10Y-2Y (pp)")
        spread = spread.loc[(spread.index >= pd.to_datetime(start)) & (spread.index <= pd.to_datetime(end))]
        fig2 = line_chart(spread, title="Yield Curve Spread", yaxis_title="percentage points")
        fig2 = add_recession_shades(fig2, rec)
        st.plotly_chart(fig2, use_container_width=True, key="overview_curve")

# --------------- Inflation ---------------
elif module == "Inflation":
    st.subheader("Inflation")
    ids = ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "T10YIE"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        yoy = df.pct_change(12) * 100
        yoy.columns = [c + " (YoY %)" for c in yoy.columns]
        fig = line_chart(yoy, title="Inflation — YoY %")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True, key="inflation_yoy")

# --------------- Labor ---------------
elif module == "Labor":
    st.subheader("Labor Market")
    ids = ["UNRATE", "PAYEMS", "ICSA"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        fig = line_chart(df, title="Unemployment, Payrolls, Claims")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True, key="labor_overview")

# --------------- Rates & Curve ---------------
elif module == "Rates & Curve":
    st.subheader("Rates & Yield Curve")
    ids = ["DGS2", "DGS10"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        fig = line_chart(df, title="Treasury Yields (%)", yaxis_title="%")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True, key="rates_yields")

        spread = (df["DGS10"] - df["DGS2"]).to_frame("10Y-2Y (pp)")
        fig2 = line_chart(spread, title="Curve Spread (10Y − 2Y)", yaxis_title="pp")
        fig2 = add_recession_shades(fig2, rec)
        st.plotly_chart(fig2, use_container_width=True, key="rates_curve_spread")

# --------------- Housing ---------------
elif module == "Housing":
    st.subheader("Housing")
    ids = ["CSUSHPINSA", "SPCS20RSA", "MORTGAGE30US", "MSPUS"]
    df = fred_dataframe(ids)
    if df.empty:
        st.info("Connect your FRED key to see data.")
    else:
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        fig = line_chart(df, title="Home Prices, Mortgage Rates, Median Price")
        fig = add_recession_shades(fig, rec)
        st.plotly_chart(fig, use_container_width=True, key="housing_prices")

# --------------- Affordability (beta) ---------------
elif module == "Affordability (beta)":
    st.subheader("Housing Affordability (beta)")
    # Simple payment model: 20% down, 30y mortgage, monthly payment from MSPUS and MORTGAGE30US
    price = fred_series("MSPUS")
    mort = fred_series("MORTGAGE30US")
    if price.empty or mort.empty:
        st.info("Need MSPUS and MORTGAGE30US.")
    else:
        # Align monthly
        df = pd.concat([price, mort], axis=1).dropna()
        df.columns = ["MSPUS", "MORTGAGE30US"]
        # Monthly rate from annual %
        r = (df["MORTGAGE30US"] / 100) / 12
        n = 30 * 12
        principal = df["MSPUS"] * 0.8  # 20% down
        # Payment formula: P = r * L / (1 - (1+r)^-n)
        payment = r * principal / (1 - (1 + r) ** (-n))
        affordability_idx = (payment / payment.iloc[0]) * 100
        out = pd.DataFrame({"Monthly Payment": payment, "Affordability Index (=100 at start)": affordability_idx})
        out = out.loc[(out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))]
        fig1 = line_chart(out[["Monthly Payment"]], title="Estimated Monthly Payment (20% down, 30y)", yaxis_title="$ per month")
        st.plotly_chart(fig1, use_container_width=True, key="affordability_payment")
        fig2 = line_chart(out[["Affordability Index (=100 at start)"]], title="Affordability Index", yaxis_title="Index = 100 at start")
        st.plotly_chart(fig2, use_container_width=True, key="affordability_index")

# --------------- Markets ---------------
elif module == "Markets":
    st.subheader("Markets Snapshot")
    tickers = list(MARKETS.values())
    df = yahoo_dataframe(tickers)
    if df.empty:
        st.info("No market data returned.")
    else:
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        df = df.rename(columns={v: k for k, v in MARKETS.items()})
        fig = line_chart(df, title="Markets (Close)")
        st.plotly_chart(fig, use_container_width=True, key="markets_close")

# --------------- Ask the data (AI) ---------------
elif module == "Ask the data (AI)":
    st.subheader("Ask the data (AI)")
    st.caption("Describe what you want to see (e.g., 'Show inflation vs unemployment YoY since 2000').")
    prompt = st.text_input("Your prompt")
    transform = st.selectbox("Transform", ["none", "yoy", "mom", "diff"], index=0)
    if st.button("Suggest & Plot") and prompt:
        spec = llm_suggest_chart(prompt, SERIES)
        ids = spec.get("series", [])
        tf = spec.get("transform") or (None if transform == "none" else transform)
        df = fred_dataframe(ids)
        if df.empty:
            st.info("Connect your FRED key to see data.")
        else:
            df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
            df = transform_df(df, tf)
            label = f"Transform: {tf or 'none'}"
            fig = line_chart(df, title=f"AI-selected series — {label}")
            fig = add_recession_shades(fig, rec)
            st.plotly_chart(fig, use_container_width=True, key="ai_dynamic_chart")

# ------------------------
# Footer
# ------------------------
st.markdown("""
---
**Notes**
- Data: FRED (Federal Reserve), Yahoo Finance (markets). Recession shading uses NBER monthly indicator (USREC).
- Caching: 15 minutes to avoid hammering APIs.
- Extend: add BEA GDP (real/nominal), JOLTS, NFIB, ISM, Atlanta Fed GDPNow, CME FedWatch, Treasury auctions, etc.
- AI: Replace stub with your preferred LLM provider to turn language into chart specs.
""")
