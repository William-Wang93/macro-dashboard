import os
from datetime import date

import dotenv
import pandas as pd
import plotly.express as px
import streamlit as st
from fredapi import Fred

# -----------------------------
# Config & setup
# -----------------------------
st.set_page_config(
    page_title="Macro Dashboard",
    layout="wide",
)

dotenv.load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    st.error("FRED_API_KEY not found in .env. Please set it before running the app.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

SERIES = {
    "CPIAUCSL": "CPI (Urban Consumers, SA)",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "DGS10": "10Y Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "T10YIE": "10Y Inflation Expectations",
}

GLOBAL_START = date(1960, 1, 1)  # 👈 hard min for all charts


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner="Loading FRED data…")
def load_fred_data(start: date, end: date) -> pd.DataFrame:
    """
    Load all required FRED series from `start` to `end` (inclusive)
    and return a single long-format dataframe with YoY % change.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    frames = []
    for series_id, series_label in SERIES.items():
        s = fred.get_series(
            series_id,
            observation_start=start_str,
            observation_end=end_str,
        )
        df = s.to_frame(name="value").reset_index()
        df = df.rename(columns={"index": "date"})
        df["variable"] = series_id
        df["label"] = series_label
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.sort_values("date")

    # 👇 Year-over-year % change (12 months)
    combined_df["yoy"] = (
        combined_df.groupby("variable")["value"].pct_change(12) * 100
    )

    return combined_df


# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("Macro Overview Dashboard")

    # 1️⃣ Load full history from 1960 → today
    today = date.today()
    combined_df = load_fred_data(GLOBAL_START, today)

    # Diagnostics (you can comment this out later)
    st.write(
        "Full data date range:",
        combined_df["date"].min(),
        "→",
        combined_df["date"].max(),
    )

    # 2️⃣ Sidebar controls
    st.sidebar.header("Controls")

    min_date = combined_df["date"].min().date()
    max_date = combined_df["date"].max().date()

    # Default start = 1960-01-01 (or earliest available)
    start_date = st.sidebar.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
    )

    end_date = st.sidebar.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    # Placeholder for future module logic if you want
    module = st.sidebar.selectbox(
        "Module",
        options=["Overview"],
        index=0,
    )

    # 3️⃣ Filter data by selected date range
    mask = (
        (combined_df["date"] >= pd.to_datetime(start_date))
        & (combined_df["date"] <= pd.to_datetime(end_date))
    )
    df_filtered = combined_df.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No data in this date range.")
        st.stop()

    # 4️⃣ Headline numbers (still using levels for now)
    latest = (
        df_filtered.sort_values("date").groupby("variable").tail(1).set_index("variable")
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    def fmt_pct(x):
        return f"{x:.2f}%" if pd.notnull(x) else "—"

    with col1:
        st.metric(
            "Inflation (CPI)",
            fmt_pct(latest.loc["CPIAUCSL", "value"])
            if "CPIAUCSL" in latest.index
            else "—",
        )
    with col2:
        st.metric(
            "Unemployment",
            fmt_pct(latest.loc["UNRATE", "value"])
            if "UNRATE" in latest.index
            else "—",
        )
    with col3:
        st.metric(
            "10Y Yield",
            fmt_pct(latest.loc["DGS10", "value"])
            if "DGS10" in latest.index
            else "—",
        )
    with col4:
        st.metric(
            "2Y Yield",
            fmt_pct(latest.loc["DGS2", "value"])
            if "DGS2" in latest.index
            else "—",
        )
    with col5:
        st.metric(
            "10Y Breakeven",
            fmt_pct(latest.loc["T10YIE", "value"])
            if "T10YIE" in latest.index
            else "—",
        )
    with col6:
        st.metric(
            "Payrolls (PAYEMS)",
            f"{latest.loc['PAYEMS', 'value']:,.0f}"
            if "PAYEMS" in latest.index
            else "—",
        )

    st.markdown("---")

    # 5️⃣ Tabs: Overview, Inflation, Labor, Rates, Housing, Markets
    tab_overview, tab_inflation, tab_labor, tab_rates, tab_housing, tab_markets = st.tabs(
        ["Overview", "Inflation", "Labor", "Rates & Curve", "Housing", "Markets"]
    )

    # ---- Helper: drop NaNs in yoy before plotting ----
    def clean_yoy(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=["yoy"])

    # ---- Overview tab: all series ----
    with tab_overview:
        st.subheader("Headline view (YoY % change)")
        df_plot = clean_yoy(df_filtered)

        fig_all = px.line(
            df_plot,
            x="date",
            y="yoy",
            color="label",
            labels={"date": "Date", "yoy": "YoY % Change", "label": "Series"},
            template="plotly_white",
        )
        fig_all.update_xaxes(
            range=[pd.Timestamp(GLOBAL_START), pd.to_datetime(end_date)]
        )
        st.plotly_chart(fig_all, use_container_width=True)

    # ---- Inflation tab ----
    with tab_inflation:
        st.subheader("Inflation & Inflation Expectations (YoY %)")

        infl_vars = ["CPIAUCSL", "T10YIE"]
        df_infl = df_filtered[df_filtered["variable"].isin(infl_vars)]
        df_infl = clean_yoy(df_infl)

        if df_infl.empty:
            st.info("No inflation data in this date range.")
        else:
            fig_infl = px.line(
                df_infl,
                x="date",
                y="yoy",
                color="label",
                labels={"date": "Date", "yoy": "YoY % Change", "label": "Series"},
                template="plotly_white",
            )
            fig_infl.update_xaxes(
                range=[pd.Timestamp(GLOBAL_START), pd.to_datetime(end_date)]
            )
            st.plotly_chart(fig_infl, use_container_width=True)

    # ---- Labor tab ----
    with tab_labor:
        st.subheader("Labor Market (YoY %)")

        labor_vars = ["UNRATE", "PAYEMS"]
        df_labor = df_filtered[df_filtered["variable"].isin(labor_vars)]
        df_labor = clean_yoy(df_labor)

        if df_labor.empty:
            st.info("No labor data in this date range.")
        else:
            fig_labor = px.line(
                df_labor,
                x="date",
                y="yoy",
                color="label",
                labels={"date": "Date", "yoy": "YoY % Change", "label": "Series"},
                template="plotly_white",
            )
            fig_labor.update_xaxes(
                range=[pd.Timestamp(GLOBAL_START), pd.to_datetime(end_date)]
            )
            st.plotly_chart(fig_labor, use_container_width=True)

    # ---- Rates & Curve tab ----
    with tab_rates:
        st.subheader("Rates & Yield Curve (YoY %)")

        rates_vars = ["DGS2", "DGS10"]
        df_rates = df_filtered[df_filtered["variable"].isin(rates_vars)]
        df_rates = clean_yoy(df_rates)

        if df_rates.empty:
            st.info("No rates data in this date range.")
        else:
            fig_rates = px.line(
                df_rates,
                x="date",
                y="yoy",
                color="label",
                labels={"date": "Date", "yoy": "YoY % Change", "label": "Series"},
                template="plotly_white",
            )
            fig_rates.update_xaxes(
                range=[pd.Timestamp(GLOBAL_START), pd.to_datetime(end_date)]
            )
            st.plotly_chart(fig_rates, use_container_width=True)

    # ---- Housing tab (placeholder for now) ----
    with tab_housing:
        st.subheader("Housing")
        st.info(
            "Hook your housing series (Case-Shiller, housing starts, mortgage rates, etc.) "
            "into this tab by adding FRED series IDs to the `SERIES` dict and filtering here."
        )

    # ---- Markets tab (placeholder for now) ----
    with tab_markets:
        st.subheader("Markets & Risk Sentiment")
        st.info(
            "You can extend this tab with equity indices, credit spreads, or VIX once you "
            "add those data sources."
        )


if __name__ == "__main__":
    main()

