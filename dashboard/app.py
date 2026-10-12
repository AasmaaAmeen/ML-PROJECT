from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

st.set_page_config(page_title="AI Blood Inventory Dashboard", page_icon="🩸", layout="wide")
st.title("🩸 AI-Driven Blood Inventory Optimization")
st.caption("TFT demand/activity forecasting • PPO inventory simulation")

forecast_path = DATA / "tft_forecast.csv"
results_path = DATA / "ppo_inventory_results.csv"
evaluation_path = DATA / "ppo_evaluation.csv"
series_path = DATA / "blood_timeseries.csv"

def read_csv(path):
    return pd.read_csv(path) if path.exists() else None

forecast = read_csv(forecast_path)
results = read_csv(results_path)
evaluation = read_csv(evaluation_path)
series = read_csv(series_path)

with st.sidebar:
    st.header("Dashboard controls")
    st.caption("Select a blood group to filter the available results.")
    choices = ["All"] 
    for frame in [forecast, results, series]:
        if frame is not None and "blood_group" in frame:
            choices += [x for x in sorted(frame["blood_group"].dropna().astype(str).unique()) if x not in choices]
    group = st.selectbox("Blood group", choices)

def filtered(df):
    if df is None or group == "All" or "blood_group" not in df:
        return df
    return df[df["blood_group"].astype(str) == group]

if series is not None and {"date","blood_group","demand"}.issubset(series.columns):
    st.subheader("Historical monthly donation activity")
    hist=filtered(series).copy()
    hist["date"]=pd.to_datetime(hist["date"],errors="coerce")
    if not hist.empty:
        fig=px.line(hist,x="date",y="demand",color="blood_group" if group=="All" else None,
                    markers=True,title="Historical recorded donation counts")
        st.plotly_chart(fig,use_container_width=True)
    st.caption("Donation counts are a supply/activity proxy, not direct hospital demand.")

if forecast is not None:
    st.subheader("TFT forecast")
    f=filtered(forecast).copy()
    if not f.empty:
        date_col="date" if "date" in f else None
        val_col="tft_predicted_demand" if "tft_predicted_demand" in f else None
        if date_col and val_col:
            f[date_col]=pd.to_datetime(f[date_col],errors="coerce")
            fig=px.line(f,x=date_col,y=val_col,color="blood_group" if group=="All" and "blood_group" in f else None,
                        markers=True,title="Predicted demand / activity")
            st.plotly_chart(fig,use_container_width=True)
        st.dataframe(f,use_container_width=True)
    else:
        st.info("No forecast rows for this selection.")
else:
    st.warning("Forecast file not found. Run preprocessing and TFT training/forecasting first.")

if results is not None:
    st.subheader("PPO inventory recommendations")
    r=filtered(results).copy()
    if not r.empty:
        numeric=[c for c in ["predicted_demand","recommended_order","shortage","excess"] if c in r]
        cols=st.columns(min(4,max(1,len(numeric))))
        for i,c in enumerate(numeric):
            cols[i].metric(c.replace("_"," ").title(),f"{r[c].sum():,.2f}")
        if {"blood_group","recommended_order"}.issubset(r.columns):
            fig=px.bar(r,x="blood_group",y="recommended_order",color="blood_group",
                       title="Recommended orders by blood group")
            st.plotly_chart(fig,use_container_width=True)
        st.dataframe(r,use_container_width=True)
else:
    st.info("PPO results will appear after running rl.generate_results.")

if evaluation is not None:
    st.subheader("PPO evaluation")
    st.dataframe(evaluation,use_container_width=True)

st.divider()
st.caption("Research prototype only. Validate with real demand, stock, expiry and operational data before interpreting recommendations.")
