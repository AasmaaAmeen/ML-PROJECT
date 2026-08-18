import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from stable_baselines3 import PPO
from rl.environment import BloodInventoryEnv

st.set_page_config(page_title="Blood Inventory AI", page_icon="🩸", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#fff7f8 0%,#f8fbff 55%,#f7f0ff 100%); }
.hero {
 background: linear-gradient(120deg,#8e1638,#d7264f 55%,#7b2cbf);
 padding:28px 32px; border-radius:22px; color:white;
 box-shadow:0 12px 30px rgba(139,24,61,.20); margin-bottom:22px;
}
.hero h1 { color:white; margin:0 0 7px 0; font-size:32px; }
.hero p { color:#fff5f7; margin:0; font-size:16px; }
.card {
 background:white; border-radius:18px; padding:20px; min-height:115px;
 box-shadow:0 7px 20px rgba(15,23,42,.08); border:1px solid #f1e9ec;
}
.red {border-left:7px solid #e63956}.blue {border-left:7px solid #3b82f6}
.purple {border-left:7px solid #7b2cbf}.green {border-left:7px solid #16a34a}
.label {color:#64748b;font-size:13px;font-weight:700}
.value {color:#172033;font-size:27px;font-weight:800;margin-top:8px}
.ai {
 background:linear-gradient(135deg,#fff1f4,#f5efff);
 border:1px solid #f1c9d4; border-radius:20px; padding:22px; margin:20px 0;
}
.rec {
 background:#ecfdf5;border:2px solid #86efac;border-radius:15px;
 padding:16px;color:#166534;font-size:18px;font-weight:700;
}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#fff1f4,#f8f2ff);}
</style>
""", unsafe_allow_html=True)

DATA="data/blood_timeseries.csv"
FORECAST="data/tft_forecast.csv"
PPO_MODEL="rl/ppo_blood_inventory.zip"

for path, command in [
    (DATA,"python data\\preprocess.py"),
    (FORECAST,"python -m forecasting.forecast"),
    (PPO_MODEL,"python -m rl.train_ppo")
]:
    if not os.path.exists(path):
        st.error(f"Required file is missing. Run: `{command}`")
        st.stop()

hist=pd.read_csv(DATA); hist["date"]=pd.to_datetime(hist["date"])
fc=pd.read_csv(FORECAST); fc["date"]=pd.to_datetime(fc["date"])
groups=sorted(hist["blood_group"].dropna().unique())

st.markdown("""
<div class="hero">
<h1>🩸 AI-Driven Blood Inventory Optimization</h1>
<p>Temporal Fusion Transformer (TFT) forecasting + PPO reinforcement-learning replenishment</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🩸 Blood Bank Controls")
    group=st.selectbox("Select Blood Group",groups)
    st.markdown("---")
    st.success("🧠 TFT Forecasting: Active")
    st.success("🤖 PPO Optimization: Active")
    st.caption("Academic prototype. Inventory is estimated because the donor dataset does not contain observed hospital stock/usage.")

h=hist[hist["blood_group"]==group].sort_values("date")
f=fc[fc["blood_group"]==group].sort_values("date")

current_inventory=max(0,int(round(h.tail(14)["donation_supply"].sum()*0.8)))
pred=float(f.iloc[0]["tft_predicted_demand"])
safety=max(5,int(round(pred*0.30)))

model=PPO.load(PPO_MODEL)
env=BloodInventoryEnv(f["tft_predicted_demand"].values.astype(float), initial_inventory=current_inventory)
obs,_=env.reset()
action,_=model.predict(obs,deterministic=True)
recommended=max(0,int(round(float(action[0]))))

if current_inventory < pred:
    risk="HIGH"; risk_color="#dc2626"
elif current_inventory < pred+safety:
    risk="MEDIUM"; risk_color="#d97706"
else:
    risk="LOW"; risk_color="#16a34a"

st.markdown(f"""
<div style="background:white;border-radius:18px;padding:18px 24px;
box-shadow:0 6px 20px rgba(15,23,42,.06);">
<h2 style="color:#8e1638;margin:0">🩸 {group} Blood Group</h2>
<span style="color:#64748b">AI-generated inventory and replenishment analysis</span>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
cards=[
    ("red","📦 CURRENT ESTIMATED INVENTORY",f"{current_inventory} units"),
    ("blue","🔮 TFT PREDICTED DEMAND",f"{pred:.1f} units"),
    ("purple","🤖 PPO RECOMMENDED ORDER",f"{recommended} units"),
    ("green","⚠️ STOCKOUT RISK",risk),
]
for col,(cls,label,value) in zip((c1,c2,c3,c4),cards):
    with col:
        color=risk_color if label.startswith("⚠️") else "#172033"
        st.markdown(f'<div class="card {cls}"><div class="label">{label}</div><div class="value" style="color:{color}">{value}</div></div>',unsafe_allow_html=True)

st.markdown(f"""
<div class="ai">
<h3 style="color:#a3153c;margin-top:0">🤖 AI Recommendation</h3>
<div class="rec">Recommended replenishment: {recommended} units</div>
<br>
<b>🔮 Predicted demand tomorrow:</b> {pred:.1f} units<br>
<b>📦 Estimated current inventory:</b> {current_inventory} units<br>
<b>🛡️ Safety stock:</b> {safety} units<br>
<b>⚠️ Stockout risk:</b> <span style="color:{risk_color};font-weight:800">{risk}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📈 Demand & Supply Analysis")
recent=h.tail(90)
fig=go.Figure()
fig.add_trace(go.Scatter(x=recent["date"],y=recent["donation_supply"],name="Donation Supply",mode="lines",line=dict(color="#e63956",width=3)))
fig.add_trace(go.Scatter(x=recent["date"],y=recent["estimated_demand"],name="Estimated Demand",mode="lines",line=dict(color="#3b82f6",width=3)))
fig.add_trace(go.Scatter(x=f["date"],y=f["tft_predicted_demand"],name="TFT Forecast",mode="lines+markers",line=dict(color="#7b2cbf",width=4,dash="dot")))
fig.update_layout(height=430,template="plotly_white",hovermode="x unified",margin=dict(l=20,r=20,t=25,b=20),legend=dict(orientation="h",y=1.08),xaxis_title="Date",yaxis_title="Units")
st.plotly_chart(fig,use_container_width=True)

st.markdown("## 🔮 7-Day TFT Forecast")
forecast_display=f[["date","tft_predicted_demand"]].copy()
forecast_display.columns=["Date","Predicted Demand (units)"]
forecast_display["Date"]=forecast_display["Date"].dt.strftime("%d %b %Y")
forecast_display["Predicted Demand (units)"]=forecast_display["Predicted Demand (units)"].round(2)
st.dataframe(forecast_display,use_container_width=True,hide_index=True)

st.markdown("## 🩸 Recent Blood-Supply Records")
recent_display=h.tail(15).copy()
recent_display["date"]=recent_display["date"].dt.strftime("%d %b %Y")
st.dataframe(recent_display,use_container_width=True,hide_index=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;padding:10px">🩸 TFT + PPO | AI Blood Inventory Optimization | Academic Project Prototype</div>',unsafe_allow_html=True)
