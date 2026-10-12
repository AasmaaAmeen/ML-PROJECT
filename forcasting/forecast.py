import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "blood_timeseries.csv"
MODEL_FILE = ROOT / "forecasting" / "tft-best.ckpt"
PARAM_FILE = ROOT / "forecasting" / "dataset_parameters.pkl"
OUT_FILE = ROOT / "data" / "tft_forecast.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

with open(PARAM_FILE, "rb") as f:
    params = pickle.load(f)

dataset = TimeSeriesDataSet.from_parameters(
    params, df, predict=True, stop_randomization=True
)
loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

model = TemporalFusionTransformer.load_from_checkpoint(MODEL_FILE)
pred = model.predict(loader, mode="prediction")

# pred is normally [series, prediction_length].
pred = np.asarray(pred)
if pred.ndim == 3:
    pred = pred[..., 0]

groups = sorted(df["blood_group"].unique())
last_date = df["date"].max()

rows = []
for i, group in enumerate(groups):
    values = np.asarray(pred[i]).reshape(-1)
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=len(values), freq="MS"
    )
    for d, y in zip(future_dates, values):
        rows.append({
            "date": d,
            "blood_group": group,
            "tft_predicted_demand": max(0.0, float(y))
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print(out.to_string(index=False))
