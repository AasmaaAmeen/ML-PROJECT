import os
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

DATA="data/blood_timeseries.csv"
MODEL="forecasting/tft_model.ckpt"
PARAMS="forecasting/dataset_parameters.pkl"
OUT="data/tft_forecast.csv"

if not os.path.exists(MODEL) or not os.path.exists(PARAMS):
    raise FileNotFoundError("Train TFT first: python -m forecasting.train_tft")

df=pd.read_csv(DATA)
df["date"]=pd.to_datetime(df["date"])
df=df.sort_values(["blood_group","date"])
params=pd.read_pickle(PARAMS)

# Build a prediction frame containing the final encoder window for each group.
frames=[]
for group,g in df.groupby("blood_group"):
    g=g.sort_values("time_idx").copy()
    enc=g.tail(params["max_encoder_length"]).copy()
    last_idx=int(g["time_idx"].max())
    last_date=g["date"].max()
    future=pd.DataFrame({
        "date":pd.date_range(last_date+pd.Timedelta(days=1),periods=params["max_prediction_length"],freq="D"),
        "blood_group":group,
        "donation_supply":0.0,
        "day_of_week":pd.date_range(last_date+pd.Timedelta(days=1),periods=params["max_prediction_length"],freq="D").dayofweek,
        "month":pd.date_range(last_date+pd.Timedelta(days=1),periods=params["max_prediction_length"],freq="D").month,
        "time_idx":np.arange(last_idx+1,last_idx+1+params["max_prediction_length"]),
        "estimated_demand":np.nan
    })
    frames.append(pd.concat([enc,future],ignore_index=True))
pred_df=pd.concat(frames,ignore_index=True)

dataset=TimeSeriesDataSet.from_parameters(params,pred_df,predict=True,stop_randomization=True)
loader=dataset.to_dataloader(train=False,batch_size=64,num_workers=0)
model=TemporalFusionTransformer.load_from_checkpoint(MODEL)

pred=model.predict(loader,mode="prediction")
pred=np.asarray(pred)

rows=[]
groups=sorted(df["blood_group"].unique())
for i,group in enumerate(groups):
    vals=pred[i]
    if vals.ndim>1: vals=vals[:,0]
    dates=pd.date_range(df[df["blood_group"]==group]["date"].max()+pd.Timedelta(days=1),periods=len(vals),freq="D")
    for d,v in zip(dates,vals):
        rows.append([d,group,max(0,float(v))])
out=pd.DataFrame(rows,columns=["date","blood_group","tft_predicted_demand"])
out.to_csv(OUT,index=False)
print("Saved:",OUT)
print(out.head())
