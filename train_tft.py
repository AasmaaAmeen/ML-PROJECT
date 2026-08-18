import os
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

DATA="data/blood_timeseries.csv"
MODEL="forecasting/tft_model.ckpt"
PARAMS="forecasting/dataset_parameters.pkl"

df=pd.read_csv(DATA)
df["date"]=pd.to_datetime(df["date"])
df=df.sort_values(["blood_group","date"])

max_encoder_length=30
max_prediction_length=7

# Hold out the final 7 days of every blood group for validation.
cutoff=df.groupby("blood_group")["time_idx"].transform("max")-max_prediction_length
train_df=df[df["time_idx"] <= cutoff].copy()

training=TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="estimated_demand",
    group_ids=["blood_group"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["blood_group"],
    time_varying_known_reals=["time_idx","day_of_week","month"],
    time_varying_unknown_reals=["estimated_demand"],
    target_normalizer=GroupNormalizer(groups=["blood_group"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

loader=training.to_dataloader(train=True,batch_size=64,num_workers=0)

model=TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
)

trainer=pl.Trainer(max_epochs=10, accelerator="auto", enable_checkpointing=False, logger=False)
trainer.fit(model, train_dataloaders=loader)

os.makedirs("forecasting",exist_ok=True)
trainer.save_checkpoint(MODEL)
pd.to_pickle(training.get_parameters(),PARAMS)
print("Saved:",MODEL)
print("Saved:",PARAMS)
