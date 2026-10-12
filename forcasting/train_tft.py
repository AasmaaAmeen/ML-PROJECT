import pickle
from pathlib import Path

import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "blood_timeseries.csv"
MODEL_DIR = ROOT / "forecasting"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df["blood_group"] = df["blood_group"].astype(str)
df["time_idx"] = df["time_idx"].astype(int)
df["month"] = df["month"].astype(int)
df["year"] = df["year"].astype(int)
df["demand"] = df["demand"].astype(float)

# Forecast the final 6 months from the preceding history.
max_idx = int(df["time_idx"].max())
training_cutoff = max_idx - 6

# Minimum history required by the encoder.
max_encoder_length = min(12, max(3, training_cutoff - int(df["time_idx"].min())))
max_prediction_length = 6

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="demand",
    group_ids=["blood_group"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["blood_group"],
    time_varying_known_reals=["time_idx", "month", "year"],
    time_varying_unknown_reals=["demand"],
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

checkpoint = ModelCheckpoint(
    dirpath=str(MODEL_DIR),
    filename="tft-best",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[checkpoint, early_stop],
    log_every_n_steps=5,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=3,
)

trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

best = checkpoint.best_model_path
print("Best checkpoint:", best)

# Save dataset configuration so forecast.py can recreate the prediction dataset.
with open(MODEL_DIR / "dataset_parameters.pkl", "wb") as f:
    pickle.dump(training.get_parameters(), f)

print("TFT training completed.")
