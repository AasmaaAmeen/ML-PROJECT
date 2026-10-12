import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DONOR_FILE = DATA / "blood_donation.csv"
BAGS_FILE = DATA / "number of blood bags.csv"

OUT_TS = DATA / "blood_timeseries.csv"
OUT_BAGS = DATA / "blood_bags_long.csv"

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

def preprocess_donations():
    df = pd.read_csv(DONOR_FILE)

    # Keep only fields needed for forecasting. Do not use PII in the ML model.
    df["Last_Donation_Date"] = pd.to_datetime(
        df["Last_Donation_Date"], dayfirst=True, errors="coerce"
    )
    df["Blood_Group"] = df["Blood_Group"].astype(str).str.strip()

    df = df[
        df["Last_Donation_Date"].notna()
        & df["Blood_Group"].isin(BLOOD_GROUPS)
    ].copy()

    # Monthly donation count is a stable forecasting target for this dataset.
    df["date"] = df["Last_Donation_Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(["date", "Blood_Group"])
          .size()
          .reset_index(name="demand")
          .rename(columns={"Blood_Group": "blood_group"})
    )

    # Make every month x blood group combination explicit.
    start = monthly["date"].min()
    end = monthly["date"].max()
    dates = pd.date_range(start, end, freq="MS")

    grid = pd.MultiIndex.from_product(
        [dates, BLOOD_GROUPS], names=["date", "blood_group"]
    ).to_frame(index=False)

    ts = grid.merge(monthly, on=["date", "blood_group"], how="left")
    ts["demand"] = ts["demand"].fillna(0).astype(float)

    # Extra calendar features for TFT.
    ts["time_idx"] = (
        (ts["date"].dt.year - ts["date"].dt.year.min()) * 12
        + ts["date"].dt.month - ts["date"].dt.month.min()
    ).astype(int)
    ts["month"] = ts["date"].dt.month.astype(int)
    ts["year"] = ts["date"].dt.year.astype(int)

    ts.to_csv(OUT_TS, index=False)

    # The blood-bag file is kept as a separate supplementary inventory dataset.
    bags = pd.read_csv(BAGS_FILE)
    value_cols = [c for c in bags.columns if c in BLOOD_GROUPS]
    bags_long = bags.melt(
        id_vars=["Year"], value_vars=value_cols,
        var_name="blood_group", value_name="blood_bags"
    )
    bags_long["Year"] = pd.to_numeric(bags_long["Year"], errors="coerce").astype("Int64")
    bags_long["blood_bags"] = pd.to_numeric(bags_long["blood_bags"], errors="coerce").fillna(0)
    bags_long = bags_long.dropna(subset=["Year"])
    bags_long.to_csv(OUT_BAGS, index=False)

    print("Saved:", OUT_TS)
    print("Rows:", len(ts))
    print("Date range:", ts["date"].min().date(), "to", ts["date"].max().date())
    print("Blood groups:", ", ".join(BLOOD_GROUPS))
    print("Saved:", OUT_BAGS)

if __name__ == "__main__":
    preprocess_donations()
