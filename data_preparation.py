import pandas as pd
import torch
import numpy as np

def prepare_dataset(
  path: str,
  split_ratio: float = 0.8,
  seasonal_features: bool = False,
  return_group_ids: bool = False
) -> tuple:

    df = pd.read_csv(path)

    # standardize all human/natural labels to be the same
    df["cause_human_or_natural"] = df["cause_human_or_natural"].astype(str).str.strip().str.title()
    df = df[df["cause_human_or_natural"].isin(["Human", "Natural"])]

    # drop all rows that have nan values in 'lat', 'lon', 'year', 'month', 'day', 'poly_area_ha' as they will be used as features
    # and for graph connectivity
    df = df.dropna(subset=['lat', 'lon', 'year', 'month', 'day', 'poly_area_ha'])

    # Sort by date to define a clear "past" and "future"
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Extract raw values
    y = torch.tensor(df["cause_human_or_natural"].map({"Human": 1, "Natural": 0}).values, dtype=torch.long)
    log_area = np.log1p(df["poly_area_ha"].values)
    x = torch.tensor(log_area, dtype=torch.float).unsqueeze(1)

    if seasonal_features:
        doy = df["date"].dt.dayofyear.values
        angle = 2.0 * np.pi * doy / 365.25
        sin_doy = torch.tensor(np.sin(angle), dtype=torch.float).unsqueeze(1)
        cos_doy = torch.tensor(np.cos(angle), dtype=torch.float).unsqueeze(1)
        x = torch.cat([x, sin_doy, cos_doy], dim=1)

    pos_spatial = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float)
    df["days"] = (df["date"] - df["date"].min()).dt.days
    pos_temporal = torch.tensor(df["days"].values, dtype=torch.float).unsqueeze(-1)

    # 2. Define the training boundary for statistics
    split_idx = int(len(df) * split_ratio)

    # 3. Standardize using only training stats
    mu_s = pos_spatial[:split_idx].mean(dim=0)
    sigma_s = pos_spatial[:split_idx].std(dim=0)
    pos_spatial = (pos_spatial - mu_s) / (sigma_s + 1e-8)

    mu_t = pos_temporal[:split_idx].mean(dim=0)
    sigma_t = pos_temporal[:split_idx].std(dim=0)
    pos_temporal = (pos_temporal - mu_t) / (sigma_t + 1e-8)

    pos_combined = torch.cat([pos_spatial, pos_temporal], dim=1)

    if return_group_ids:
        raw = df["mtbs_ID"].fillna("").astype(str).str.strip()
        group_ids = [g if g and g not in ("N/A", "nan", "NaN", "None") else "" for g in raw]
        return x, y, pos_combined, pos_spatial, pos_temporal, group_ids

    return x, y, pos_combined, pos_spatial, pos_temporal

def get_loss_weights(
    y: torch.Tensor
) -> tuple[float]:
    natural = (y == 0).sum().item()
    human_made = (y == 1).sum().item()
    total = y.size(0)

    natural_percent = natural / total * 100
    human_percent = 100 - natural_percent

    print(f"There are {natural} natural fires ({natural_percent:.2f} %)")
    print(f"There are {human_made} human made fires ({human_percent:.2f} %)")

    weight_human = total / (2 * human_made)
    weight_natural = total / (2 * natural)

    class_weights = torch.tensor([weight_natural, weight_human], dtype=torch.float)
    return class_weights





