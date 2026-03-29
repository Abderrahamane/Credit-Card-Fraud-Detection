import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_FEATURE_ORDER = [f"V{i}" for i in range(1, 29)] + ["scaled_amount"]


class AmountScalerParams:
    def __init__(self, log_amount_mean: float, log_amount_std: float):
        self.log_amount_mean = float(log_amount_mean)
        self.log_amount_std = float(log_amount_std)


def load_amount_scaler(params_path: pathlib.Path) -> Optional[AmountScalerParams]:
    if not params_path.exists():
        return None

    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return AmountScalerParams(
        log_amount_mean=data["log_amount_mean"],
        log_amount_std=data["log_amount_std"],
    )


def build_single_input(v_features: dict, amount: float, time_value: Optional[float] = None) -> pd.DataFrame:
    row = {f"V{i}": float(v_features.get(f"V{i}", 0.0)) for i in range(1, 29)}
    row["Amount"] = float(amount)
    if time_value is not None:
        row["Time"] = float(time_value)
    return pd.DataFrame([row])


def preprocess_for_inference(
    raw_df: pd.DataFrame,
    required_features: list[str],
    scaler_params: Optional[AmountScalerParams],
) -> pd.DataFrame:
    df = raw_df.copy()

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    if "Amount" in df.columns:
        # Match notebook preprocessing: log transform, then standardize.
        df["LogAmount"] = np.log1p(df["Amount"].clip(lower=0.0))
        if scaler_params is None or scaler_params.log_amount_std == 0:
            # Fallback keeps app usable even without saved scaler parameters.
            df["scaled_amount"] = df["LogAmount"]
        else:
            df["scaled_amount"] = (df["LogAmount"] - scaler_params.log_amount_mean) / scaler_params.log_amount_std
        df = df.drop(columns=["Amount", "LogAmount"])

    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0.0

    # Enforce exact feature order expected by the trained model.
    df = df[required_features]
    return df

