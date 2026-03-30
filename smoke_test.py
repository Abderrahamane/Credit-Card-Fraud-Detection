from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"
FALLBACK_FEATURES = [f"V{i}" for i in range(1, 29)] + ["scaled_amount"]


def main() -> None:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(feature) for feature in model.feature_names_in_]
    else:
        feature_names = FALLBACK_FEATURES

    payload = {name: 0.0 for name in feature_names}
    input_frame = pd.DataFrame([payload], columns=feature_names)

    predicted_class = int(model.predict(input_frame)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_frame)[0][1])

    print(f"model_type={type(model).__name__}")
    print(f"predicted_class={predicted_class}")
    if probability is not None:
        print(f"fraud_probability={probability:.6f}")


if __name__ == "__main__":
    main()

