import pathlib

import pandas as pd

from preprocessing import DEFAULT_FEATURE_ORDER, preprocess_for_inference

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"


def main():
    model = pd.read_pickle(MODEL_PATH)
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Amount"] = 100.0
    row["Time"] = 10.0

    required = list(getattr(model, "feature_names_in_", DEFAULT_FEATURE_ORDER))
    x = preprocess_for_inference(pd.DataFrame([row]), required_features=required, scaler_params=None)
    pred = int(model.predict(x)[0])

    print("Smoke test passed")
    print("Input shape:", x.shape)
    print("Prediction:", pred)


if __name__ == "__main__":
    main()

