from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"
FALLBACK_FEATURES = [f"V{i}" for i in range(1, 29)] + ["scaled_amount"]


@st.cache_resource
def load_model(model_path: Path):
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def get_feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(feature) for feature in model.feature_names_in_]
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) == len(FALLBACK_FEATURES):
        return FALLBACK_FEATURES
    raise ValueError("Model is missing expected feature metadata.")


def build_input_frame(feature_names: list[str], values: dict[str, float]) -> pd.DataFrame:
    row = {name: float(values.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def get_probability(model, model_input: pd.DataFrame) -> float | None:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(model_input)[0][1]
        return float(probability)
    return None


def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    st.title("Credit Card Fraud Detection")
    st.caption("Predict whether a transaction is likely fraudulent using the trained model.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return

    try:
        model = load_model(MODEL_PATH)
        feature_names = get_feature_names(model)
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        return

    st.sidebar.header("Prediction Settings")
    threshold = st.sidebar.slider("Fraud threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    with st.form("predict_form"):
        st.subheader("Transaction Features")
        st.write("Enter principal component values and the scaled amount.")

        feature_values: dict[str, float] = {}
        columns = st.columns(3)
        for index, feature in enumerate(feature_names):
            with columns[index % 3]:
                default_value = 0.0
                if feature == "scaled_amount":
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        value=default_value,
                        step=0.1,
                        format="%.4f",
                        help="Scaled transaction amount used during model training.",
                    )
                else:
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        value=default_value,
                        step=0.01,
                        format="%.4f",
                        help="PCA-transformed feature from the source dataset.",
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_frame = build_input_frame(feature_names, feature_values)
        prediction = int(model.predict(input_frame)[0])
        fraud_probability = get_probability(model, input_frame)

        st.divider()
        st.subheader("Prediction Result")

        if fraud_probability is not None:
            is_fraud = fraud_probability >= threshold
            st.metric("Fraud probability", f"{fraud_probability * 100:.2f}%")
        else:
            is_fraud = prediction == 1

        if is_fraud:
            st.error("Potential Fraud Detected")
        else:
            st.success("Transaction appears legitimate")

        with st.expander("Input payload"):
            st.dataframe(input_frame, use_container_width=True)


if __name__ == "__main__":
    main()


