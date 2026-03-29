import pathlib

import pandas as pd
import streamlit as st

from preprocessing import (
    DEFAULT_FEATURE_ORDER,
    build_single_input,
    load_amount_scaler,
    preprocess_for_inference,
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"
SCALER_PARAMS_PATH = BASE_DIR / "model" / "scaler_params.json"


@st.cache_resource
def load_model(model_path: pathlib.Path):
    model = pd.read_pickle(model_path)
    return model


@st.cache_data
def get_feature_order(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return DEFAULT_FEATURE_ORDER


def render_single_prediction(model, feature_order, scaler_params):
    st.subheader("Single Transaction Prediction")
    st.caption("Enter transaction features. The app applies the same preprocessing used in your notebook.")

    amount = st.number_input("Amount", min_value=0.0, value=50.0, step=1.0)
    include_time = st.checkbox("Provide Time (optional; it will be dropped)", value=False)
    time_value = st.number_input("Time", min_value=0.0, value=0.0, step=1.0) if include_time else None

    with st.expander("V1 - V28 feature inputs", expanded=False):
        v_features = {}
        cols = st.columns(4)
        for idx in range(1, 29):
            key = f"V{idx}"
            with cols[(idx - 1) % 4]:
                v_features[key] = st.number_input(key, value=0.0, format="%.6f")

    if st.button("Predict Fraud Risk", type="primary"):
        single_row = build_single_input(v_features=v_features, amount=amount, time_value=time_value)
        processed = preprocess_for_inference(
            raw_df=single_row,
            required_features=feature_order,
            scaler_params=scaler_params,
        )

        pred = int(model.predict(processed)[0])
        proba = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.error("Prediction: Fraudulent transaction")
        else:
            st.success("Prediction: Legitimate transaction")

        if proba is not None:
            st.metric("Fraud probability", f"{proba:.4f}")

        st.dataframe(processed, use_container_width=True)


def render_batch_prediction(model, feature_order, scaler_params):
    st.subheader("Batch Prediction (CSV Upload)")
    st.caption("Upload a CSV with columns V1..V28 and Amount. Optional columns like Time or Class are handled.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    processed = preprocess_for_inference(
        raw_df=df,
        required_features=feature_order,
        scaler_params=scaler_params,
    )

    preds = model.predict(processed)
    probs = model.predict_proba(processed)[:, 1] if hasattr(model, "predict_proba") else None

    results = df.copy()
    results["prediction"] = preds
    if probs is not None:
        results["fraud_probability"] = probs

    st.write(f"Processed {len(results)} rows")
    st.dataframe(results.head(50), use_container_width=True)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_data,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    st.title("Credit Card Fraud Detection")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH)
    feature_order = get_feature_order(model)
    scaler_params = load_amount_scaler(SCALER_PARAMS_PATH)

    if scaler_params is None:
        st.warning(
            "`model/scaler_params.json` was not found. The app still runs, but Amount scaling falls back "
            "to `scaled_amount = log1p(Amount)`. For best accuracy, generate scaler params from your training data."
        )

    mode = st.radio("Choose mode", options=["Single prediction", "Batch CSV prediction"], horizontal=True)

    if mode == "Single prediction":
        render_single_prediction(model, feature_order, scaler_params)
    else:
        render_batch_prediction(model, feature_order, scaler_params)


if __name__ == "__main__":
    main()

