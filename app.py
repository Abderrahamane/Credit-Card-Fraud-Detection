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


def validate_batch_input(batch_df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    missing_columns = [feature for feature in feature_names if feature not in batch_df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return None, errors

    model_input = batch_df[feature_names].copy()
    for feature in feature_names:
        model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce")

    invalid_rows = model_input.isna().any(axis=1)
    if invalid_rows.any():
        errors.append(
            f"Found non-numeric or missing values in required columns at rows: "
            f"{', '.join(str(index + 1) for index in model_input[invalid_rows].index[:10])}"
        )
        return None, errors

    return model_input, errors


def score_batch(model, raw_df: pd.DataFrame, model_input: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output_df = raw_df.copy()
    predictions = model.predict(model_input)
    output_df["predicted_class"] = predictions.astype(int)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[:, 1]
        output_df["fraud_probability"] = probabilities
        output_df["is_fraud_at_threshold"] = probabilities >= threshold
    else:
        output_df["is_fraud_at_threshold"] = output_df["predicted_class"] == 1

    return output_df


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

    mode = st.radio("Prediction mode", options=["Single Transaction", "Batch CSV"], horizontal=True)

    if mode == "Single Transaction":
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
    else:
        st.subheader("Batch CSV Scoring")
        st.write("Upload a CSV that includes all required model features.")
        with st.expander("Required columns"):
            st.code(", ".join(feature_names))

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read CSV file: {exc}")
                return

            if batch_df.empty:
                st.error("Uploaded CSV is empty.")
                return

            extra_columns = [column for column in batch_df.columns if column not in feature_names]
            if extra_columns:
                st.info("Extra columns detected and kept in output: " + ", ".join(extra_columns))

            model_input, errors = validate_batch_input(batch_df, feature_names)
            if errors:
                for error in errors:
                    st.error(error)
                return

            scored_df = score_batch(model, batch_df, model_input, threshold)
            st.success(f"Scored {len(scored_df)} transactions.")
            st.dataframe(scored_df.head(50), use_container_width=True)

            csv_output = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()


