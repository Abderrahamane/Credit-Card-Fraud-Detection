# Credit Card Fraud Detection

A machine-learning project that predicts whether a credit-card transaction is fraudulent.

## Why this project

Digital payments are fast and convenient, but fraud remains a major risk. Fraud detection models help teams flag suspicious transactions early so they can reduce financial losses and protect customer trust.

This project demonstrates an end-to-end workflow:
- train a fraud classifier in `notebook.ipynb`
- persist the trained model as `model/model.pkl`
- deploy an interactive web app using Streamlit (`app.py`)

## Business impact

A reliable fraud detection pipeline can create value in several ways:
- lower direct fraud losses by catching risky transactions earlier
- reduce manual review volume by prioritizing high-risk cases
- improve customer experience by minimizing false positives and payment friction
- support risk and compliance teams with consistent scoring logic

## Project structure

- `notebook.ipynb` - training, evaluation, and model export
- `model/model.pkl` - trained XGBoost model artifact
- `app.py` - Streamlit inference app (single transaction + batch CSV)
- `smoke_test.py` - quick local model sanity check
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - optional Streamlit UI theme config

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify model loading

```bash
python smoke_test.py
```

Expected output includes model type, predicted class, and fraud probability.

## Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Batch CSV scoring

In the app, switch to **Batch CSV** mode and upload a `.csv` file.

- Required columns: `V1` to `V28` and `scaled_amount`
- Extra columns are preserved in output
- Output adds: `predicted_class`, `fraud_probability`, and `is_fraud_at_threshold`

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and click **New app**.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Ensure `requirements.txt` is detected.
5. Click **Deploy**.

## Notes and limitations

- The model expects PCA-based inputs (`V1` to `V28`) and `scaled_amount`.
- Real production deployment should include monitoring, drift checks, and periodic retraining.
- Use threshold tuning based on your business cost trade-off between false positives and false negatives.

