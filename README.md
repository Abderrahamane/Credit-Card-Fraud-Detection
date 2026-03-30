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
- `runtime.txt` - Streamlit Cloud Python runtime pin (`python-3.12`)
- `.streamlit/config.toml` - optional Streamlit UI theme config

## Python version compatibility

This project is tested with **Python 3.12 and 3.14**.

`requirements.txt` is pinned to versions that provide prebuilt Windows wheels for both versions, so `pip` does not need to compile `scikit-learn` from source.

## Local setup (Windows cmd)

```cmd
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer Python 3.12, replace `-3.14` with `-3.12`.

## Quick fix for scikit-learn install error

If installation fails after changing dependency versions, recreate the virtual environment:

```cmd
rmdir /s /q .venv
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt
```

## Verify model loading

```cmd
python smoke_test.py
```

Expected output includes model type, predicted class, and fraud probability.

## Run the Streamlit app

```cmd
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Click **New app** and select:
   - Repository: your project repo
   - Branch: `main` (or your deployment branch)
   - Main file path: `app.py`
4. Keep `requirements.txt` in the repo root so dependencies install automatically.
5. Keep `runtime.txt` in the repo root to pin the cloud Python version.
6. Click **Deploy**.

If deployment fails, open the app logs in Streamlit Cloud and check dependency or model-path errors first.

## Batch CSV scoring

In the app, switch to **Batch CSV** mode and upload a `.csv` file.

- Required columns: `V1` to `V28` and `scaled_amount`
- Extra columns are preserved in output
- Output adds: `predicted_class`, `fraud_probability`, and `is_fraud_at_threshold`

## Notes and limitations

- The model expects PCA-based inputs (`V1` to `V28`) and `scaled_amount`.
- Real production deployment should include monitoring, drift checks, and periodic retraining.
- Use threshold tuning based on your business cost trade-off between false positives and false negatives.

