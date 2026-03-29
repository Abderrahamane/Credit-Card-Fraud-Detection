# Credit Card Fraud Detection - Streamlit Deployment

This project deploys your trained fraud model (`model/fraud_model.pkl`) as a Streamlit app.

The app keeps notebook preprocessing at inference time:
- drops `Time` if present
- computes `LogAmount = log1p(Amount)`
- creates `scaled_amount`
- sends model features in exact training order (`V1`..`V28`, `scaled_amount`)

## 1) Project setup

```bash
cd C:\Users\MICROSOFT\PycharmProjects\Credit-Card-Fraud-Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) (Recommended) Save Amount scaling parameters

Your notebook used `StandardScaler` for `LogAmount`. For best prediction quality, generate and save those stats once from `creditcard.csv`.

```bash
python scripts\create_scaler_params.py --dataset C:\path\to\creditcard.csv --output model\scaler_params.json
```

This creates `model/scaler_params.json`:

```json
{
  "log_amount_mean": 0.0,
  "log_amount_std": 1.0
}
```

If this file is missing, the app still runs with a fallback (`scaled_amount = log1p(Amount)`), but accuracy may be lower.

## 3) Run locally

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## 4) Use the app

### Single prediction
1. Enter `Amount`.
2. Enter feature values `V1` to `V28`.
3. (Optional) enter `Time`; it will be dropped automatically.
4. Click **Predict Fraud Risk**.

### Batch prediction
1. Switch to **Batch CSV prediction**.
2. Upload CSV containing at least `V1`..`V28` and `Amount`.
3. Optional columns (`Time`, `Class`) are handled automatically.
4. Download prediction results as CSV.

## 5) Deploy to Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app**.
4. Select your repo and branch.
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.

## 6) Files added for deployment

- `app.py`: Streamlit UI + inference flow
- `preprocessing.py`: notebook-aligned preprocessing for inference
- `scripts/create_scaler_params.py`: helper to export scaling params from dataset
- `smoke_test.py`: quick local inference sanity test
- `requirements.txt`: runtime dependencies

## 7) Quick sanity test

```bash
python smoke_test.py
```

You should see `Smoke test passed` in terminal.

