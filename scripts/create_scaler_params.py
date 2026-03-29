import argparse
import json
import pathlib

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create Amount scaling parameters from creditcard.csv used during training."
    )
    parser.add_argument("--dataset", required=True, help="Path to creditcard.csv")
    parser.add_argument(
        "--output",
        default="model/scaler_params.json",
        help="Output JSON path (default: model/scaler_params.json)",
    )
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    output_path = pathlib.Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "Amount" not in df.columns:
        raise ValueError("Dataset must contain an 'Amount' column")

    log_amount = np.log1p(df["Amount"].clip(lower=0.0))
    stats = {
        "log_amount_mean": float(log_amount.mean()),
        "log_amount_std": float(log_amount.std(ddof=0)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved scaler params to: {output_path}")
    print(stats)


if __name__ == "__main__":
    main()

