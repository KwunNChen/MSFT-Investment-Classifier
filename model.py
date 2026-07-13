"""Model training + the project's single test-set evaluation (step 6).

Logistic regression on the four features, with explicit overfitting guards:
- capacity: 4 weights + 1 intercept learned from ~2,000 training rows
- L2 regularization: sklearn's default (penalty="l2", C=1.0), stated here
  rather than relied on silently
- the train-vs-test accuracy gap is printed as the overfitting meter
- no hyperparameter tuning against the test set (rule 4): defaults only;
  any future tuning must use a validation slice from the tail of the
  training period

The test set is evaluated ONCE. Test predictions are saved to results/ so
step 7 diagnostics read the file instead of touching the test set again.

Usage:
    python model.py [--n 5] [--test-frac 0.2] [--ticker MSFT]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines import always_up, evaluate_baseline, persistence
from data_pull import fetch_raw
from features import build_dataset
from splits import time_split

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def train_model(train_features: pd.DataFrame, train_labels: pd.Series) -> Pipeline:
    """Fit StandardScaler + LogisticRegression as one Pipeline.

    The Pipeline guarantees rule 3: the scaler's means/stds come from the
    training rows only, and test rows are later scaled with those same
    train-derived numbers — never with statistics of their own.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),  # L2, C=1.0 (defaults)
        ]
    )
    pipeline.fit(train_features, train_labels)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--test-frac", type=float, default=0.2, help="share of rows in the test set")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    dataset = build_dataset(price_data, args.n)
    label_column = f"label_up_{args.n}d"
    feature_columns = [column for column in dataset.columns if column != label_column]

    train_dates, gap_dates, test_dates = time_split(dataset.index, args.n, args.test_frac)
    train_features = dataset.loc[train_dates, feature_columns]
    train_labels = dataset.loc[train_dates, label_column]
    test_features = dataset.loc[test_dates, feature_columns]
    test_labels = dataset.loc[test_dates, label_column]

    model = train_model(train_features, train_labels)

    train_predictions = pd.Series(model.predict(train_features), index=train_dates)
    train_accuracy = evaluate_baseline(train_predictions, train_labels)["accuracy"]

    # ---- the single test-set evaluation (rule 4) ----
    test_predictions = pd.Series(model.predict(test_features), index=test_dates)
    test_up_probability = pd.Series(model.predict_proba(test_features)[:, 1], index=test_dates)
    model_test_result = evaluate_baseline(test_predictions, test_labels)

    # Baselines re-scored on the SAME test rows: the full-dataset numbers
    # don't apply to the flatter test regime.
    always_up_test_result = evaluate_baseline(always_up(test_labels), test_labels)
    persistence_all = persistence(dataset[label_column], args.n)
    persistence_test_result = evaluate_baseline(persistence_all.loc[test_dates], test_labels)

    print(f"\n=== Logistic regression: {args.ticker}, n={args.n} ===")
    print(f"Train rows: {len(train_dates)}   gap: {len(gap_dates)}   test rows: {len(test_dates)}")

    print(f"\n--- HEADLINE: out-of-sample (test) accuracy vs baselines, same {len(test_dates)} rows ---")
    print(f"  logistic regression   {model_test_result['accuracy'] * 100:5.1f}%")
    print(f"  always_up             {always_up_test_result['accuracy'] * 100:5.1f}%")
    print(f"  persistence           {persistence_test_result['accuracy'] * 100:5.1f}%")
    beats_always_up = model_test_result["accuracy"] > always_up_test_result["accuracy"]
    print(
        f"\nVerdict: the model {'BEATS' if beats_always_up else 'does NOT beat'} the "
        "always-up baseline on the test period."
    )
    print(
        "(If it's at/below the baseline, that is the expected honest result for "
        "4 weak features on daily data — a finding, not a bug. Rule 5.)"
    )

    overfit_gap = train_accuracy - model_test_result["accuracy"]
    print(f"\n--- Overfitting meter ---")
    print(f"  train accuracy {train_accuracy * 100:5.1f}%  (NOT the headline)")
    print(f"  test accuracy  {model_test_result['accuracy'] * 100:5.1f}%")
    print(f"  gap            {overfit_gap * 100:+5.1f} points  (near zero = no memorization)")
    if train_accuracy > 0.65:
        print(
            "  WARNING (rule 7): train accuracy above 65% is suspicious for this "
            "problem — check for leakage before believing anything."
        )

    coefficients = dict(
        zip(feature_columns, model.named_steps["logreg"].coef_[0].round(4).tolist())
    )
    print("\nCoefficients (on standardized features; signs diagnosed in step 7):")
    for feature_name, weight in coefficients.items():
        print(f"  {feature_name:<13} {weight:+.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"model_logreg_{args.ticker}_n{args.n}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "ticker": args.ticker,
                "n": args.n,
                "test_frac": args.test_frac,
                "train_rows": len(train_dates),
                "test_rows": len(test_dates),
                "train_accuracy": train_accuracy,
                "test_accuracy": model_test_result["accuracy"],
                "overfit_gap": overfit_gap,
                "baselines_on_test": {
                    "always_up": always_up_test_result,
                    "persistence": persistence_test_result,
                },
                "coefficients": coefficients,
            },
            indent=2,
        )
    )

    predictions_table = pd.DataFrame(
        {
            "true_label": test_labels,
            "predicted_label": test_predictions,
            "up_probability": test_up_probability,
        }
    )
    predictions_path = RESULTS_DIR / f"predictions_logreg_{args.ticker}_n{args.n}.parquet"
    predictions_table.to_parquet(predictions_path)
    print(f"\nSaved metrics    -> {metrics_path}")
    print(f"Saved test preds -> {predictions_path}  (step 7 reads this file, not the test set)")


if __name__ == "__main__":
    main()
