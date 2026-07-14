"""Model training + the project's single test-set evaluation per model.

Two models, one protocol (--model logreg or rf):
- logreg: 4 weights + 1 intercept, L2 regularized (sklearn defaults). Too
  small to memorize; the honest linear read of the clues.
- rf: RandomForestClassifier, sklearn defaults, random_state=42. CAN
  memorize the training data, which makes the train-vs-test gap a real
  overfitting meter for the first time.

Shared guards:
- the StandardScaler is fitted on training rows only (rule 3). Trees don't
  need scaling (they're scale invariant) but the scaler stays so the
  protocol is byte-for-byte identical across models.
- no hyperparameter tuning against the test set (rule 4): defaults only;
  any future tuning must use a validation slice from the tail of the
  training period.
- each model's test set evaluation happens ONCE; predictions are saved to
  results/ so diagnostics read the file instead of re-touching the test set.

Usage:
    python model.py [--model logreg|rf] [--n 5] [--test-frac 0.2] [--ticker MSFT]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines import always_up, evaluate_baseline, persistence
from data_pull import fetch_raw
from features import build_dataset
from splits import time_split

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_NAMES = {"logreg": "logistic regression", "rf": "random forest"}


def train_model(train_features: pd.DataFrame, train_labels: pd.Series, model_kind: str = "logreg") -> Pipeline:
    """Fit StandardScaler + the chosen classifier as one Pipeline.

    The Pipeline guarantees rule 3: the scaler's means/stds come from the
    training rows only, and test rows are later scaled with those same
    train-derived numbers — never with statistics of their own.
    """
    if model_kind == "logreg":
        classifier = LogisticRegression(max_iter=1000)  # L2, C=1.0 (defaults)
    elif model_kind == "rf":
        # sklearn defaults (100 trees, unlimited depth). random_state pins
        # the coin flips so reruns are reproducible; n_jobs uses all cores.
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"unknown model kind: {model_kind}")

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    pipeline.fit(train_features, train_labels)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--test-frac", type=float, default=0.2, help="share of rows in the test set")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()
    model_label = MODEL_NAMES[args.model]

    price_data = fetch_raw(args.ticker, years=10)
    dataset = build_dataset(price_data, args.n)
    label_column = f"label_up_{args.n}d"
    feature_columns = [column for column in dataset.columns if column != label_column]

    train_dates, gap_dates, test_dates = time_split(dataset.index, args.n, args.test_frac)
    train_features = dataset.loc[train_dates, feature_columns]
    train_labels = dataset.loc[train_dates, label_column]
    test_features = dataset.loc[test_dates, feature_columns]
    test_labels = dataset.loc[test_dates, label_column]

    model = train_model(train_features, train_labels, args.model)

    train_predictions = pd.Series(model.predict(train_features), index=train_dates)
    train_accuracy = evaluate_baseline(train_predictions, train_labels)["accuracy"]

    # ---- the single test-set evaluation for this model (rule 4) ----
    test_predictions = pd.Series(model.predict(test_features), index=test_dates)
    test_up_probability = pd.Series(model.predict_proba(test_features)[:, 1], index=test_dates)
    model_test_result = evaluate_baseline(test_predictions, test_labels)

    # Baselines re-scored on the SAME test rows: the full-dataset numbers
    # don't apply to the flatter test regime.
    always_up_test_result = evaluate_baseline(always_up(test_labels), test_labels)
    persistence_all = persistence(dataset[label_column], args.n)
    persistence_test_result = evaluate_baseline(persistence_all.loc[test_dates], test_labels)

    print(f"\n=== {model_label}: {args.ticker}, n={args.n} ===")
    print(f"Train rows: {len(train_dates)}   gap: {len(gap_dates)}   test rows: {len(test_dates)}")

    print(f"\n--- HEADLINE: out-of-sample (test) accuracy vs baselines, same {len(test_dates)} rows ---")
    print(f"  {model_label:<21} {model_test_result['accuracy'] * 100:5.1f}%")
    print(f"  always_up             {always_up_test_result['accuracy'] * 100:5.1f}%")
    print(f"  persistence           {persistence_test_result['accuracy'] * 100:5.1f}%")
    beats_always_up = model_test_result["accuracy"] > always_up_test_result["accuracy"]
    print(
        f"\nVerdict: the model {'BEATS' if beats_always_up else 'does NOT beat'} the "
        "always-up baseline on the test period."
    )
    if beats_always_up:
        print("(Rule 7: before believing this, the next step is a leak hunt, not a celebration.)")
    else:
        print(
            "(If it's at/below the baseline, that is the expected honest result for "
            "4 weak features on daily data — a finding, not a bug. Rule 5.)"
        )

    overfit_gap = train_accuracy - model_test_result["accuracy"]
    print(f"\n--- Overfitting meter ---")
    print(f"  train accuracy {train_accuracy * 100:5.1f}%  (NOT the headline)")
    print(f"  test accuracy  {model_test_result['accuracy'] * 100:5.1f}%")
    print(f"  gap            {overfit_gap * 100:+5.1f} points")
    if train_accuracy > 0.65:
        print(
            "  NOTE: train accuracy above 65% means the model is memorizing training "
            "data (expected for a default random forest) or something is leaking. "
            "Judge it ONLY by the test number."
        )

    # What the model leaned on: signed weights for logreg, non-negative
    # importance shares for rf (importances say how much a clue was used,
    # never which direction it pushed).
    classifier = model.named_steps["classifier"]
    if args.model == "logreg":
        learned = dict(zip(feature_columns, classifier.coef_[0].round(4).tolist()))
        learned_key, learned_title = "coefficients", "Coefficients (on standardized features)"
    else:
        learned = dict(zip(feature_columns, classifier.feature_importances_.round(4).tolist()))
        learned_key, learned_title = "feature_importances", "Feature importances (usage shares, no direction)"
    print(f"\n{learned_title}:")
    for feature_name, value in learned.items():
        print(f"  {feature_name:<13} {value:+.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"model_{args.model}_{args.ticker}_n{args.n}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model": args.model,
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
                learned_key: learned,
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
    predictions_path = RESULTS_DIR / f"predictions_{args.model}_{args.ticker}_n{args.n}.parquet"
    predictions_table.to_parquet(predictions_path)
    print(f"\nSaved metrics    -> {metrics_path}")
    print(f"Saved test preds -> {predictions_path}  (diagnostics read this file, not the test set)")

    # ---- side-by-side with the other model, if it has run ----
    other_model = "rf" if args.model == "logreg" else "logreg"
    other_metrics_path = RESULTS_DIR / f"model_{other_model}_{args.ticker}_n{args.n}.json"
    if other_metrics_path.exists():
        other_metrics = json.loads(other_metrics_path.read_text())
        print(f"\n--- Comparison: {model_label} vs {MODEL_NAMES[other_model]} (same split, same grader) ---")
        print(f"  {'':<16} {model_label:>20} {MODEL_NAMES[other_model]:>20}")
        print(
            f"  {'train accuracy':<16} {train_accuracy * 100:19.1f}% "
            f"{other_metrics['train_accuracy'] * 100:19.1f}%"
        )
        print(
            f"  {'test accuracy':<16} {model_test_result['accuracy'] * 100:19.1f}% "
            f"{other_metrics['test_accuracy'] * 100:19.1f}%"
        )
        print(
            f"  {'gap':<16} {overfit_gap * 100:+19.1f} {other_metrics['overfit_gap'] * 100:+19.1f}"
        )
        print(
            "  Reading the gaps: logreg's comes from the flatter test regime (always_up "
            "drops ~8 points between eras too); anything far beyond that is memorization."
        )


if __name__ == "__main__":
    main()
