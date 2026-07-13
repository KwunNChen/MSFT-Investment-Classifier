"""Naive baselines: the accuracy bar any model must beat (project rule 6).

Zero-intelligence prediction rules scored against the true N-day direction
labels. No training, no features. These numbers exist so that model accuracy
in later steps gets compared against honest dumb strategies, not against 50%.

Usage:
    python baselines.py [--n 5] [--ticker MSFT]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pull import fetch_raw
from labels import make_label

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def always_up(labels: pd.Series) -> pd.Series:
    """Predict 1 (up) on every day. Scores the majority-class share:
    how far pure market drift gets you with zero thought."""
    return pd.Series(1.0, index=labels.index)


def persistence(labels: pd.Series, n: int) -> pd.Series:
    """At day T, predict that the last COMPLETED n-day move continues.

    Implemented as labels.shift(n): label[T-n] describes the move from
    T-n to T, which is fully observable at T's close.

    LOOKAHEAD WARNING: the tempting labels.shift(1) would cheat. label[T-1]
    describes the move ending at day T+n-1 — still in the future at day T.
    Only windows that have fully closed by T are fair game. The first n
    labeled days have no completed prior window and stay NaN.
    """
    if n < 1:
        raise ValueError(f"horizon n must be >= 1, got {n}")
    return labels.shift(n)


def evaluate_baseline(predictions: pd.Series, labels: pd.Series) -> dict:
    """Accuracy over rows where both a prediction and a true label exist.

    Reused by later model-evaluation steps so every accuracy in this project
    is computed the same way.
    """
    has_prediction_and_label = predictions.notna() & labels.notna()
    covered_rows = int(has_prediction_and_label.sum())
    accuracy = (
        float((predictions[has_prediction_and_label] == labels[has_prediction_and_label]).mean())
        if covered_rows
        else float("nan")
    )
    return {
        "accuracy": accuracy,
        "covered_rows": covered_rows,
        "total_labeled_rows": int(labels.notna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    labels = make_label(price_data["Adj Close"], args.n)
    labeled_days = labels.dropna()

    results = {
        "always_up": evaluate_baseline(always_up(labels), labels),
        "always_down": evaluate_baseline(1.0 - always_up(labels), labels),
        "persistence": evaluate_baseline(persistence(labels, args.n), labels),
    }

    print(f"\n=== Naive baselines: {args.ticker}, horizon {args.n} trading days ===")
    print(
        f"Labeled rows: {len(labeled_days)} "
        f"({labeled_days.index.min().date()} -> {labeled_days.index.max().date()})\n"
    )
    for baseline_name, result in results.items():
        print(
            f"  {baseline_name:<12} accuracy {result['accuracy'] * 100:5.1f}%   "
            f"(coverage {result['covered_rows']}/{result['total_labeled_rows']})"
        )

    majority_share = max(labeled_days.mean(), 1 - labeled_days.mean())
    always_up_matches_percent_up = (
        abs(results["always_up"]["accuracy"] - labeled_days.mean()) < 1e-12
    )
    print(
        f"\nCross-check vs step 2: majority class share = {majority_share * 100:.1f}%, "
        f"always_up accuracy matches % up exactly: "
        f"{'OK' if always_up_matches_percent_up else 'MISMATCH'}"
    )
    print(
        f"Persistence coverage is {args.n} rows short: the first {args.n} labeled days "
        "have no completed prior window to extrapolate from."
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"baselines_{args.ticker}_n{args.n}.json"
    summary = {
        "ticker": args.ticker,
        "n": args.n,
        "date_range": [str(labeled_days.index.min().date()), str(labeled_days.index.max().date())],
        "labeled_rows": len(labeled_days),
        "baselines": results,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
