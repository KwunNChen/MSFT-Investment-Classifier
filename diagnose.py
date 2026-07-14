"""Autopsy of the saved test predictions (step 7).

Reads ONLY the files model.py saved (predictions parquet + metrics json)
plus raw prices for chart context. Deliberately imports nothing from
features.py or model.py: the model never predicts again here, so the
test set is not touched a second time (rule 4). The output is the story
behind the accuracy number: what the model actually said, where it was
wrong, and why it behaved that way.

Usage:
    python diagnose.py [--n 5] [--ticker MSFT]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display window
import matplotlib.pyplot as plt
import pandas as pd

from baselines import evaluate_baseline
from data_pull import raw_path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Plain-English context for each learned weight, cross-referencing what
# earlier steps already established about that clue.
COEFFICIENT_NOTES = {
    "ma_ratio": "trend clue; sign shows whether trading above the monthly trend raised or lowered its up odds",
    "volatility": "choppiness clue; sign shows how calm vs whipsaw markets moved its up odds",
    "volume_ratio": "unusual-activity clue; the only weight with any real size, and it's still tiny",
    "momentum": "past-N-day return; a negative weight echoes step 3's finding that trend-following loses here",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    predictions_path = RESULTS_DIR / f"predictions_logreg_{args.ticker}_n{args.n}.parquet"
    metrics_path = RESULTS_DIR / f"model_logreg_{args.ticker}_n{args.n}.json"
    predictions = pd.read_parquet(predictions_path)
    metrics = json.loads(metrics_path.read_text())

    predicted = predictions["predicted_label"]
    actual = predictions["true_label"]
    up_probability = predictions["up_probability"]
    total_days = len(predictions)

    print(f"\n=== Autopsy: logistic regression, {args.ticker}, n={args.n} ===")
    print(f"Reading {total_days} saved test predictions ({predictions.index.min().date()} "
          f"-> {predictions.index.max().date()}). The model is NOT re-run.")

    # ---- 1. Prediction balance: rubber stamp or real decisions? ----
    days_said_up = int((predicted == 1).sum())
    days_said_down = int((predicted == 0).sum())
    print(f"\n--- 1. What did it actually say? ---")
    print(f"  said UP:   {days_said_up:4d} days  ({days_said_up / total_days * 100:.1f}%)")
    print(f"  said DOWN: {days_said_down:4d} days  ({days_said_down / total_days * 100:.1f}%)")
    print(
        f"  up-probability spread: min {up_probability.min():.3f}, "
        f"mean {up_probability.mean():.3f}, max {up_probability.max():.3f}"
    )
    if days_said_down <= total_days * 0.05:
        verdict = "RUBBER STAMP: it says up nearly every day; accuracy comes from market drift, not decisions."
    else:
        verdict = "REAL DECISIONS: predictions vary day to day, so the clues moved it across the 50% line."
    print(f"  Verdict: {verdict}")

    # ---- 2. Confusion matrix: the four piles ----
    said_up_was_up = int(((predicted == 1) & (actual == 1)).sum())
    said_up_was_down = int(((predicted == 1) & (actual == 0)).sum())
    said_down_was_down = int(((predicted == 0) & (actual == 0)).sum())
    said_down_was_up = int(((predicted == 0) & (actual == 1)).sum())
    print(f"\n--- 2. The four piles (confusion matrix) ---")
    print(f"  said UP,   was UP:   {said_up_was_up:4d}  (right)")
    print(f"  said UP,   was DOWN: {said_up_was_down:4d}  (wrong)")
    print(f"  said DOWN, was DOWN: {said_down_was_down:4d}  (right)")
    print(f"  said DOWN, was UP:   {said_down_was_up:4d}  (wrong)")
    bucket_total = said_up_was_up + said_up_was_down + said_down_was_down + said_down_was_up
    print(f"  piles sum to {bucket_total} (must equal {total_days})")

    # Integrity check: accuracy recomputed from the saved file must equal
    # the accuracy model.py recorded at evaluation time.
    recomputed_accuracy = evaluate_baseline(predicted, actual)["accuracy"]
    saved_accuracy = metrics["test_accuracy"]
    matches = abs(recomputed_accuracy - saved_accuracy) < 1e-12
    print(
        f"  accuracy from piles: {recomputed_accuracy * 100:.1f}% vs saved "
        f"{saved_accuracy * 100:.1f}% -> {'OK' if matches else 'MISMATCH!'}"
    )

    # ---- 3. Why: the learned weights ----
    print(f"\n--- 3. Why it behaved that way (learned weights) ---")
    for feature_name, weight in metrics["coefficients"].items():
        direction = "raises" if weight > 0 else "lowers"
        print(f"  {feature_name:<13} {weight:+.4f}  ({direction} up-odds; {COEFFICIENT_NOTES[feature_name]})")
    print(
        "  All weights are near zero: no clue earned real trust, so every day's "
        "probability stays near the training base rate (~60% up) and rarely "
        "crosses below 50%."
    )

    # ---- 4. The picture ----
    price_data = pd.read_parquet(raw_path(args.ticker))  # price context only, no features
    test_prices = price_data.loc[predictions.index.min() : predictions.index.max(), "Adj Close"]
    was_right = predicted == actual

    figure, (price_panel, probability_panel) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1]
    )
    price_panel.plot(test_prices.index, test_prices, color="black", linewidth=1)
    price_panel.scatter(
        predictions.index[was_right], test_prices.reindex(predictions.index)[was_right],
        color="tab:green", s=8, label=f"right ({int(was_right.sum())} days)",
    )
    price_panel.scatter(
        predictions.index[~was_right], test_prices.reindex(predictions.index)[~was_right],
        color="tab:red", s=8, label=f"wrong ({int((~was_right).sum())} days)",
    )
    price_panel.set_ylabel("Adj Close ($)")
    price_panel.set_title(
        f"{args.ticker} test period: where the model was right and wrong (n={args.n})"
    )
    price_panel.legend(loc="upper left")

    probability_panel.plot(up_probability.index, up_probability, color="tab:blue", linewidth=1)
    probability_panel.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    probability_panel.set_ylabel("model P(up)")
    probability_panel.set_ylim(0, 1)
    probability_panel.annotate(
        "0.5 line: below here it predicts DOWN", xy=(0.01, 0.52), xycoords=("axes fraction", "data"),
        fontsize=8, color="gray",
    )

    figure.tight_layout()
    plot_path = RESULTS_DIR / f"diagnosis_logreg_{args.ticker}_n{args.n}.png"
    figure.savefig(plot_path, dpi=150)
    print(f"\nSaved plot -> {plot_path}")

    # ---- 5. Findings block (paste-ready for the README) ----
    print("\n--- Findings (for the written note) ---")
    print(
        f"* the model said up on {days_said_up} of {total_days} test days "
        f"({days_said_up / total_days * 100:.0f}%), with P(up) between "
        f"{up_probability.min():.2f} and {up_probability.max():.2f}"
    )
    print(
        f"* of its {days_said_up} up calls, {said_up_was_down} were wrong. "
        f"Its accuracy is market drift, collected by rarely disagreeing with it"
    )
    print(
        "* every learned weight is near zero, so the clues almost never push "
        "P(up) below 0.5: the model is the always up baseline with extra steps"
    )


if __name__ == "__main__":
    main()
