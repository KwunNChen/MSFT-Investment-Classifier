"""Walk-forward validation: the same experiment rerun across rolling windows.

NOT a trading simulator. This is out-of-sample evaluation across eras:
each window trains a FRESH model on everything before its cutoff, skips
the n-row gap, and takes exactly one exam on the next ~year of days it
has never seen. Yesterday's test window becomes tomorrow's training data
(past -> future only), which mirrors real deployment.

Per project rules: per-window accuracy is reported, never just the
average; baselines are re-scored on every window's exact rows; nothing
is tuned between windows.

Usage:
    python backtest.py [--model both|logreg|rf] [--n 5] [--ticker MSFT]
                       [--test-rows 250] [--min-train-rows 750]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display window
import matplotlib.pyplot as plt
import pandas as pd

from baselines import always_up, evaluate_baseline, persistence
from data_pull import fetch_raw
from features import build_dataset
from model import MODEL_NAMES, train_model
from splits import walk_forward_windows

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", "logreg", "rf"], default="both")
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--test-rows", type=int, default=250, help="rows per test window (~1 year)")
    parser.add_argument("--min-train-rows", type=int, default=750, help="training rows before window 1 (~3 years)")
    args = parser.parse_args()
    model_kinds = ["logreg", "rf"] if args.model == "both" else [args.model]

    price_data = fetch_raw(args.ticker, years=10)
    dataset = build_dataset(price_data, args.n)
    label_column = f"label_up_{args.n}d"
    feature_columns = [column for column in dataset.columns if column != label_column]
    persistence_all = persistence(dataset[label_column], args.n)

    windows = walk_forward_windows(
        dataset.index, args.n, min_train_rows=args.min_train_rows, test_rows=args.test_rows
    )

    print(f"\n=== Walk-forward: {args.ticker}, n={args.n}, {len(windows)} windows ===")
    print(f"Models: {', '.join(MODEL_NAMES[kind] for kind in model_kinds)}. "
          "Each window trains a FRESH model and takes one exam it has never seen.\n")

    window_results = []
    for window_number, (train_dates, test_dates) in enumerate(windows, start=1):
        train_features = dataset.loc[train_dates, feature_columns]
        train_labels = dataset.loc[train_dates, label_column]
        test_features = dataset.loc[test_dates, feature_columns]
        test_labels = dataset.loc[test_dates, label_column]

        row = {
            "window": window_number,
            "test_start": str(test_dates.min().date()),
            "test_end": str(test_dates.max().date()),
            "train_rows": len(train_dates),
            "test_rows": len(test_dates),
            "always_up": evaluate_baseline(always_up(test_labels), test_labels)["accuracy"],
            "persistence": evaluate_baseline(persistence_all.loc[test_dates], test_labels)["accuracy"],
        }
        for kind in model_kinds:
            model = train_model(train_features, train_labels, kind)
            test_predictions = pd.Series(model.predict(test_features), index=test_dates)
            row[kind] = evaluate_baseline(test_predictions, test_labels)["accuracy"]
        window_results.append(row)

    # ---- per-window table (the deliverable: not just an average) ----
    model_headers = "".join(f"{MODEL_NAMES[kind]:>21}" for kind in model_kinds)
    print(f"  win  test period            rows{model_headers}{'always_up':>11}{'persist':>9}")
    for row in window_results:
        model_cells = "".join(f"{row[kind] * 100:20.1f}%" for kind in model_kinds)
        print(
            f"  {row['window']:>3}  {row['test_start']} -> {row['test_end']}  "
            f"{row['test_rows']:>4}{model_cells}{row['always_up'] * 100:10.1f}%"
            f"{row['persistence'] * 100:8.1f}%"
        )

    # ---- stability summary per model ----
    for kind in model_kinds:
        wins = sum(1 for row in window_results if row[kind] > row["always_up"] + 1e-9)
        ties = sum(1 for row in window_results if abs(row[kind] - row["always_up"]) <= 1e-9)
        losses = len(window_results) - wins - ties
        accuracies = [row[kind] for row in window_results]
        print(
            f"\n{MODEL_NAMES[kind]} vs always_up: {wins} wins, {ties} ties, {losses} losses "
            f"across {len(window_results)} windows"
        )
        print(
            f"  accuracy range {min(accuracies) * 100:.1f}% to {max(accuracies) * 100:.1f}% "
            "(range reported on purpose; a single average would hide the swings)"
        )
    print(
        "\nReading it honestly: losing or tying in every window is not failure, it is "
        "the v1 finding independently confirmed once per era (rule 5)."
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"walkforward_{args.ticker}_n{args.n}.json"
    json_path.write_text(
        json.dumps(
            {
                "ticker": args.ticker,
                "n": args.n,
                "test_rows_per_window": args.test_rows,
                "min_train_rows": args.min_train_rows,
                "models": model_kinds,
                "windows": window_results,
            },
            indent=2,
        )
    )

    # ---- the stability picture ----
    figure, axis = plt.subplots(figsize=(11, 5))
    window_labels = [f"{row['test_start'][:4]}\nw{row['window']}" for row in window_results]
    axis.plot(window_labels, [row["always_up"] * 100 for row in window_results],
              color="black", linestyle="--", marker="o", label="always up (the bar)")
    colors = {"logreg": "tab:blue", "rf": "tab:orange"}
    for kind in model_kinds:
        axis.plot(window_labels, [row[kind] * 100 for row in window_results],
                  color=colors[kind], marker="o", label=MODEL_NAMES[kind])
    axis.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    axis.set_ylabel("out-of-sample accuracy (%)")
    axis.set_title(f"{args.ticker} walk-forward, n={args.n}: every era takes a turn as the exam")
    axis.legend()
    figure.tight_layout()
    plot_path = RESULTS_DIR / f"walkforward_{args.ticker}_n{args.n}.png"
    figure.savefig(plot_path, dpi=150)

    print(f"\nSaved numbers -> {json_path}")
    print(f"Saved chart   -> {plot_path}")


if __name__ == "__main__":
    main()
