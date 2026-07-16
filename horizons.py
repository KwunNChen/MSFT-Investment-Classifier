"""Multi-horizon comparison: the unchanged pipeline rerun at several N.

The payoff for never hardcoding N. For each horizon this rebuilds the
labels and the N-dependent momentum feature, then runs the exact same
walk-forward code path as backtest.py (fresh model per window, gap = n,
one exam each). Nothing else changes between horizons.

The bar moves with the horizon (longer windows drift up more often), so
raw accuracies across horizons are NOT comparable. The only fair read is
each model against its own same-rows always_up, per window.

Usage:
    python horizons.py [--horizons 1 5 21 63] [--ticker MSFT]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display window
import matplotlib.pyplot as plt

from backtest import run_walk_forward
from data_pull import fetch_raw
from features import build_dataset
from model import MODEL_NAMES

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def record_vs_bar(window_rows: list, model_kind: str) -> dict:
    """Wins/ties/losses vs always_up across windows, plus accuracy range."""
    wins = sum(1 for row in window_rows if row[model_kind] > row["always_up"] + 1e-9)
    ties = sum(1 for row in window_rows if abs(row[model_kind] - row["always_up"]) <= 1e-9)
    accuracies = [row[model_kind] for row in window_rows]
    return {
        "wins": wins,
        "ties": ties,
        "losses": len(window_rows) - wins - ties,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21, 63])
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()
    model_kinds = ["logreg", "rf"]

    price_data = fetch_raw(args.ticker, years=10)

    results_by_horizon = {}
    for n in args.horizons:
        dataset = build_dataset(price_data, n)
        label_column = f"label_up_{n}d"
        feature_columns = [column for column in dataset.columns if column != label_column]
        window_rows = run_walk_forward(dataset, label_column, feature_columns, model_kinds, n)
        results_by_horizon[n] = {
            "labeled_rows": len(dataset),
            "share_up": float(dataset[label_column].mean()),
            "windows": window_rows,
        }
        print(f"[done] n={n}: {len(dataset)} rows, {len(window_rows)} walk-forward windows")

    print(f"\n=== Multi-horizon comparison: {args.ticker} ===")
    print("Each model judged ONLY against always_up on the same windows (the bar")
    print("rises with the horizon, so raw accuracies are not comparable across N).\n")
    header = f"  {'N':>3} {'rows':>6} {'% up':>6}"
    for kind in model_kinds:
        header += f"   {MODEL_NAMES[kind]}: W-T-L (acc range)"
    print(header)
    for n, block in results_by_horizon.items():
        line = f"  {n:>3} {block['labeled_rows']:>6} {block['share_up'] * 100:5.1f}%"
        for kind in model_kinds:
            record = record_vs_bar(block["windows"], kind)
            line += (
                f"   {record['wins']}-{record['ties']}-{record['losses']} "
                f"({record['min_accuracy'] * 100:.1f}% to {record['max_accuracy'] * 100:.1f}%)"
            )
        print(line)

    total_wins = sum(
        record_vs_bar(block["windows"], kind)["wins"]
        for block in results_by_horizon.values()
        for kind in model_kinds
    )
    total_fights = sum(
        len(block["windows"]) * len(model_kinds) for block in results_by_horizon.values()
    )
    print(
        f"\nTotal model-vs-bar fights across all horizons: {total_fights}, "
        f"wins: {total_wins}."
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"horizons_{args.ticker}.json"
    json_path.write_text(
        json.dumps(
            {
                "ticker": args.ticker,
                "models": model_kinds,
                "horizons": {str(n): block for n, block in results_by_horizon.items()},
            },
            indent=2,
        )
    )

    # 2x2 grid, one panel per horizon, same style as the step 9 chart.
    panel_count = len(results_by_horizon)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    colors = {"logreg": "tab:blue", "rf": "tab:orange"}
    for axis, (n, block) in zip(axes.flat, results_by_horizon.items()):
        window_rows = block["windows"]
        window_labels = [row["test_start"][:4] for row in window_rows]
        axis.plot(window_labels, [row["always_up"] * 100 for row in window_rows],
                  color="black", linestyle="--", marker="o", label="always up (the bar)")
        for kind in model_kinds:
            axis.plot(window_labels, [row[kind] * 100 for row in window_rows],
                      color=colors[kind], marker="o", label=MODEL_NAMES[kind])
        axis.axhline(50, color="gray", linewidth=0.8, linestyle=":")
        axis.set_title(f"N = {n} trading days")
        axis.set_ylabel("accuracy (%)")
    for axis in axes.flat[panel_count:]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(f"{args.ticker} walk-forward across horizons: does any N ever beat the bar?", y=0.99)
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    plot_path = RESULTS_DIR / f"horizons_{args.ticker}.png"
    figure.savefig(plot_path, dpi=150)

    print(f"\nSaved numbers -> {json_path}")
    print(f"Saved chart   -> {plot_path}")


if __name__ == "__main__":
    main()
