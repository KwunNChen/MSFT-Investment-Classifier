"""Target construction: N-trading-day-forward direction label.

This module is the ONLY place in the pipeline allowed to look forward in
time, because it defines the thing we're predicting. Every other module
(features, splits, models) must be strictly backward-looking.

Usage:
    python labels.py [--n 5] [--ticker MSFT]
"""

import argparse

import pandas as pd

from data_pull import fetch_raw


def make_label(prices: pd.Series, n: int) -> pd.Series:
    """Binary direction label: 1 if price closes higher n trading days ahead.

    label[T] = 1 if prices[T+n] > prices[T] else 0. Exact ties count as 0
    (down/flat) — with float prices this is effectively never hit.

    `prices` should be the dividend/split-adjusted close (Adj Close): using
    the raw Close would inject a fake ~0.2% down-move at every ex-dividend
    date. The last n rows have no future price and come back as NaN — the
    caller drops them explicitly so the row loss stays visible.
    """
    if n < 1:
        raise ValueError(f"horizon n must be >= 1, got {n}")
    future_price = prices.shift(-n)
    label = (future_price > prices).astype(float)
    label[future_price.isna()] = float("nan")
    return label.rename(f"label_up_{n}d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    labels = make_label(price_data["Adj Close"], args.n)

    unlabelable_rows = labels.isna().sum()
    labeled_days = labels.dropna()

    print(f"\n=== Label: {args.ticker} direction {args.n} trading days ahead ===")
    print(f"Total rows:   {len(price_data)}")
    print(f"Unlabelable:  {unlabelable_rows} (last {args.n} rows, no future price yet)")
    print(f"Usable rows:  {len(labeled_days)}")

    label_counts = labeled_days.value_counts().sort_index()
    down_days, up_days = int(label_counts.get(0.0, 0)), int(label_counts.get(1.0, 0))
    percent_up = up_days / len(labeled_days) * 100
    print(f"\nClass distribution:")
    print(f"  up   (1): {up_days:5d}  ({percent_up:.1f}%)")
    print(f"  down (0): {down_days:5d}  ({100 - percent_up:.1f}%)")
    print(
        f"\nMajority class share: {max(percent_up, 100 - percent_up):.1f}% "
        f"-> the bar any model must beat (this is the 'always "
        f"{'up' if percent_up >= 50 else 'down'}' baseline, formalized in step 3)."
    )

    print("\n% up by calendar year (class balance drifts with market regime):")
    stats_by_year = labeled_days.groupby(labeled_days.index.year).agg(["mean", "count"])
    for year, year_stats in stats_by_year.iterrows():
        histogram_bar = "#" * int(round(year_stats["mean"] * 30))
        print(
            f"  {year}: {year_stats['mean'] * 100:5.1f}% up  "
            f"({int(year_stats['count'])} rows)  {histogram_bar}"
        )


if __name__ == "__main__":
    main()
