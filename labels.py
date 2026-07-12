"""Target construction: N-trading-day-forward direction label.

This module is the ONLY place in the pipeline allowed to look forward in
time, because it defines the thing we're predicting. Every other module
(features, splits, models) must be strictly backward-looking.

Usage:
    python labels.py [--n 5] [--ticker MSFT]
"""
from __future__ import annotations

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
    future = prices.shift(-n)
    label = (future > prices).astype(float)
    label[future.isna()] = float("nan")
    return label.rename(f"label_up_{n}d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    df = fetch_raw(args.ticker, years=10)
    label = make_label(df["Adj Close"], args.n)

    dropped = label.isna().sum()
    labeled = label.dropna()

    print(f"\n=== Label: {args.ticker} direction {args.n} trading days ahead ===")
    print(f"Total rows:   {len(df)}")
    print(f"Unlabelable:  {dropped} (last {args.n} rows, no future price yet)")
    print(f"Usable rows:  {len(labeled)}")

    counts = labeled.value_counts().sort_index()
    n_down, n_up = int(counts.get(0.0, 0)), int(counts.get(1.0, 0))
    pct_up = n_up / len(labeled) * 100
    print(f"\nClass distribution:")
    print(f"  up   (1): {n_up:5d}  ({pct_up:.1f}%)")
    print(f"  down (0): {n_down:5d}  ({100 - pct_up:.1f}%)")
    print(
        f"\nMajority class share: {max(pct_up, 100 - pct_up):.1f}% "
        f"-> the bar any model must beat (this is the 'always "
        f"{'up' if pct_up >= 50 else 'down'}' baseline, formalized in step 3)."
    )

    print("\n% up by calendar year (class balance drifts with market regime):")
    by_year = labeled.groupby(labeled.index.year).agg(["mean", "count"])
    for year, row in by_year.iterrows():
        bar = "#" * int(round(row["mean"] * 30))
        print(f"  {year}: {row['mean'] * 100:5.1f}% up  ({int(row['count'])} rows)  {bar}")


if __name__ == "__main__":
    main()
