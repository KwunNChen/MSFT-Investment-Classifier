"""Feature engineering: the model's inputs, all strictly backward-looking.

Every feature at day T is computed from prices/volume through T's close and
nothing later (project rule 1). pandas rolling windows are trailing by
default (the window ENDS at row T), and momentum uses a positive shift =
the past. Both properties are verified by the truncation test in the
step-5 checks, not assumed.

Usage:
    python features.py [--n 5] [--ticker MSFT]
"""

import argparse
from pathlib import Path

import pandas as pd

from data_pull import fetch_raw
from labels import make_label
from splits import time_split

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"


def build_features(price_data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Four trailing-window features per day.

    ma_ratio     : MA5/MA20 of Adj Close, minus 1. The moving-average
                   crossover as one scale-free number (>0: short MA on top).
    volatility   : 20-day rolling std of daily Adj Close returns.
    volume_ratio : today's volume vs its own 20-day average, minus 1.
    momentum     : return over the past n days (same n as the label horizon).

    The first ~20 rows are NaN while the rolling windows fill up (warmup);
    callers drop them explicitly.
    """
    if n < 1:
        raise ValueError(f"horizon n must be >= 1, got {n}")
    adj_close = price_data["Adj Close"]
    daily_returns = adj_close.pct_change()
    volume = price_data["Volume"]

    features = pd.DataFrame(index=price_data.index)
    features["ma_ratio"] = adj_close.rolling(5).mean() / adj_close.rolling(20).mean() - 1
    features["volatility"] = daily_returns.rolling(20).std()
    features["volume_ratio"] = volume / volume.rolling(20).mean() - 1
    features["momentum"] = adj_close / adj_close.shift(n) - 1
    return features


def build_dataset(price_data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Features + label in one table, warmup and unlabeled rows dropped."""
    dataset = build_features(price_data, n)
    dataset[f"label_up_{n}d"] = make_label(price_data["Adj Close"], n)
    return dataset.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    label_column = f"label_up_{args.n}d"

    dataset = build_dataset(price_data, args.n)
    rows_dropped_at_head = (price_data.index < dataset.index.min()).sum()
    rows_dropped_at_tail = (price_data.index > dataset.index.max()).sum()
    print(f"\n=== Features: {args.ticker}, n={args.n} ===")
    print(f"Raw rows: {len(price_data)}  ->  usable rows: {len(dataset)}")
    print(f"  dropped at head: {rows_dropped_at_head} (rolling-window warmup)")
    print(f"  dropped at tail: {rows_dropped_at_tail} (no label yet)")

    feature_columns = [column for column in dataset.columns if column != label_column]
    print("\nFeature summary:")
    print(dataset[feature_columns].describe().loc[["mean", "std", "min", "max"]].T.to_string())

    # Feature-label relationships inspected on the TRAIN slice only: even
    # looking at test-period correlations would let test information steer
    # design decisions (rule 4 in spirit).
    train_dates, _, _ = time_split(dataset.index, args.n)
    train_rows = dataset.loc[train_dates]
    print(f"\nCorrelation with {label_column} (train slice only, {len(train_rows)} rows):")
    correlations = (
        train_rows[feature_columns]
        .corrwith(train_rows[label_column])
        .sort_values(key=abs, ascending=False)
    )
    print(correlations.to_string(float_format=lambda value: f"{value:+.3f}"))
    print("(Near zero is the expected honest result; anything large would be suspicious.)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"features_{args.ticker}_n{args.n}.parquet"
    dataset.to_parquet(output_path)
    print(f"\nSaved {len(dataset)} rows x {len(dataset.columns)} cols -> {output_path}")


if __name__ == "__main__":
    main()
