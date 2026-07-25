"""Feature engineering: the model's inputs, all strictly backward-looking.

Every feature at day T is computed from prices/volume through T's close and
nothing later (project rule 1). pandas rolling windows are trailing by
default (the window ENDS at row T), and momentum uses a positive shift =
the past. Both properties are verified by the truncation test in the
step-5 checks, not assumed.

V3 adds optional market-context features (S&P 500, Nasdaq, VIX). Context is
OPT-IN: without it, this module produces exactly the v1 four features, so
every existing call path is unchanged. Context values are same-session
closes joined onto MSFT's own trading dates, so a context feature at day T
knows only what the market knew when day T closed.

Usage:
    python features.py [--n 5] [--ticker MSFT] [--context]
"""

import argparse
from pathlib import Path

import pandas as pd

from data_pull import fetch_raw
from labels import make_label
from splits import time_split

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"

# Short name -> yfinance symbol. ^GSPC is the S&P 500, ^IXIC the Nasdaq
# Composite, ^VIX the CBOE volatility index.
CONTEXT_TICKERS = {"spx": "^GSPC", "ndx": "^IXIC", "vix": "^VIX"}


def load_context(years: int = 10) -> dict:
    """Load the cached context tickers as {short name: DataFrame}."""
    return {
        name: fetch_raw(symbol, years=years) for name, symbol in CONTEXT_TICKERS.items()
    }


def align_context(context_data: dict, dates: pd.Index) -> dict:
    """Reindex each context frame onto MSFT's trading dates.

    Two things this join fixes. A context ticker carrying a day MSFT did not
    trade (Yahoo hands back a phantom ^VIX quote on Memorial Day 2026) gets
    dropped. A day MSFT traded but a context ticker lacks becomes NaN, which
    later dropna() removes visibly, rather than being forward filled into a
    stale value nobody notices.
    """
    return {name: frame.reindex(dates) for name, frame in context_data.items()}


def context_coverage(context_data: dict, dates: pd.Index) -> dict:
    """How many of MSFT's trading days each context ticker actually covers."""
    return {
        name: int(frame.reindex(dates)["Adj Close"].notna().sum())
        for name, frame in context_data.items()
    }


def build_features(price_data: pd.DataFrame, n: int, context_data: dict = None) -> pd.DataFrame:
    """Trailing-window features per day. Four always, eight with context.

    MSFT only (v1):
      ma_ratio     : MA5/MA20 of Adj Close, minus 1. The moving-average
                     crossover as one scale-free number (>0: short MA on top).
      volatility   : 20-day rolling std of daily Adj Close returns.
      volume_ratio : today's volume vs its own 20-day average, minus 1.
      momentum     : return over the past n days (same n as the label horizon).

    Market context (v3, only when context_data is supplied):
      spx_momentum : S&P 500 return over the past n days. Is the whole
                     market trending, not just this stock?
      rel_strength : MSFT's n-day return minus the Nasdaq's. Is MSFT beating
                     its own sector, or only riding it? (Nasdaq rather than a
                     second S&P feature: broad-index momentum measures are
                     near-duplicates, while MSFT-minus-Nasdaq is a different
                     question.)
      vix_level    : VIX close. Left raw because it is an annualized implied
                     volatility percentage, so 20 means the same thing in
                     2016 and 2026, unlike a share price.
      vix_change   : VIX change over the past n days. Fear rising or falling.

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

    if context_data is not None:
        aligned = align_context(context_data, price_data.index)
        spx_close = aligned["spx"]["Adj Close"]
        ndx_close = aligned["ndx"]["Adj Close"]
        vix_close = aligned["vix"]["Adj Close"]

        features["spx_momentum"] = spx_close / spx_close.shift(n) - 1
        features["rel_strength"] = features["momentum"] - (ndx_close / ndx_close.shift(n) - 1)
        features["vix_level"] = vix_close
        features["vix_change"] = vix_close - vix_close.shift(n)

    return features


def build_dataset(price_data: pd.DataFrame, n: int, context_data: dict = None) -> pd.DataFrame:
    """Features + label in one table, warmup and unlabeled rows dropped."""
    dataset = build_features(price_data, n, context_data)
    dataset[f"label_up_{n}d"] = make_label(price_data["Adj Close"], n)
    return dataset.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument(
        "--context", action="store_true", help="add the V3 market-context features"
    )
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    label_column = f"label_up_{args.n}d"

    context_data = None
    if args.context:
        context_data = load_context()
        coverage = context_coverage(context_data, price_data.index)
        print(f"\nContext coverage of {args.ticker}'s {len(price_data)} trading days:")
        for name, covered in coverage.items():
            missing = len(price_data) - covered
            note = "" if missing == 0 else f"  ({missing} MSFT days missing, will be dropped)"
            print(f"  {CONTEXT_TICKERS[name]:<7} {covered} days{note}")

    dataset = build_dataset(price_data, args.n, context_data)
    rows_dropped_at_head = (price_data.index < dataset.index.min()).sum()
    rows_dropped_at_tail = (price_data.index > dataset.index.max()).sum()
    print(f"\n=== Features: {args.ticker}, n={args.n}"
          f"{', with market context' if args.context else ''} ===")
    print(f"Raw rows: {len(price_data)}  ->  usable rows: {len(dataset)}")
    print(f"  dropped at head: {rows_dropped_at_head} (rolling-window warmup)")
    print(f"  dropped at tail: {rows_dropped_at_tail} (no label yet)")

    feature_columns = [column for column in dataset.columns if column != label_column]
    print("\nFeature summary:")
    print(dataset[feature_columns].describe().loc[["mean", "std", "min", "max"]].T.to_string())

    # Feature-label relationships inspected on the TRAIN slice only: even
    # looking at test-period correlations would let test information steer
    # design decisions (rule 4 in spirit). This is inspection, not selection;
    # V3 selection happens on the validation slice in step 4.
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
    suffix = "_context" if args.context else ""
    output_path = PROCESSED_DIR / f"features_{args.ticker}_n{args.n}{suffix}.parquet"
    dataset.to_parquet(output_path)
    print(f"\nSaved {len(dataset)} rows x {len(dataset.columns)} cols -> {output_path}")


if __name__ == "__main__":
    main()
