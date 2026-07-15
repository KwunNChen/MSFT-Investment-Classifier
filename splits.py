"""Time-based train/test split with an N-row purge gap (project rule 2).

The split is chronological — train on older rows, test on strictly newer
rows — never shuffled. On top of that, the n rows immediately before the
test period belong to NEITHER set. Reason: label[T] is computed from the
price n trading days after T, so a training row within n rows of the
boundary has a label that already encodes prices from inside the test
period. Chronology alone does not stop that leak; the gap does. The gap
must always equal the current horizon n.

Usage:
    python splits.py [--n 5] [--test-frac 0.2] [--ticker MSFT]
"""

import argparse

import pandas as pd

from data_pull import fetch_raw
from labels import make_label


def time_split(dates: pd.Index, n: int, test_frac: float = 0.2) -> tuple:
    """Split a chronologically sorted date index into (train, gap, test).

    test = the final `test_frac` share of rows; gap = the n rows right
    before the test period (excluded from training); train = everything
    earlier. The three pieces are disjoint and cover the whole index.
    """
    if n < 1:
        raise ValueError(f"gap must equal the horizon n >= 1, got {n}")
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be between 0 and 1, got {test_frac}")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted oldest -> newest before splitting")

    test_row_count = int(round(len(dates) * test_frac))
    test_start_position = len(dates) - test_row_count
    train_end_position = test_start_position - n
    if train_end_position < 1:
        raise ValueError(
            f"not enough rows ({len(dates)}) for test_frac={test_frac} plus a {n}-row gap"
        )

    train_dates = dates[:train_end_position]
    gap_dates = dates[train_end_position:test_start_position]
    test_dates = dates[test_start_position:]
    return train_dates, gap_dates, test_dates


def walk_forward_windows(
    dates: pd.Index, n: int, min_train_rows: int = 750, test_rows: int = 250
) -> list:
    """Build (train_dates, test_dates) pairs for walk-forward validation.

    Each window trains on EVERYTHING from the start of the data up to its
    cutoff (expanding train, like real deployment: retrain on all history
    available at that point), skips a gap of exactly n rows (same
    contamination logic as time_split), then tests on the next `test_rows`
    rows. Windows slide forward by `test_rows`, so test chunks never
    overlap and every era after the warmup gets exactly one turn as the
    exam. A final short chunk is kept only if it has at least half of
    `test_rows`.
    """
    if n < 1:
        raise ValueError(f"gap must equal the horizon n >= 1, got {n}")
    if min_train_rows < 1 or test_rows < 1:
        raise ValueError("min_train_rows and test_rows must be positive")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted oldest -> newest before splitting")

    windows = []
    test_start_position = min_train_rows + n
    while test_start_position < len(dates):
        test_end_position = min(test_start_position + test_rows, len(dates))
        if test_end_position - test_start_position < test_rows / 2:
            break  # leftover chunk too small to be a fair exam
        train_end_position = test_start_position - n
        windows.append(
            (dates[:train_end_position], dates[test_start_position:test_end_position])
        )
        test_start_position += test_rows

    if not windows:
        raise ValueError(
            f"not enough rows ({len(dates)}) for min_train_rows={min_train_rows} "
            f"plus a {n}-row gap and a test chunk"
        )
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days (= gap size)")
    parser.add_argument("--test-frac", type=float, default=0.2, help="share of rows in the test set")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, years=10)
    labels = make_label(price_data["Adj Close"], args.n).dropna()

    train_dates, gap_dates, test_dates = time_split(labels.index, args.n, args.test_frac)

    print(f"\n=== Time split: {args.ticker}, n={args.n}, test fraction {args.test_frac} ===")
    print(f"Labeled rows: {len(labels)}\n")
    for piece_name, piece_dates in [("train", train_dates), ("gap", gap_dates), ("test", test_dates)]:
        note = "  (dropped from training; size = horizon n)" if piece_name == "gap" else ""
        print(
            f"  {piece_name:<5} {len(piece_dates):5d} rows   "
            f"{piece_dates.min().date()} -> {piece_dates.max().date()}{note}"
        )

    # Leak check: the label of the LAST training row looks n rows forward,
    # landing on the last gap day — strictly before the first test day.
    last_train_position = len(train_dates) - 1
    label_window_end = labels.index[last_train_position + args.n]
    first_test_day = test_dates.min()
    no_leak = label_window_end < first_test_day
    print(
        f"\nLeak check: last train label looks at the price on {label_window_end.date()}, "
        f"first test day is {first_test_day.date()} -> {'OK, no overlap' if no_leak else 'LEAK!'}"
    )

    print(
        f"\nClass balance ('% up'):  train {labels[train_dates].mean() * 100:.1f}%   "
        f"test {labels[test_dates].mean() * 100:.1f}%"
    )
    print("(A train/test balance mismatch is a regime shift, not a bug — see step 2 notes.)")


if __name__ == "__main__":
    main()
