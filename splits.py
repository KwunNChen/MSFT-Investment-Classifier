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


def time_split(index: pd.Index, n: int, test_frac: float = 0.2) -> tuple:
    """Split a chronologically sorted index into (train, gap, test) indexes.

    test = the final `test_frac` share of rows; gap = the n rows right
    before the test period (excluded from training); train = everything
    earlier. The three pieces are disjoint and cover the whole index.
    """
    if n < 1:
        raise ValueError(f"gap must equal the horizon n >= 1, got {n}")
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be between 0 and 1, got {test_frac}")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted oldest -> newest before splitting")

    test_rows = int(round(len(index) * test_frac))
    test_start = len(index) - test_rows
    train_end = test_start - n
    if train_end < 1:
        raise ValueError(
            f"not enough rows ({len(index)}) for test_frac={test_frac} plus a {n}-row gap"
        )

    return index[:train_end], index[train_end:test_start], index[test_start:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="horizon in trading days (= gap size)")
    parser.add_argument("--test-frac", type=float, default=0.2, help="share of rows in the test set")
    parser.add_argument("--ticker", default="MSFT")
    args = parser.parse_args()

    df = fetch_raw(args.ticker, years=10)
    labels = make_label(df["Adj Close"], args.n).dropna()

    train_idx, gap_idx, test_idx = time_split(labels.index, args.n, args.test_frac)

    print(f"\n=== Time split: {args.ticker}, n={args.n}, test fraction {args.test_frac} ===")
    print(f"Labeled rows: {len(labels)}\n")
    for name, idx in [("train", train_idx), ("gap", gap_idx), ("test", test_idx)]:
        note = "  (dropped from training; size = horizon n)" if name == "gap" else ""
        print(f"  {name:<5} {len(idx):5d} rows   {idx.min().date()} -> {idx.max().date()}{note}")

    # Leak check: the label of the LAST training row looks n rows forward,
    # landing on the last gap day — strictly before the first test day.
    last_train_pos = len(train_idx) - 1
    label_window_end = labels.index[last_train_pos + args.n]
    first_test_day = test_idx.min()
    ok = label_window_end < first_test_day
    print(
        f"\nLeak check: last train label looks at the price on {label_window_end.date()}, "
        f"first test day is {first_test_day.date()} -> {'OK, no overlap' if ok else 'LEAK!'}"
    )

    print(
        f"\nClass balance ('% up'):  train {labels[train_idx].mean() * 100:.1f}%   "
        f"test {labels[test_idx].mean() * 100:.1f}%"
    )
    print("(A train/test balance mismatch is a regime shift, not a bug — see step 2 notes.)")


if __name__ == "__main__":
    main()
