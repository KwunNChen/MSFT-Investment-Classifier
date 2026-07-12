"""Pull and cache raw daily OHLCV data via yfinance, then sanity-check it.

Usage:
    python data_pull.py [--ticker MSFT] [--years 10] [--force]

The raw pull is cached in data/raw/ (gitignored) and never modified after
download. Downloads use auto_adjust=False so the frame keeps both the raw
`Close` and the dividend/split-adjusted `Adj Close`, making the adjustment
status explicit instead of implied by a library default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}_daily.parquet"


def fetch_raw(ticker: str, years: int, force: bool = False) -> pd.DataFrame:
    """Return ~`years` of daily OHLCV for `ticker`, cached in data/raw/."""
    path = raw_path(ticker)
    if path.exists() and not force:
        print(f"[cache] using existing {path.relative_to(RAW_DIR.parent.parent)}")
        return pd.read_parquet(path)

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    print(f"[download] {ticker} daily OHLCV {start.date()} -> {end.date()}")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yf.download returns MultiIndex columns (field, ticker) even for a
    # single ticker; flatten to plain field names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.sort_index()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"[cache] wrote {len(df)} rows -> {path.relative_to(RAW_DIR.parent.parent)}")
    return df


def sanity_check(df: pd.DataFrame, ticker: str) -> None:
    print(f"\n=== Sanity check: {ticker} ===")
    print(f"Rows:        {len(df)}")
    print(f"Date range:  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Columns:     {list(df.columns)}")

    dupes = df.index.duplicated().sum()
    print(f"Duplicate dates: {dupes}")

    print("\nNaN count per column:")
    nans = df.isna().sum()
    print(nans.to_string() if nans.any() else "  none")

    # Trading-calendar gaps: weekends are 3 calendar days (Fri->Mon), a
    # Monday holiday makes 4 (Fri->Tue). Anything longer is worth eyeballing.
    day_gaps = df.index.to_series().diff().dt.days
    suspicious = day_gaps[day_gaps > 4]
    print(f"\nGaps > 4 calendar days between consecutive rows: {len(suspicious)}")
    for date, gap in suspicious.items():
        prev = date - pd.Timedelta(days=int(gap))
        print(f"  {prev.date()} -> {date.date()}  ({int(gap)} days)")

    # Adjustment status: Close is the raw print; Adj Close folds in
    # dividends (and splits, but MSFT hasn't split since 2003).
    if "Adj Close" in df.columns:
        diff_pct = (df["Close"] - df["Adj Close"]).abs() / df["Close"] * 100
        n_diff = (diff_pct > 1e-9).sum()
        print(
            f"\nAdjustment: Close vs Adj Close differ on {n_diff}/{len(df)} rows "
            f"(max {diff_pct.max():.2f}%, oldest rows differ most)."
        )
        print(
            "  -> 'Close' is UNadjusted; 'Adj Close' is dividend/split-adjusted. "
            "Returns-based features must use Adj Close."
        )
    else:
        print("\nWARNING: no 'Adj Close' column — adjustment status unclear.")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nLast 3 rows:")
    print(df.tail(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    df = fetch_raw(args.ticker, args.years, force=args.force)
    sanity_check(df, args.ticker)


if __name__ == "__main__":
    main()
