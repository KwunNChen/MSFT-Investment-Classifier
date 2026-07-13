"""Pull and cache raw daily OHLCV data via yfinance, then sanity-check it.

Usage:
    python data_pull.py [--ticker MSFT] [--years 10] [--force]

The raw pull is cached in data/raw/ and never modified after download.
Downloads use auto_adjust=False so the frame keeps both the raw `Close`
and the dividend/split-adjusted `Adj Close`, making the adjustment status
explicit instead of implied by a library default.
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}_daily.parquet"


def fetch_raw(ticker: str, years: int, force: bool = False) -> pd.DataFrame:
    """Return ~`years` of daily OHLCV for `ticker`, cached in data/raw/."""
    cache_path = raw_path(ticker)
    if cache_path.exists() and not force:
        print(f"[cache] using existing {cache_path}")
        return pd.read_parquet(cache_path)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    print(f"[download] {ticker} daily OHLCV {start_date.date()} -> {end_date.date()}")
    price_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if price_data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yf.download returns MultiIndex columns (field, ticker) even for a
    # single ticker; flatten to plain field names.
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = [column[0] for column in price_data.columns]
    price_data = price_data.sort_index()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    price_data.to_parquet(cache_path)
    print(f"[cache] wrote {len(price_data)} rows -> {cache_path}")
    return price_data


def sanity_check(price_data: pd.DataFrame, ticker: str) -> None:
    print(f"\n=== Sanity check: {ticker} ===")
    print(f"Rows:        {len(price_data)}")
    print(f"Date range:  {price_data.index.min().date()} -> {price_data.index.max().date()}")
    print(f"Columns:     {list(price_data.columns)}")

    duplicate_dates = price_data.index.duplicated().sum()
    print(f"Duplicate dates: {duplicate_dates}")

    print("\nNaN count per column:")
    nan_counts = price_data.isna().sum()
    print(nan_counts.to_string() if nan_counts.any() else "  none")

    # Trading-calendar gaps: weekends are 3 calendar days (Fri->Mon), a
    # Monday holiday makes 4 (Fri->Tue). Anything longer is worth eyeballing.
    days_between_rows = price_data.index.to_series().diff().dt.days
    suspicious_gaps = days_between_rows[days_between_rows > 4]
    print(f"\nGaps > 4 calendar days between consecutive rows: {len(suspicious_gaps)}")
    for gap_end_date, gap_days in suspicious_gaps.items():
        gap_start_date = gap_end_date - pd.Timedelta(days=int(gap_days))
        print(f"  {gap_start_date.date()} -> {gap_end_date.date()}  ({int(gap_days)} days)")

    # Adjustment status: Close is the raw print; Adj Close folds in
    # dividends (and splits, but MSFT hasn't split since 2003).
    if "Adj Close" in price_data.columns:
        adjustment_pct_diff = (
            (price_data["Close"] - price_data["Adj Close"]).abs() / price_data["Close"] * 100
        )
        rows_differing = (adjustment_pct_diff > 1e-9).sum()
        print(
            f"\nAdjustment: Close vs Adj Close differ on {rows_differing}/{len(price_data)} rows "
            f"(max {adjustment_pct_diff.max():.2f}%, oldest rows differ most)."
        )
        print(
            "  -> 'Close' is UNadjusted; 'Adj Close' is dividend/split-adjusted. "
            "Returns-based features must use Adj Close."
        )
    else:
        print("\nWARNING: no 'Adj Close' column — adjustment status unclear.")

    print("\nFirst 3 rows:")
    print(price_data.head(3).to_string())
    print("\nLast 3 rows:")
    print(price_data.tail(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    price_data = fetch_raw(args.ticker, args.years, force=args.force)
    sanity_check(price_data, args.ticker)


if __name__ == "__main__":
    main()
