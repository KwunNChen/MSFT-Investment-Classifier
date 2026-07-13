# MSFT Stock Direction Predictor

* Educational ML project answering one question: will Microsoft close higher or lower N trading days from now? Up or down, nothing else. No price targets.
* The whole thing is designed around not cheating rather than around accuracy. Every stage exists to block one of the ways stock models silently leak future information into training.
* This is a learning project, not a trading system, and it will never place trades.

### The pipeline

```
Yahoo Finance API > data_pull.py > labels.py > baselines.py > splits.py > features.py > model.py
```

* every stage runs on its own, e.g. `python labels.py`
* big thing here is the parameter N: it is the horizon in trading days (default 5), passed in everywhere and never hardcoded, so the same pipeline reruns unchanged at N of 1, 21, or 63

## data_pull.py

* downloads 10 years of daily OHLCV (about 2,500 rows) and caches it in data/raw
* sanity checks on every run: row count, date range, NaNs, duplicate dates, calendar gaps
* yfinance receives JSON from Yahoo and hands back a pandas DataFrame, which gets cleaned and sorted
* keeps both Close and Adj Close. Everything downstream uses Adj Close, because dividends make the raw Close drop on days where nothing market related happened

## labels.py

* answers: did the price close higher N trading days later? 1 for yes, 0 for no
* the last N rows can't be labeled, their "N days later" hasn't happened yet, so they get NaN and are dropped
    * at N = 5 that's why 2,514 rows but only 2,509 labels
* `shift(-N)` makes each row look N down (its own future). `shift(N)` makes each row look N up (the past)
    * the label is the ONLY thing allowed to look down. Everything else looks up, or it's lookahead bias

## baselines.py

* this is the bar. Zero ML strategies scored first, so a future model gets judged against them and not against 50%
* always_up(): a column of 1s, one per day, never looks at any data. Scores 58.9% purely off the market's upward drift
    * MSFT went up in about 59% of all 5 day windows. A model scoring 60% would look impressive next to a coin flip while actually being nearly worthless. That's why the number gets written down
* persistence(): whatever the stock did over the last N days, guess it keeps doing that. Scored 49.8%, a literal coin flip
    * finding: last week's direction tells you nothing about next week
    * implemented by sliding the label column down N rows (shift with positive N, looking at the past), so each day sees the direction of the window that just finished
* evaluate_baseline() is the grader. Lines up guesses against true labels, counts % correct, and only counts days where both a guess and an answer exist (coverage)
    * every model later gets graded by this exact function, so all accuracies in the project are computed identically
* scores are saved to results/ so later steps compare against saved numbers, not memory

## splits.py

* cuts the timeline into train, then gap, then test, strictly in date order. The model learns only from the past and gets graded on a future it never saw
* chronology alone is not enough though. This part is easier to visualize, with N = 2:

```
Day:    1    2    3    4  |  5    6  |  7    8    9    10
             TRAIN            GAP           TEST
```

* since labels look N forward, day 6's label reads day 8, and day 8 sits inside the test set. Training on day 6 would leak test prices into training
* so the last N days before the boundary go into a gap that belongs to neither side. That's why the gap always equals the current N

## features.py

* goal: turn raw prices into four clues per day for the model to train on, all computed strictly from the past
* ma_ratio: this week's average price vs this month's, as one number. Positive means trading above the monthly trend
    * we use the ratio instead of two raw averages because a raw average is just a price level ($52 in 2016, $385 in 2026). +0.02 means "2% above trend" in any era
* volatility: the spread (standard deviation) of the last 20 daily returns. Small = calm grind, large = whipsaw. Ours ranged 0.3% to 7%
* volume_ratio: today's volume vs its own 20 day average, minus 1. +3.0 means four times normal volume, something happened
* momentum: the return over the past N days. Same N as the label, so the clue and the question always cover matching window sizes
* the first 20 rows get dropped since the rolling windows have no history yet, mirroring how the label loses the last N rows

## model.py

* logistic regression trained on the pre 2024 slice, evaluated exactly ONCE on the held out test slice
* the scaler that normalizes features is fitted on training rows only, so test statistics never leak into training
* test predictions get saved to results/, so every later diagnostic reads that file instead of touching the test set again
* basically this is the part that puts the other code together

### Results at N = 5

```
Logistic regression   52.4%
Always up             52.4%
Persistence           49.4%
```

* the model ties the naive baseline exactly. Its learned weights are all near zero: it found no usable signal in the four clues and basically rediscovered the always up strategy
* reported as the honest finding, not tuned away. Four standard technical indicators carry no detectable edge over market drift at the 5 day horizon
* train accuracy was 60.4% vs 52.4% on test, which looks like overfitting until you notice always_up itself drops the same 8 points between eras. That gap is a harder market regime, not memorization

### Rules the project runs on

1. No lookahead. A feature on day T may only use information available by day T's close.
2. Split by time, never shuffled, with a gap equal to N before the test period.
3. Fit scalers on training data only.
4. Touch the test set once per real iteration.
5. Out of sample accuracy is the headline number, never training accuracy.
6. Every evaluation gets compared against naive baselines.
7. A great looking result triggers a leak hunt, not a celebration.

### Data refresh

* a GitHub Actions workflow redownloads the full 10 year window every Saturday and commits the updated parquet
* full redownload is deliberate: Yahoo recalculates Adj Close retroactively whenever a dividend is paid, so appending new rows would leave stale history
* the sliding window also drops data older than 10 years automatically

#### Status

* done (steps 1 to 6): data, labels, baselines, gapped split, features, logistic regression evaluation
* next: prediction diagnostics, Random Forest under the same protocol, walk forward validation, and the horizon comparison at N of 1, 21, and 63
