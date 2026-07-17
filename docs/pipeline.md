# How the pipeline works

Module by module. Back to the [main page](../README.md), or jump to the [findings](findings.md).

### The pipeline

```
Yahoo Finance API > data_pull.py > labels.py > baselines.py > splits.py > features.py > model.py > diagnose.py > backtest.py
```

* every stage runs on its own, e.g. `python labels.py`
* big thing here is the parameter N: the horizon in trading days (default 5), passed in everywhere and NEVER hardcoded
    * Basically, change one number and the whole pipeline reruns unchanged at N of 1, 21, or 63. Nothing else needs touching.

## data_pull.py

* downloads 10 years of daily OHLCV (about 2,500 rows) and caches it in data/raw
    * Basically, OHLCV = Open, High, Low, Close, Volume. One row per trading day.
* sanity checks on every run: row count, date range, NaNs, duplicate dates, calendar gaps
    * This is because every one of those problems would show up weeks later as a mystery bug in the model. Way cheaper to catch it at the door.
* yfinance receives JSON from Yahoo and hands back a pandas DataFrame, which gets cleaned and sorted
* keeps both Close and Adj Close, and everything downstream uses Adj Close ONLY
    * This is because dividends make the raw Close drop about 0.2% on days where nothing market related happened. Use raw Close and you inject hundreds of fake down moves into the data.
* for V3 the same script also pulls the market context tickers (^GSPC, ^IXIC, ^VIX) with ZERO code changes, since ticker was a parameter from day one
    * the alignment pass caught a phantom VIX quote on Memorial Day 2026 (a market holiday, Yahoo glitch) and confirmed VIX volume is all zeros. Both quirks documented and handled before any feature touches the data

## labels.py

* answers: did the price close higher N trading days later? 1 for yes, 0 for no
* the last N rows can't be labeled, so they get NaN and are dropped
    * Basically, the newest days are questions with no answers yet. Their "N days later" hasn't happened. At N = 5 that's why 2,514 rows but only 2,509 labels.
* `shift(-N)` makes each row look N down (its own future). `shift(N)` makes each row look N up (the past)
    * the label is the ONLY thing in the entire project allowed to look down. Everything else looks up, or it's lookahead bias, the number one way stock models cheat.

## baselines.py

* this is the bar. Zero ML strategies get scored FIRST, so any future model gets judged against them and not against 50%
* always_up(): a column of 1s, one per day. Never looks at any data, still scores 58.9%
    * This is because MSFT went up in about 59% of all 5 day windows. Pure market drift. A model scoring 60% would look impressive next to a coin flip while actually being nearly worthless, and that's exactly why the number gets written down.
* persistence(): whatever the stock did over the last N days, guess it keeps doing that. Scored 49.8%, a LITERAL coin flip
    * Finding: last week's direction tells you nothing about next week.
    * Implemented by sliding the label column down N rows (positive shift, looking at the past), so each day only sees the direction of the window that already finished. Sliding it down just 1 row would peek at a window that hasn't closed yet.
* evaluate_baseline() is the grader. Lines up guesses against true labels, counts % correct
    * only counts days where both a guess and an answer exist (called coverage)
    * every model later gets graded by this EXACT function, so all accuracies in the project are computed identically
* scores get saved to results/ so later steps compare against saved numbers, not memory

## splits.py

* cuts the timeline into train, then gap, then test, strictly in date order
    * Basically, the model studies old data and gets graded on newer data it never saw. Shuffling would let it study the future.
* chronology alone is not enough though. This part is easier to visualize, with N = 2:

```
Day:    1    2    3    4  |  5    6  |  7    8    9    10
             TRAIN            GAP           TEST
```

* since labels look N forward, day 6's label reads day 8, and day 8 sits inside the test set. Training on day 6 leaks test prices into training even though the split LOOKS clean
* so the last N days before the boundary go into a gap that belongs to neither side
    * This is because the contamination zone is always exactly as deep as the label's reach. That's why the gap always equals the current N.
* also home to walk_forward_windows(), which chains many of these cuts through the decade for [backtest.py](findings.md)

## features.py

* goal: turn raw prices into four clues per day for the model to train on, all computed strictly from the past
* ma_ratio: this week's average price vs this month's, as one number. Positive means trading above the monthly trend
    * This is because a raw average is just a price level ($52 in 2016, $385 in 2026), useless across eras. The ratio means the same thing in any year: +0.02 = 2% above trend.
* volatility: the spread (standard deviation) of the last 20 daily returns
    * Basically, small = calm grind, large = whipsaw. Ours ranged 0.3% to 7%.
* volume_ratio: today's volume vs its own 20 day average, minus 1
    * Basically, +3.0 means four times normal volume. Something happened that day.
* momentum: the return over the past N days. Same N as the label, so the clue and the question always cover matching window sizes
* the first 20 rows get dropped since the rolling windows have no history to average yet
    * Mirror image of labels.py: features lose the FIRST rows (no past yet), labels lose the LAST rows (no future yet). The table gets nibbled from both ends.

## model.py

* basically this is the part that puts the other code together: assemble the table, split it, learn, predict once, save everything

##### Logistic Regression

* logistic regression trained on the pre 2024 slice, evaluated exactly ONCE on the held out test slice
    * Basically, the model learns 5 numbers: how much to trust each of the four clues, plus a base rate. Each day it multiplies clues by trust, adds it up, and squashes that into a probability of up. Above 50% means predict up.
* the scaler that normalizes features is fitted on training rows ONLY
    * This is because "what does normal look like" has to come from the past. Learn it from all the data and test era statistics sneak into the training, which is leakage even with a perfect split.
* test predictions get saved to results/, so every later diagnostic reads that file instead of touching the test set again
    * Basically, the model sits the exam once and the answer sheet gets filed. Peeking at the test set repeatedly while tweaking would slowly turn it into a second training set.

##### Random Forest

* random forest, run with `python model.py` and the model flag set to rf. Same protocol as logistic regression: same split, same gap, same grader, ONE test evaluation
    * Basically, instead of learning 5 numbers it grows 100 decision trees. Each tree is a flowchart of yes/no questions about the clues ("is volatility above 2%? then is momentum negative?") and all 100 vote on up or down
    * This means it can catch combo patterns the straight line model is blind to, like "momentum only matters when volatility is high." If any such pattern existed in the clues, the forest could find it
* the catch: a default forest has enough capacity to MEMORIZE the entire training set, which finally gives the train vs test gap something real to measure

Both models' results live in [findings.md](findings.md).

## backtest.py and horizons.py

* backtest.py is walk forward validation: it reruns the whole experiment across seven rolling eras, training a FRESH model per era so every year of the decade gets one turn as the exam
* horizons.py loops all of that over N of 1, 5, 21, and 63. Zero pipeline edits needed, since N was never hardcoded
* both are evaluation only, NOT trading simulators. The full story and the charts live in [findings.md](findings.md)

### Data refresh

* a GitHub Actions workflow redownloads the full 10 year window every Saturday and commits the updated parquet
    * This is because Yahoo recalculates Adj Close retroactively whenever a dividend is paid. Append only the new rows and your old rows quietly go stale.
* the sliding window also drops data older than 10 years automatically. Fresh data and cleanup in one move
