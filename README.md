# MSFT Stock Direction Predictor

* Educational ML project answering one question: will Microsoft close higher or lower N trading days from now? Up or down, nothing else. No price targets.
* The whole thing is designed around NOT cheating rather than around accuracy. Every stage exists to block one of the ways stock models silently leak future information into training.
    * Basically, beginner stock models usually "predict" well because they accidentally saw the future. This project is built so that can't happen, so whatever accuracy comes out is real.
* This is a learning project, not a trading system, and it will never place trades.


## Nerd stuff below

### The pipeline

```
Yahoo Finance API > data_pull.py > labels.py > baselines.py > splits.py > features.py > model.py > diagnose.py
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
* logistic regression trained on the pre 2024 slice, evaluated exactly ONCE on the held out test slice
    * Basically, the model learns 5 numbers: how much to trust each of the four clues, plus a base rate. Each day it multiplies clues by trust, adds it up, and squashes that into a probability of up. Above 50% means predict up.
* the scaler that normalizes features is fitted on training rows ONLY
    * This is because "what does normal look like" has to come from the past. Learn it from all the data and test era statistics sneak into the training, which is leakage even with a perfect split.
* test predictions get saved to results/, so every later diagnostic reads that file instead of touching the test set again
    * Basically, the model sits the exam once and the answer sheet gets filed. Peeking at the test set repeatedly while tweaking would slowly turn it into a second training set.

### Results at N = 5

```
Logistic regression   52.4%
Always up             52.4%
Persistence           49.4%
```

* the model TIES the naive baseline exactly. Its learned weights are all near zero: it found no usable signal in the four clues and basically rediscovered the always up strategy
* reported as the honest finding, not tuned away. Four standard technical indicators carry no detectable edge over market drift at the 5 day horizon
* train accuracy was 60.4% vs 52.4% on test, which looks like overfitting until you notice always_up itself drops the same 8 points between eras
    * Basically, the test years were just a flatter market. The gap is a harder regime, not memorization.

### diagnose.py

* given four standard technical clues, logistic regression concluded none of them beat "MSFT usually drifts up," and became the always up baseline with extra steps
    * Basically, the model always predicted up. ALWAYS, 100% of the time (498 out of 498 test days). This is because its up probability never crossed below 0.5
    * This is because the probability starts near 60% thanks to the training set (60% of training days were up), and the near zero weights never push it far from there. It lived between 0.52 and 0.74 for two straight years
* the four piles (confusion matrix): 261 right (the up days), 237 wrong (the down days), 0 and 0 in the "said down" piles, because it never said down once
* so its errors are exactly the market's down weeks, concentrated in every decline on the chart
    * the chart in results/ shows it best: the red (wrong) dots paint every drop, because the model kept saying up while the price fell

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
    * This is because Yahoo recalculates Adj Close retroactively whenever a dividend is paid. Append only the new rows and your old rows quietly go stale.
* the sliding window also drops data older than 10 years automatically. Fresh data and cleanup in one move
