# MSFT Stock Direction Predictor

* Educational ML project answering one question: will Microsoft close higher or lower N trading days from now? Up or down, nothing else. No price targets.
* The whole thing is designed around NOT cheating rather than around accuracy. Every stage exists to block one of the ways stock models silently leak future information into training.
    * Basically, beginner stock models usually "predict" well because they accidentally saw the future. This one is built so that can't happen, so whatever accuracy comes out is real.
* This is a learning project, not a trading system, and it will never place trades.

## The headline

```
                      test accuracy   the bar (always up)
Logistic regression        52.4%            52.4%
Random forest              47.6%            52.4%
```

* neither model beats "just say up every day." Logistic regression tied the bar because it ended up predicting up every single day itself. The random forest memorized all of its training data and still came in below the baseline
* walk forward validation reran the same experiment across seven eras (2019 to 2026) and the models lost or tied all 14 matchups against the baseline
* rerunning the unchanged pipeline at N of 1, 21, and 63 trading days added 40 more matchups. The overall record came out 8 wins in 54, scattered across unrelated windows, never adding up to a winning record at any horizon
* so: four textbook technical indicators carry no edge over market drift, from a day out to a quarter out. The pipeline exists to make that conclusion trustworthy

![walk forward results](results/walkforward_MSFT_n5.png)

## Where it's headed

* v1 and v2 settled the MSFT only question: no signal anywhere
* V3 (in progress) asks whether market context helps instead. The S&P 500, the Nasdaq, and the VIX might know something the stock's own history does not
* the goal is a model with a winning record against always up across walk forward windows. Every design decision along the way gets made on a validation slice, so the final judgment stays untouched until the design is frozen

## Read more

* [How the pipeline works](docs/pipeline.md): every module explained, plus the automated data refresh
* [Findings in full](docs/findings.md): the results, the autopsy of what the models actually did, and the walk forward story
* [What's next: V3](docs/v3.md): the market context attempt, its win condition, and the build order

## Run it

Install the packages in requirements.txt, then run the stages in order:

```
python data_pull.py     # fetch and cache 10 years of MSFT
python labels.py        # build the target + class distribution
python baselines.py     # score the zero ML strategies
python splits.py        # show the train / gap / test cut
python features.py      # build the four clues
python model.py         # train logistic regression, one test evaluation
python diagnose.py      # autopsy the saved predictions
python backtest.py      # walk forward validation across eras
python horizons.py      # rerun everything at N of 1, 5, 21, 63
```

Run model.py again with the model flag set to rf for the random forest (same protocol). Every script also takes a flag for N (the horizon in trading days, default 5) and one for the ticker (default MSFT).

## Rules the project runs on

1. No lookahead. A feature on day T may only use information available by day T's close.
2. Split by time, never shuffled, with a gap equal to N before the test period.
3. Fit scalers on training data only.
4. Touch the test set once per real iteration.
5. Out of sample accuracy is the headline number, never training accuracy.
6. Every evaluation gets compared against naive baselines.
7. A great looking result triggers a leak hunt, not a celebration.
