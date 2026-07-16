# MSFT Stock Direction Predictor

* Educational ML project answering one question: will Microsoft close higher or lower N trading days from now? Up or down, nothing else. No price targets.
* The whole thing is designed around NOT cheating rather than around accuracy. Every stage exists to block one of the ways stock models silently leak future information into training.
    * Basically, beginner stock models usually "predict" well because they accidentally saw the future. This project is built so that can't happen, so whatever accuracy comes out is real.
* This is a learning project, not a trading system, and it will never place trades.

## The headline

```
                      test accuracy   the bar (always up)
Logistic regression        52.4%            52.4%
Random forest              47.6%            52.4%
```

* neither model beats "just say up every day." The linear model tied the bar by literally becoming it, and the forest memorized the training data perfectly then did WORSE than doing nothing
* walk forward validation reran the same experiment across seven eras (2019 to 2026): the models went 0 for 14 against the baseline. The finding holds in every regime
* then the unchanged pipeline reran at N of 1, 21, and 63 trading days: 8 lucky wins in 54 total fights, zero winning records at any horizon
* conclusion: four textbook technical indicators carry no edge over market drift at ANY horizon from a day out to a quarter out, and the pipeline is built carefully enough that you can trust that sentence

![walk forward results](results/walkforward_MSFT_n5.png)

## Read more

* [How the pipeline works](docs/pipeline.md): every module explained, from the data pull to the models, plus the automated data refresh
* [Findings in full](docs/findings.md): the results, the autopsy of what the models actually did, and the walk forward story

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
