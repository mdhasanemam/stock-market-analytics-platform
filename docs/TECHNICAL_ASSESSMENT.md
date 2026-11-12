# Technical assessment of the original `stock-price-prediction` repository

Scope: README, the six notebooks and `indiv-datasets/` (AAPL, FB, MSFT, TSLA; Train/Validation/Test CSVs) of
`mdhasanemam/stock-price-prediction` (single commit "few updates"). Numbers below were measured while preparing this project.

## What exists
| Item | Finding |
|---|---|
| Data | Daily OHLCV + constant `OpenInt` (all zeros). AAPL 1984-09 → 2017-11, MSFT 1986-03 → 2017-11, TSLA 2010-06 → 2017-11, FB 2012-05 → 2017-11. Pre-split chronologically (≈70/15/15). Ends Nov 2017, so no live use is possible. |
| Features | KNN/RF notebooks: same-day `Open, High, Low, Volume`. Decision tree: all columns except Close/Stock/Date (includes constant `OpenInt`). Bayesian/SVR: those plus calendar parts and one-hot ticker, pooled across the four stocks. Gradient boosting: adds `Close_lag1-3` and `Close_ma3/ma5`. |
| Target | Same-day `Close` (price level), i.e. *nowcasting*, not forecasting. No future horizon exists in any notebook. |
| Missing values | `ffill` in KNN/RF; none needed elsewhere. |
| Models / tuning | KNN, RF, DT, GB via `GridSearchCV` (random K-fold, cv=3/5); Bayesian Ridge and SVR via a handful of manual configs on the validation split. |
| Metrics | MAE, MSE, RMSE, R². Plots are actual vs predicted on a sample index. |
| Engineering | Colab/Kaggle-specific paths and `files.upload()`; KNN and RF notebooks repeat the same 6-cell block (upload → features → split → grid search → metrics → plot) four times; no shared code, no tests, no saved models except Bayesian/SVR pickles inside Kaggle. |

## Problems found
1. **Same-day OHLC predicts same-day Close.** `High` and `Low` bracket the close and `Open` is correlated with it, so this is closer to interpolation than prediction. It cannot be used to forecast, because those values are not known beforehand.
2. **Shuffled splits.** KNN/RF use `train_test_split(shuffle=True)` — neighbouring days land in both train and test. `GridSearchCV`'s default K-fold is not time-aware either, so folds from the future train models that are validated on the past. Time-series data must be split and cross-validated chronologically.
3. **Target leakage in gradient boosting.** `Close_ma3` / `Close_ma5` are rolling means that *include the current Close*, which is the target.
4. **Extrapolation failure hidden by (1)+(2).** Trees and KNN cannot predict outside the training range. In a chronological 80/20 split of the AAPL file the training maximum close is 46.5 while the test maximum is 175.6.
5. **No baseline.** Without "tomorrow = today" nothing shows whether a model has any skill.
6. **Pooled price-level regression across tickers** mixes scales (≈$0.07 to $300+) and relies on a one-hot ticker to separate them.
7. **Constant/irrelevant columns** (`OpenInt`) and day-of-month/year as features invite spurious fits.
8. R² on price levels is near 1 for almost any model because prices are highly autocorrelated; it is not evidence of skill.

## Measured effect (AAPL, Random Forest 100 trees, one-off experiment run during the assessment; not part of the test suite)
| Setup | R² | MAE |
|---|---|---|
| Notebook-style: shuffled split, same-day OHLC → same-day Close | **0.9999** | 0.14 |
| Same features, chronological 80/20 split | **−2.00** | 44.5 |
| Honest next-day Close from day-t features, chronological split | −2.03 | 44.8 |
| Persistence (tomorrow = today) on that same test set | **0.998** | **0.98** |

The headline notebook result is a leakage artefact; a naive baseline beats the same model by ~45× in MAE once the evaluation is honest.

## What was reused
Model families and their intent (KNN, RF, Bayesian Ridge, Decision Tree, Gradient Boosting, SVR), the metric set,
the idea of lag/moving-average features (rebuilt without leakage), Bayesian predictive uncertainty as motivation for intervals,
and the four CSVs (concatenated chronologically into `data/sample/` for an offline demo mode).

## What changed
* Target is the **h-day log-return**; price forecast = last close × exp(predicted return).
* All features use information available at the close of day *t* (unit-tested, incl. mutation tests).
* **Walk-forward** validation with purging of the last *h* training targets; scalers live inside sklearn Pipelines.
* Persistence, historical-drift and MA-reversion **baselines** are scored beside every model.
* Hyper-parameters are conservative, fixed and shallow (daily returns have very low signal-to-noise). **No hyper-parameter search is implemented yet** — see roadmap.
