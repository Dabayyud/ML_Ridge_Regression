# Pets at Home: machine-learning signal

Data snapshot: 11 November 2026. Last full month in the model: October 2026.

## Finding

A Ridge regression on nine public inputs does **not** predict next month's return on
Pets at Home relative to the FTSE 250. Under the rule agreed before the results were
seen, the model gets **no weight** in the Buy/Sell call.

| Measure (79 unseen months, Apr 2020 to Oct 2026) | Ridge | Plain OLS | Historical average |
| --- | --- | --- | --- |
| Out-of-sample R² vs historical average | −2.2% | −60.4% | 0 (benchmark) |
| RMSE | 9.34% | 11.71% | 9.24% |
| Months with direction right | 44.3% | 41.8% | 45.6% |

- 90% bootstrap range for Ridge R²: −5.5% to −0.1%.
- Diebold–Mariano p = 0.94; Clark–West p = 0.92 (no evidence Ridge beats the average).
- Backtest (hold PETS when forecast > 0, else the index, 0.5% round-trip cost):
  −0.60 cumulative log excess return, against −0.46 for always holding PETS.
- Forecast for November 2026: −0.39% excess return. Actual to 11 November: +8.7%.
- 12-month tilt: −3.7%, with a one-standard-deviation band of ±32%.

## Run it

```
pip install pandas numpy scikit-learn scipy matplotlib
python pets_ml_model.py     # model, tests, robustness -> results/
python make_charts.py       # three PNG charts -> results/
python power_check.py       # optional, 15-20 minutes
```

## Choices, reasons and drawbacks

| Choice | Reason | Drawback |
| --- | --- | --- |
| Target: next month's log excess return over the FTSE 250 | Matches the committee's question (this share or the market); gives 139 observations | Decision horizon is 12 months; a 1-month result only tilts it |
| Ridge regression | Interpretable; penalty controls overfitting with few observations and correlated inputs | Linear only; shrinks true effects too |
| Penalty chosen by time-series cross-validation inside each window | No future data used for tuning | Noisy with 60–138 months; hit the top of the grid in 39 of 79 windows |
| Walk-forward, 60-month first window | Mimics real-time forecasting | Only 79 test months, starting in the pandemic |
| Benchmark forecast: expanding historical mean | Standard in the literature (Campbell and Thompson, 2008) | A low bar in absolute terms |
| Index: iShares FTSE 250 ETF total return | Dividends counted on both sides | Net of the ETF's fee, which slightly flatters PETS |
| Yahoo Finance prices, dividends added back by hand | Free and reproducible; Yahoo's own adjusted close under-adjusts London dividends | Not an institutional source; check against LSEG if available |
| ONS retail sales and CPI, lagged one month | Released in the third week of the following month | Revised figures, not first releases |
| Google Trends "puppy", year-on-year change | Timely pet-demand proxy; yearly change removes seasonality and index rescaling | Search interest is not spending; values are rounded integers |
| CMA announcement dummy | The vets inquiry moved the shares | Seven events, all after September 2023 |

## Differences from the written plan

- Consumer confidence dropped: the free OECD series on FRED stops in January 2024.
- Benchmark is the FTSE 250 ETF total return, not the ^FTMC price index (kept as a robustness row).
- Macro lag is one month, not two, after checking ONS release dates.
- Google Trends uses a 12-month change, not 3-month, to remove seasonality.
- The pandemic is handled as a robustness check, not as a model input.
- Clark–West test, bootstrap range and a planted-signal power check added.

## Limits

- The planted-signal check shows low power: even a true correlation of 0.3 gave a
  positive out-of-sample R² in about a third of runs. Read the result as "no usable
  evidence", not "proof of no predictability".
- One share, one horizon, public data only.

## Sources

- Yahoo Finance: PETS.L, MIDD.L, ^FTMC, GBPUSD=X (daily, downloaded 11 Nov 2026).
- ONS time series D7G7 (CPI annual rate, released 21 Oct 2026) and J5EK (retail sales volume, released 23 Oct 2026).
- Google Trends, United Kingdom, "puppy" and "pet insurance", Jan 2013 to Oct 2026 (downloaded 11 Nov 2026).
- CMA, Veterinary services for household pets case page (gov.uk).
