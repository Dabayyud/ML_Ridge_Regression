# Change log

Every change to the plan or the code, with the reason, and whether results had been
seen first.

## 11 Nov 2026: first run (the numbers in the group report)

Changes from the written plan, all made **before** any model was run:

| Change | Reason |
|---|---|
| UK consumer confidence dropped | No free series runs past January 2024 |
| Benchmark = FTSE 250 ETF total return (MIDD.L), not the ^FTMC price index | Dividends on both sides; the price index is kept as a robustness check |
| ONS data lagged one month, not two | Release dates checked: month m comes out in the third week of m+1 |
| Google Trends as a 12-month change, not 3-month | Removes seasonality and the effect of Google rescaling the index |
| Pandemic handled as a robustness check, not as an input | Keeps the input list fixed in advance |
| Clark-West test, bootstrap interval and power check added | Stronger evidence on significance and on the test's ability to detect a signal |

## 23-26 Nov 2026: code review and repository version (results already seen)

The model and its settings are **unchanged**, so every number in the report is reproduced
exactly. Changes are to documentation, outputs and checks only:

| Change | Why |
|---|---|
| CMA dates checked against the CMA case page | They were first entered from memory; all seven were correct |
| `walk_forward_forecasts.csv` now dated: feature month, training window, target month, live Nov 2026 row | Lets anyone audit each forecast |
| Counts added to the output: months with direction right, windows at the top of the penalty grid, forecast and actual sd | These are quoted in the report |
| Power check (`--power`) built into the script | The earlier figures were copied from a console log |
| `make_charts.py`, `tests/test_pipeline.py`, `MANIFEST.json` added | Charts, leak checks and a check of every number in the report |
| Comments rewritten to explain each step and link it to the report | Readability |

## 26 Nov 2026: folder organised for reproducibility

- Files placed in `data/`, `tests/`, `tools/` and `docs/`.
- `reproduce.sh` builds a Python 3.13 environment and runs everything in one command.
- `requirements-lock.txt` pins all 18 packages, including sub-dependencies.
- A clean copy was rebuilt from scratch with `./reproduce.sh --power`; all tests passed
  and every number matched the report.

## Known limitations, kept so the report's numbers stay reproducible

These were found in the 23 Nov review. Fixing them changes the numbers, so they are
documented here instead. A separate rerun with fixes 1-3 applied (scaler inside each fold,
MSE tuning, penalty grid up to 1,000,000) gave out-of-sample R² of -1.3%: the conclusion
does not change.

1. The scaler is fitted once per training window, then RidgeCV tunes on folds of the
   already-scaled data. No test data leak in, but tuning folds share scaling information.
2. RidgeCV scores each ~10-month tuning fold by R², which is noisy and favours strong penalties.
3. The penalty grid stops at 10,000 and the top was chosen in 39 of 79 windows.
4. The backtest charges 0.25% per switch (half of 0.5% round trip). A switch sells one
   holding and buys the other, so 0.5% per switch is arguably more realistic; the strategy
   trails holding PETS even at zero cost (see `cost_sensitivity.csv`).
5. In the 3-month robustness check the inner tuning folds have no gap.
6. The bootstrap treats months as independent.
