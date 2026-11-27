# Data sources and provenance

All inputs are a frozen snapshot taken on **11 November 2026**. `MANIFEST.json` holds the
SHA-256 of each file; `tests/test_pipeline.py` fails if any file changes.

## data/prices_monthly.csv

| Column | Source | Ticker | Construction |
|---|---|---|---|
| `pets_close` | Yahoo Finance chart API | PETS.L (Pets at Home Group plc, GBp) | Close on the last trading day of the month |
| `pets_tr` | same | PETS.L + its cash dividends | Total-return index, dividends reinvested at the ex-date close |
| `midd_tr` | same | MIDD.L (iShares FTSE 250 UCITS ETF **GBP (Dist)**, distributing share class) | Total-return index built the same way from close + distributions |
| `ftmc` | same | ^FTMC (FTSE 250 price index, no dividends) | Month-end close; used only for the price-index robustness check |
| `gbpusd` | same | GBPUSD=X | Month-end close (last available FX day of the month) |
| `vol63` | derived | from `pets_tr` | Sample standard deviation (ddof = 1) of the last 63 daily log total returns, x sqrt(252), on the month's last trading day |

- **Retrieval:** Yahoo chart API (`/v8/finance/chart/<ticker>`, `interval=1d`, `events=div`,
  `period1` = 1 Mar 2014, `period2` = retrieval time), fetched in a browser session on 11 Nov 2026.
  Daily rows with no close were dropped. PETS has no share splits in the period.
- **Total return:** `TR_t = TR_{t-1} x (P_t + D_t) / P_{t-1}`, with `P` the unadjusted close and `D` the
  cash dividend on its ex-date. Yahoo's own adjusted close was not used, because it under-adjusted
  London dividends when checked. Dividends are reinvested gross (no withholding tax).
- **Why the ETF as benchmark:** it includes dividends, as `pets_tr` does. Its ongoing charge is
  deducted from its returns, which slightly lowers the hurdle PETS has to clear.
- **November 2026 is a part month** (to 11 Nov). It is used only to report the live forecast's
  partial outcome, never as a training target.
- **Known gap:** the raw daily files behind this snapshot were not saved. `tools/rebuild_prices.py`
  re-downloads them with the same steps and reports any difference from the snapshot (it has not
  been run yet; it needs internet access to Yahoo).

## data/ons_raw.json (Office for National Statistics)

| Key | Series ID | Dataset | Title | Release used |
|---|---|---|---|---|
| `cpi` | D7G7 | MM23 | CPI annual rate, all items | 21 Oct 2026 (data to Sep 2026) |
| `rsi` | J5EK | DRSI | Retail sales volume, seasonally adjusted, all retailers incl. fuel | 23 Oct 2026 (data to Sep 2026) |

- Retrieved 11 Nov 2026 from the ONS time-series pages' `/data` JSON endpoint; monthly values from Jan 2013.
- **Lag:** ONS publishes month m in the third week of month m+1, so the model uses month t-1 at the
  end of month t (one-month shift in `build_features`).
- **Limitation:** these are today's **revised** values, not the first estimates investors saw.

## data/google_trends_gb.csv (Google Trends)

- Query: keywords **"puppy"** and **"pet insurance"** compared in one request (so both share one scale),
  geography **GB**, time range **2013-01-01 to 2026-10-31**, category **All**, property **Web search**,
  language `en-GB`, time zone 0. Monthly values, integers 0-100. Retrieved 11 Nov 2026 via the
  Trends `explore` and `widgetdata/multiline` endpoints.
- Only "puppy" enters the model, as `log(value_t / value_{t-12})`. The year-on-year ratio removes
  seasonality and does not depend on the 0-100 rescaling, which changes with each download.
- **Limitations:** values are rounded integers; search interest is not spending.

## CMA announcement months (in `pets_ml_model.py`, `CMA_MONTHS`)

Checked on 23 Nov 2026 against the CMA case page
(gov.uk/cma-cases/veterinary-services-market-for-pets-review):

| Month | Event |
|---|---|
| 2023-09 | 7 Sep 2023, review launched |
| 2024-03 | 12 Mar 2024, update report and consultation on a market investigation |
| 2024-05 | 23 May 2024, market investigation reference made |
| 2025-06 | 10 Jun 2025, statutory deadline extended by 6 months |
| 2025-10 | 15 Oct 2025, provisional decision |
| 2026-03 | 24 Mar 2026, final decision report |
| 2026-09 | 22 Sep 2026, substantive Order and Undertakings made |

## Planned but not used

- **UK consumer confidence:** the free OECD series on FRED (`CSCICP03GBM665S`) ends in January 2024,
  so it could not cover the test period. Dropped before any model was run.
