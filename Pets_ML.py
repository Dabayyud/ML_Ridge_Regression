"""
Pets at Home (LSE: PETS) - machine-learning signal for the FI4008 pitch.

Model   : Ridge regression, refitted every month on past data only (walk-forward).
Target  : next month's log total return of PETS minus the FTSE 250's.
Compare : plain OLS (same features) and the expanding historical mean.

Run     : python pets_ml_model.py           (reads ./data, writes ./results)
Needs   : pandas numpy scikit-learn scipy

Data snapshot (taken 11 Nov 2026):
  data/prices_monthly.csv   Yahoo Finance daily closes + dividends, turned into month-end
                            total-return series. pets_tr = PETS.L, midd_tr = iShares FTSE 250
                            UCITS ETF (MIDD.L), ftmc = FTSE 250 price index, gbpusd = GBPUSD=X,
                            vol63 = annualised st. dev. of PETS daily returns, last 63 days.
  data/ons_raw.json         ONS series D7G7 (CPI annual rate) and J5EK (retail sales volume).
  data/google_trends_gb.csv Google Trends, UK, monthly, search terms "puppy", "pet insurance".
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "results"
OUT.mkdir(exist_ok=True)

LAST_FULL_MONTH = "2026-10"      # November 2026 is a part-month in the snapshot
START = 60                       # months in the first training window
ALPHAS = np.logspace(-2, 4, 30)  # Ridge penalties tried inside each training window
COST = 0.005                     # round-trip trading cost used in the backtest

# CMA vets-market milestones (month of announcement). Source: CMA case page
# gov.uk/cma-cases/veterinary-services-market-for-pets-review
CMA_MONTHS = ["2023-09",  # review launched
              "2024-03",  # CMA proposes a formal market investigation
              "2024-05",  # market investigation reference made
              "2025-06",  # timetable extended
              "2025-10",  # provisional decision
              "2026-03",  # final decision
              "2026-09"]  # final orders

FEATURES = ["reversal", "momentum", "vol3m", "mkt_mom3",
            "retail_yoy", "cpi_yoy", "gbp_chg3", "puppy_yoy", "cma_event"]
PRICE_ONLY = FEATURES[:4]


# ----------------------------------------------------------------------------- data
def load(benchmark="midd_tr"):
    px = pd.read_csv(DATA / "prices_monthly.csv", index_col="ym")
    px.index = pd.PeriodIndex(px.index, freq="M")
    ons = json.loads((DATA / "ons_raw.json").read_text())
    to_s = lambda k: pd.Series({pd.Period(m, "M"): float(v) for m, v in ons[k]["data"].items()})
    cpi, rsi = to_s("cpi"), to_s("rsi")
    gt = pd.read_csv(DATA / "google_trends_gb.csv", index_col="ym")
    gt.index = pd.PeriodIndex(gt.index, freq="M")

    r_p = np.log(px["pets_tr"]).diff()
    r_m = np.log(px[benchmark]).diff()
    ex = r_p - r_m                                   # monthly log excess return

    X = pd.DataFrame(index=px.index)
    X["reversal"] = ex                               # this month's excess return
    X["momentum"] = ex.shift(1).rolling(11).sum()    # months t-11 .. t-1 (skips month t)
    X["vol3m"] = px["vol63"]
    X["mkt_mom3"] = r_m.rolling(3).sum()
    # ONS publishes month m in the third week of m+1, so at the end of month t the
    # newest figure an investor has is for t-1: shift by one month.
    X["retail_yoy"] = ((rsi / rsi.shift(12) - 1) * 100).reindex(px.index).shift(1)
    X["cpi_yoy"] = cpi.reindex(px.index).shift(1)
    X["gbp_chg3"] = np.log(px["gbpusd"]).diff(3)
    # Year-on-year change removes the strong seasonality in searches and is unaffected
    # by Google rescaling the 0-100 index.
    X["puppy_yoy"] = np.log(gt["puppy"] / gt["puppy"].shift(12)).reindex(px.index)
    X["cma_event"] = [1.0 if str(p) in CMA_MONTHS else 0.0 for p in px.index]

    ex_full = ex.loc[:LAST_FULL_MONTH]               # drop the part-month before making targets
    y = ex_full.shift(-1).rename("y")                # NEXT month's excess return
    full = pd.concat([X, y], axis=1).loc[:LAST_FULL_MONTH]
    train = full.dropna()                            # rows whose target is known
    latest = full[FEATURES].iloc[[-1]]               # features at the last full month-end
    return train, latest, ex_full, px


# ----------------------------------------------------------------------------- models
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(5)))


def ols():
    return make_pipeline(StandardScaler(), LinearRegression())


def lasso():
    return make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), max_iter=50000))


def walk_forward(data, cols, make_model=ridge, start=START, gap=0, target="y", keep_coefs=False):
    """Expanding window. At row t the model sees rows 0..t-1-gap only, then predicts row t.
    gap > 0 is needed when the target spans several months (its value is not yet known)."""
    X, y = data[cols], data[target]
    rows, coefs = [], []
    for t in range(start, len(X)):
        Xtr, ytr = X.iloc[: t - gap], y.iloc[: t - gap]
        m = make_model().fit(Xtr, ytr)
        o = ols().fit(Xtr, ytr)
        rows.append({"actual": y.iloc[t], "model": m.predict(X.iloc[[t]])[0],
                     "ols": o.predict(X.iloc[[t]])[0], "mean": ytr.mean(),
                     "alpha": getattr(m[-1], "alpha_", np.nan)})
        if keep_coefs:
            coefs.append(m[-1].coef_)
    res = pd.DataFrame(rows, index=X.index[start:])
    # label each forecast with the month whose return it predicts
    res.index = (res.index + 1).rename("target_month")
    if keep_coefs:
        return res, pd.DataFrame(coefs, index=res.index, columns=cols)
    return res


# ----------------------------------------------------------------------------- evaluation
def r2_oos(res, col="model"):
    return 1 - ((res["actual"] - res[col]) ** 2).sum() / ((res["actual"] - res["mean"]) ** 2).sum()


def dm_test(res, col="model"):
    """Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold small-sample correction.
    H1: the model's squared errors are smaller than the historical mean's. One-sided."""
    d = (res["actual"] - res["mean"]) ** 2 - (res["actual"] - res[col]) ** 2
    n = len(d)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return stat, 1 - stats.t.cdf(stat, n - 1)


def cw_test(res, col="model"):
    """Clark-West (2007) test for nested models: adds back the noise a larger model
    carries under the null, which the DM test penalises."""
    f = ((res["actual"] - res["mean"]) ** 2
         - ((res["actual"] - res[col]) ** 2 - (res["mean"] - res[col]) ** 2))
    stat = f.mean() / np.sqrt(f.var(ddof=1) / len(f))
    return stat, 1 - stats.norm.cdf(stat)


def bootstrap_r2(res, col="model", n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, p, b = res["actual"].values, res[col].values, res["mean"].values
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r2 = 1 - ((a[idx] - p[idx]) ** 2).sum(1) / ((a[idx] - b[idx]) ** 2).sum(1)
    return np.percentile(r2, [5, 95])


def backtest(res, col="model", cost=COST):
    """Each month: hold PETS if the forecast excess return is positive, otherwise hold the
    FTSE 250 (excess return = 0). Half the round-trip cost is charged on every switch."""
    pos = (res[col] > 0).astype(float)
    switches = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * res["actual"] - switches * cost / 2
    ir = lambda s: s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else np.nan
    return {"strategy_cum_excess": strat.sum(), "always_pets_cum_excess": res["actual"].sum(),
            "strategy_info_ratio": ir(strat), "always_pets_info_ratio": ir(res["actual"]),
            "months_in_pets": int(pos.sum()), "switches": int(switches.sum())}, strat


def summary(res):
    a = res["actual"]
    err = lambda c: a - res[c]
    dm, cw = dm_test(res), cw_test(res)
    lo, hi = bootstrap_r2(res)
    return {
        "n_oos": len(res),
        "r2_oos_ridge": r2_oos(res), "r2_oos_ols": r2_oos(res, "ols"),
        "r2_boot_5": lo, "r2_boot_95": hi,
        "rmse_ridge": np.sqrt((err("model") ** 2).mean()),
        "rmse_ols": np.sqrt((err("ols") ** 2).mean()),
        "rmse_mean": np.sqrt((err("mean") ** 2).mean()),
        "mae_ridge": err("model").abs().mean(), "mae_ols": err("ols").abs().mean(),
        "mae_mean": err("mean").abs().mean(),
        "hit_ridge": (np.sign(res["model"]) == np.sign(a)).mean(),
        "hit_ols": (np.sign(res["ols"]) == np.sign(a)).mean(),
        "hit_mean": (np.sign(res["mean"]) == np.sign(a)).mean(),
        "share_up_months": (a > 0).mean(),
        "dm_stat": dm[0], "dm_p": dm[1], "cw_stat": cw[0], "cw_p": cw[1],
        "median_alpha": res["alpha"].median(),
    }


# ----------------------------------------------------------------------------- main
def main():
    train, latest, ex, px = load()
    print(f"Training rows: {len(train)}  ({train.index[0]} to {train.index[-1]}), "
          f"features: {len(FEATURES)}")
    train.to_csv(OUT / "model_dataset.csv")
    train.describe().T.to_csv(OUT / "summary_stats.csv")
    train[FEATURES].corr().to_csv(OUT / "feature_correlations.csv")

    # 1. main walk-forward run
    res, coef_path = walk_forward(train, FEATURES, keep_coefs=True)
    S = summary(res)
    bt, strat = backtest(res)
    res.assign(strategy=strat).to_csv(OUT / "walk_forward_forecasts.csv")
    coef_path.to_csv(OUT / "coefficient_path.csv")

    # 2. robustness - one change at a time
    pandemic = pd.period_range("2020-03", "2021-06", freq="M")
    rob = {"Main model (Ridge, 9 features)": r2_oos(res),
           "Lasso instead of Ridge": r2_oos(walk_forward(train, FEATURES, lasso)),
           "Price features only (4)": r2_oos(walk_forward(train, PRICE_ONLY)),
           "Scored without Mar 2020 - Jun 2021": r2_oos(res[~res.index.isin(pandemic)])}
    t2, _, _, _ = load("ftmc")
    rob["Benchmark = FTSE 250 price index"] = r2_oos(walk_forward(t2, FEATURES))
    t3 = train.copy()
    t3["y3"] = (ex.shift(-1) + ex.shift(-2) + ex.shift(-3)).reindex(t3.index)
    rob["3-month horizon"] = r2_oos(walk_forward(t3.dropna(), FEATURES, gap=2, target="y3"))
    pd.Series(rob, name="r2_oos").to_csv(OUT / "robustness.csv")

    # 3. final fit on everything -> forecast for the month after the last full one
    final = ridge().fit(train[FEATURES], train["y"])
    coefs = pd.Series(final[-1].coef_, index=FEATURES)
    coefs.to_csv(OUT / "final_coefficients.csv", header=["std_coef"])
    fc = final.predict(latest)[0]
    contrib = pd.Series(final[-1].coef_ * final[0].transform(latest)[0], index=FEATURES)
    resid_sd = (res["actual"] - res["model"]).std()
    F = {"forecast_month": str(latest.index[0] + 1), "forecast_excess": fc,
         "final_alpha": final[-1].alpha_, "intercept": final[-1].intercept_,
         "tilt_12m": np.mean(res["model"].tail(5).tolist() + [fc]) * 12,   # last 6 forecasts
         "band_12m": resid_sd * np.sqrt(12), "resid_sd_monthly": resid_sd,
         "sign_flips": {c: int((np.sign(coef_path[c]).diff().abs() > 0).sum()) for c in FEATURES},
         "latest_features": latest.iloc[0].to_dict(), "contributions": contrib.to_dict()}

    # part-month check: what has actually happened in November so far
    nov = px.loc["2026-11"] if pd.Period("2026-11", "M") in px.index else None
    if nov is not None:
        oct_ = px.loc["2026-10"]
        F["nov_to_date_excess"] = (np.log(nov["pets_tr"] / oct_["pets_tr"])
                                   - np.log(nov["midd_tr"] / oct_["midd_tr"]))

    out = {"summary": S, "backtest": bt, "robustness": rob, "final": F,
           "coefficients": coefs.to_dict()}
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float))

    pct = lambda v: f"{v * 100:6.2f}%"
    print(f"\nOut-of-sample months: {S['n_oos']}  ({res.index[0]} to {res.index[-1]})")
    print(f"R2 out-of-sample   Ridge {pct(S['r2_oos_ridge'])}   OLS {pct(S['r2_oos_ols'])}"
          f"   (90% bootstrap range for Ridge {pct(S['r2_boot_5'])} to {pct(S['r2_boot_95'])})")
    print(f"RMSE               Ridge {pct(S['rmse_ridge'])}   OLS {pct(S['rmse_ols'])}   mean {pct(S['rmse_mean'])}")
    print(f"MAE                Ridge {pct(S['mae_ridge'])}   OLS {pct(S['mae_ols'])}   mean {pct(S['mae_mean'])}")
    print(f"Hit rate           Ridge {pct(S['hit_ridge'])}   OLS {pct(S['hit_ols'])}   mean {pct(S['hit_mean'])}"
          f"   (PETS beat the index in {pct(S['share_up_months'])} of months)")
    print(f"Diebold-Mariano    stat {S['dm_stat']:.2f}  p {S['dm_p']:.3f}     Clark-West  stat {S['cw_stat']:.2f}  p {S['cw_p']:.3f}")
    print(f"Median Ridge penalty chosen: {S['median_alpha']:.1f}")
    print("\nBacktest:", {k: round(v, 3) for k, v in bt.items()})
    print("\nRobustness (R2 out-of-sample):")
    for k, v in rob.items():
        print(f"  {k:38s} {pct(v)}")
    print("\nStandardised coefficients, final fit:")
    print((coefs * 100).round(3).sort_values().to_string())
    print("\nSign changes along the walk-forward path:", F["sign_flips"])
    print(f"\nForecast for {F['forecast_month']}: {pct(fc)} excess return vs FTSE 250")
    print(f"12-month tilt (average of last 6 forecasts x 12): {pct(F['tilt_12m'])}"
          f"   +/- {pct(F['band_12m'])} (one st. dev.)")
    if "nov_to_date_excess" in F:
        print(f"November to 11th, actual excess return so far: {pct(F['nov_to_date_excess'])}")


if __name__ == "__main__":
    main()
