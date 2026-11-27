"""
Checks for the ML signal (FI4008 Group 10). Run after pets_ml_model.py:

    python tests/test_pipeline.py        (or: python -m pytest tests)

1. No look-ahead: a month's features don't change when later data are removed.
2. Placebo: with the target shuffled, the model can't beat the historical mean.
3. Data: input files are byte-for-byte the 11 Nov 2026 snapshot (MANIFEST.json).
4. Report: every number quoted in the report's ML sections matches results.json.
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pets_ml_model as m  # noqa: E402

warnings.filterwarnings("ignore")


def test_no_lookahead():
    """Cut every input off at month t and rebuild: month t's features must be identical."""
    px, cpi, rsi, gt = m.load_inputs()
    full, _, _ = m.build_features(px, cpi, rsi, gt)
    for t in ["2016-06", "2018-01", "2020-03", "2022-09", "2024-12", "2026-08"]:
        p = pd.Period(t, "M")
        cut = lambda s: s.loc[:p]
        _, latest, _ = m.build_features(cut(px), cut(cpi), cut(rsi), cut(gt), last_full_month=t)
        assert np.allclose(latest.iloc[0].values.astype(float),
                           full.loc[p, m.FEATURES].values.astype(float), equal_nan=True), \
            f"features for {t} change when later data are removed"


def test_placebo_shuffled_target():
    """If shuffled returns could be 'predicted', something would be leaking."""
    train, _, _, _ = m.load()
    t = train.copy()
    t["y"] = np.random.default_rng(1).permutation(t["y"].values)
    r2 = m.r2_oos(m.walk_forward(t, m.FEATURES))
    assert r2 < 0.02, f"shuffled target still 'predicted' (R2 {r2:.3f})"


def test_input_checksums():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for rel, digest in manifest["inputs"].items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == digest, f"{rel} differs from the frozen snapshot"


def test_report_numbers():
    """Each number as written in the report (rounded) must match the code's output."""
    R = json.loads((ROOT / "results" / "results.json").read_text())
    S, B, rob = R["summary"], R["backtest"], R["robustness"]
    pct = lambda v: v * 100
    checks = [  # (label in report, value written, value from code, tolerance)
        ("139 feature months", 139, R["sample"]["rows"], 0),
        ("79 scored forecasts", 79, S["n_oos"], 0),
        ("out-of-sample R2 -2.2%", -2.2, pct(S["r2_oos_ridge"]), 0.05),
        ("bootstrap 5th pct -5.5%", -5.5, pct(S["r2_boot_5"]), 0.05),
        ("bootstrap 95th pct -0.1%", -0.1, pct(S["r2_boot_95"]), 0.05),
        ("Diebold-Mariano p 0.94", 0.94, S["dm_p"], 0.005),
        ("Clark-West p 0.92", 0.92, S["cw_p"], 0.005),
        ("35 of 79 direction right", 35, S["hits_ridge"], 0),
        ("historical mean 46%", 46, pct(S["hit_mean"]), 0.5),
        ("always-underperform 57%", 57, pct(1 - S["share_up_months"]), 0.5),
        ("OLS R2 -60.4%", -60.4, pct(S["r2_oos_ols"]), 0.05),
        ("39 windows at max penalty", 39, S["windows_at_max_alpha"], 0),
        ("forecast sd 0.7%", 0.7, pct(S["forecast_sd"]), 0.05),
        ("actual sd 9.2%", 9.2, pct(S["actual_sd"]), 0.05),
        ("Lasso -4.5%", -4.5, pct(rob["Lasso instead of Ridge"]), 0.05),
        ("price inputs only -2.4%", -2.4, pct(rob["Price features only (4)"]), 0.05),
        ("without pandemic -1.9%", -1.9, pct(rob["Scored without Mar 2020 - Jun 2021"]), 0.05),
        ("price-index benchmark -2.6%", -2.6, pct(rob["Benchmark = FTSE 250 price index"]), 0.05),
        ("3-month horizon -27.6%", -27.6, pct(rob["3-month horizon"]), 0.05),
        ("strategy -0.60", -0.60, B["strategy_cum_excess"], 0.005),
        ("always PETS -0.46", -0.46, B["always_pets_cum_excess"], 0.005),
        ("0.25% per switch", 0.25, pct(B["per_switch_cost"]), 1e-9),
    ]
    if "power_check" in R:
        p3 = [r for r in R["power_check"] if abs(r["planted_correlation"] - 0.3) < 1e-9][0]
        checks.append(("planted rho 0.3 detected in about a third (35%) of runs", 35,
                       pct(p3["share_r2_above_zero"]), 0.5))
    bad = [(lab, w, round(g, 3)) for lab, w, g, tol in checks if abs(w - g) > tol + 1e-9]
    assert not bad, f"report numbers that don't match the code: {bad}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
