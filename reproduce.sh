#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Rebuild every result for the Pets at Home ML signal from the frozen data snapshot.
#
#   ./reproduce.sh            model + charts + tests        (about 2 minutes)
#   ./reproduce.sh --power    also the power check          (about 30-40 minutes)
#
# First run: creates a private Python 3.13 environment in .venv/ and installs the exact
# package versions in requirements-lock.txt. Later runs reuse it.
#
# Needs either `uv` (recommended: `brew install uv`) or Python 3.11+ (`brew install python@3.13`).
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python

# 1. Create the environment once
if [ ! -x "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo ">> Creating .venv with uv (Python 3.13)"
        uv venv --python 3.13 .venv
    else
        BASE=""
        for c in python3.13 python3.12 python3.11 python3; do
            if command -v "$c" >/dev/null 2>&1 && \
               "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
                BASE="$c"; break
            fi
        done
        if [ -z "$BASE" ]; then
            echo "Need Python 3.11 or newer. Install one of these, then rerun:"
            echo "   brew install uv            (recommended)"
            echo "   brew install python@3.13"
            exit 1
        fi
        echo ">> Creating .venv with $BASE ($("$BASE" --version))"
        "$BASE" -m venv .venv
    fi
fi

# 2. Install the exact package versions (quick if already installed)
echo ">> Installing pinned packages"
if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$PY" -r requirements-lock.txt
else
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements-lock.txt
fi

# 3. Run everything
echo ">> Model";  "$PY" -W ignore pets_ml_model.py "$@"
echo ">> Charts"; "$PY" -W ignore make_charts.py
echo ">> Tests";  "$PY" -W ignore tests/test_pipeline.py
echo ">> Done. Results in results/, charts in results/charts/"
