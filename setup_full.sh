#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.cache/uv"
PROJECT_PROXY="${PROJECT_PROXY:-$(python3 -c 'import urllib.request; print(urllib.request.getproxies().get("https", ""))')}"
if [[ -n "$PROJECT_PROXY" ]]; then export HTTPS_PROXY="$PROJECT_PROXY" HTTP_PROXY="$PROJECT_PROXY"; fi
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r requirements-light.lock
uv pip install --python .venv/bin/python --target reproduction_full/legacy_python --no-deps -r requirements-original-overlay.txt
.venv/bin/python fetch_light_model.py
if [[ ! -f artifacts/tofu/manifest.json ]]; then .venv/bin/python fetch_tofu.py; fi
.venv/bin/python run_original.py audit_full.py
printf '\nRun: .venv/bin/python run_full_experiment.py\n'
