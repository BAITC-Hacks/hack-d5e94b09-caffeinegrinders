#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade >/dev/null
fi
PIP_CACHE_DIR=.venv/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python run.py "$@"
