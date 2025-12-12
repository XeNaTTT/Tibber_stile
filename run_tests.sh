#!/usr/bin/env bash
set -euo pipefail
python -m py_compile Tibber_stile.py PV_Chart.py
printf "py_compile completed.\n"
