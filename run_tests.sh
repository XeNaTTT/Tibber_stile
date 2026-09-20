#!/usr/bin/env bash
set -euo pipefail
python -m py_compile Tibber_stile.py tibber_live.py energy_chart_utils.py PV_Chart.py render_preview.py
printf "py_compile completed.\n"
python -m unittest -v
