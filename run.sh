#!/usr/bin/env bash
# Everything, from a clean checkout. Offline: no API key, no network, no model
# download, no notebook. Regenerates every table and artifact in the README.
# Verified on Python 3.12.3. Total runtime ~12 minutes.
set -euo pipefail

echo "=== 1/5  one-command reconcile (headline numbers + HTML report) ==="
python -m recon.cli --orders 300 --seed 7

echo; echo "=== 2/5  test suite ==="
python -m pytest -q tests/

echo; echo "=== 3/5  static benchmark: difficulty, multiway, unmodeled, AI ablation ==="
python benchmark.py

echo; echo "=== 4/5  live layer: replay gate (bit-identical or fail) ==="
python replay.py --steps 50 --seed 5

echo; echo "=== 5/5  live layer: adversarial mutation + override suite ==="
python mutate.py --steps 60 --seeds 6

echo; echo "artifacts/  run.json  benchmark.md  benchmark.json"
echo "            exceptions.csv  matches.csv  close_report.html  mutation_runs.json"
