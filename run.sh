#!/usr/bin/env bash
# Everything, from a clean checkout. Offline: no API key, no network, no model
# download, no notebook. Regenerates every table and artifact in the README.
# Verified on Python 3.12.3. Total runtime ~12 minutes.
set -euo pipefail

echo "=== 1/6  the loop, at the brief's 50-record floor ==="
python -m recon.cli --orders 50 --seed 7 --no-ablation

echo; echo "=== 2/6  the agent: close the books, then work the exception queue ==="
python -m recon.agent --orders 300 --seed 7

echo; echo "=== 3/6  full batch + HTML report ==="
python -m recon.cli --orders 300 --seed 7

echo; echo "=== 4/6  test suite ==="
python -m pytest -q tests/

echo; echo "=== 5/6  static benchmark: difficulty, multiway, unmodeled, AI ablation ==="
python benchmark.py

echo; echo "=== 6/6  live layer: replay gate + adversarial suite ==="
python replay.py --steps 50 --seed 5
python mutate.py --steps 60 --seeds 6

echo; echo "artifacts/  run.json  benchmark.md  benchmark.json"
echo "            exceptions.csv  matches.csv  close_report.html  mutation_runs.json"
