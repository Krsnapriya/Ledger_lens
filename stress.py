"""Adversarial sweep. A single seed proves nothing; variance is the real signal.

Runs the full loop across many seeds and escalating difficulty, and reports the
distribution rather than the best case. Any config where false-match rate is
non-zero is printed loudly, because that is the failure that matters.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

from recon import evaluate, generate, normalize
from recon.match import MatcherConfig, reconcile


def one(seed: int, difficulty: float, orders: int = 300) -> dict:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        g = generate.generate(n_orders=orders, seed=seed, difficulty=difficulty)
        generate.write(g, d)
        truth = json.loads((d / "truth.json").read_text())
        data, rejects = normalize.load_all(d)
        r = reconcile(data, MatcherConfig())
        mr = evaluate.match_rate(r, truth)
        xs = evaluate.score_exceptions(r.exceptions, truth)
        h = xs["excluding_trivial_ORDER_NO_PAYMENT"]
        return {"seed": seed, "difficulty": difficulty,
                "match_rate": mr["settlement_match_rate_pct"],
                "false_match_rate": mr["false_match_rate_pct"],
                "exc_precision": h["precision"], "exc_recall": h["recall"],
                "exc_spurious": h["fp"], "exc_missed": h["fn"],
                "exc_expected": h["tp"] + h["fn"],
                "rec_per_s": r.stats["records_per_second"],
                "rejects": len(rejects)}


def main() -> int:
    seeds = list(range(1, 41))
    bad = []
    print(f"{'difficulty':>10} {'match rate %':>26} {'false match %':>22} "
          f"{'exc precision':>22} {'exc recall':>20}")
    print(f"{'':>10} {'mean   min   max':>26} {'mean   max':>22} "
          f"{'mean   min':>22} {'mean   min':>20}")
    print("-" * 104)
    for diff in (0.0, 0.5, 1.0, 2.0, 3.0):
        rows = [one(s, diff) for s in seeds]
        mrs = [r["match_rate"] for r in rows]
        fmr = [r["false_match_rate"] for r in rows]
        ep = [r["exc_precision"] for r in rows]
        er = [r["exc_recall"] for r in rows]
        flag = "  <-- FALSE MATCHES" if max(fmr) > 0 else ""
        print(f"{diff:>10.1f} {statistics.mean(mrs):>10.2f} {min(mrs):>7.2f} {max(mrs):>7.2f} "
              f"{statistics.mean(fmr):>10.3f} {max(fmr):>11.3f} "
              f"{statistics.mean(ep):>12.4f} {min(ep):>9.4f} "
              f"{statistics.mean(er):>10.4f} {min(er):>9.4f}{flag}")
        # Only judge exception P/R where there were exceptions to find. At
        # difficulty 0 there are none, and 0/0 was being scored as a failure.
        bad += [r for r in rows if r["false_match_rate"] > 0
                or (r["exc_expected"] > 0 and (r["exc_precision"] < 0.95 or r["exc_recall"] < 0.95))]

    print("\nworst cases (false matches, or exception P/R below 0.95):")
    if not bad:
        print("  none")
    for r in sorted(bad, key=lambda x: (-x["false_match_rate"], x["exc_precision"]))[:18]:
        print(f"  seed={r['seed']:<3} diff={r['difficulty']:<4} match={r['match_rate']:>6.2f}% "
              f"falsematch={r['false_match_rate']:>6.3f}% excP={r['exc_precision']:.3f} "
              f"excR={r['exc_recall']:.3f} spurious={r['exc_spurious']} missed={r['exc_missed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
