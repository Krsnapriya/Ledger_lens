"""BenchRec — running the engine on real-world reconciliation data.

BenchRec (Operartis, ICAIF 2023) is a real cash-reconciliation dataset: bank
statement entries matched to internal ledger entries, with grouped and null
matches. It is the closest public artefact to the problem this engine solves,
and it is not the problem this engine was built for. That gap is the point.

    python benchrec.py --data /path/to/benchrec --year 2023

What transfers and what does not
--------------------------------
The identifier layer does not transfer at all. R3.0a, R3.1 and R3.2 look for
`setl_0000` and `UTR############` in a narration; BenchRec references are
anonymised strings from a different institution's format, so those rules are
dead on arrival. The withholding-rate gate does not transfer either — 0.1/1/2/5%
are Indian payment-gateway rates.

What is left is the part that is actually general: exact-amount plus date-window
assignment resolved by Hungarian, cardinality-constrained meet-in-the-middle
subset sum with a uniqueness certificate, duplicate clustering, and abstention.
Measuring only that is the honest test — it says how much of the engine is a
reconciliation engine and how much of it is a Razorpay-shaped reconciliation
engine.

Scoring
-------
A match is correct when the ledger row and the bank row it names share a
`matchId` in the published ground truth. Everything else is a false match. This
is deliberately harsher than the competition's own metric, which scores an
allocation string: a pair that lands in the wrong group counts against us here
even if the allocation text would have looked similar.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from recon.ai import SemanticResolver
from recon.match import MatcherConfig, match_settlements_bank
from recon.models import BankLine, Settlement
from recon.money import fmt


def _paise(s: str) -> int:
    if not s or not s.strip():
        return 0
    return int((Decimal(s.strip()) * 100).to_integral_value(rounding="ROUND_HALF_UP"))


def _d(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def load(path: Path, year: str | None, cap: int):
    """Ledger rows become 'settlements', bank rows become 'bank lines'.

    Amounts are taken as magnitudes. BenchRec pairs a ledger CR against a bank DR
    for the same movement, and its own ground truth balances on magnitude, so
    sign is a partition of the file rather than of the economics.
    """
    ledger: list[Settlement] = []
    bank: list[BankLine] = []
    truth_group: dict[str, str] = {}
    seen_a, seen_b = set(), set()

    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            mid = r["matchId"]
            if r["A_id"]:
                dt = _d(r["A_valueDate"])
                if dt is None or (year and not r["A_valueDate"].startswith(year)):
                    continue
                if r["A_id"] in seen_a:
                    continue
                seen_a.add(r["A_id"])
                ledger.append(Settlement(f"A{r['A_id']}", dt,
                                         abs(_paise(r["A_amount"])), None))
                if mid:
                    truth_group[f"A{r['A_id']}"] = mid
            elif r["B_id"]:
                dt = _d(r["B_valueDate"])
                if dt is None or (year and not r["B_valueDate"].startswith(year)):
                    continue
                if r["B_id"] in seen_b:
                    continue
                seen_b.add(r["B_id"])
                narr = (r["B_transactionReferences"] + " " +
                        r["B_transactionAttributes"]).strip()
                bank.append(BankLine(f"B{r['B_id']}", dt, narr,
                                     abs(_paise(r["B_amount"])), 0))
                if mid:
                    truth_group[f"B{r['B_id']}"] = mid
            if cap and len(ledger) >= cap and len(bank) >= cap:
                break

    ledger.sort(key=lambda s: (s.settled_on, s.settlement_id))
    bank.sort(key=lambda b: (b.value_date, b.txn_id))
    return ledger, bank, truth_group


def run(ledger, bank, truth, cfg: MatcherConfig):
    import time
    t0 = time.perf_counter()
    raw, excs = match_settlements_bank(ledger, bank, cfg,
                                       SemanticResolver(use_llm=False))
    elapsed = time.perf_counter() - t0

    # BUG FOUND AND FIXED IN THIS RUN: this function called match_settlements_bank
    # directly and treated every returned Match as asserted. reconcile() withdraws
    # anything below cfg.confidence_threshold to BELOW_CONFIDENCE_THRESHOLD before
    # counting a match as engine-asserted (recon/match.py:1052). Skipping that
    # step meant every low-confidence tie-break the engine itself would have
    # abstained on was counted as a false match. The first BenchRec run reported
    # 8.944% FMR measuring a harness bug, not the engine.
    matches = [m for m in raw if m.confidence >= cfg.confidence_threshold]
    withdrawn = [m for m in raw if m.confidence < cfg.confidence_threshold]

    tp = fp = 0
    unlabelled = 0
    fp_by_rule: Counter = Counter()
    tp_by_rule: Counter = Counter()
    withdrawn_would_be_fp = withdrawn_would_be_tp = 0
    for m in withdrawn:
        for r in m.right_ids:
            ga, gb = truth.get(m.left_id), truth.get(r)
            if ga is not None and gb is not None:
                if ga == gb:
                    withdrawn_would_be_tp += 1
                else:
                    withdrawn_would_be_fp += 1
    for m in matches:
        for r in m.right_ids:
            ga, gb = truth.get(m.left_id), truth.get(r)
            if ga is None or gb is None:
                unlabelled += 1
                continue
            if ga == gb:
                tp += 1
                tp_by_rule[m.rule] += 1
            else:
                fp += 1
                fp_by_rule[m.rule] += 1

    # How many ledger rows COULD have been matched: those whose group also
    # contains at least one bank row present in this slice.
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"a": 0, "b": 0})
    for eid, g in truth.items():
        groups[g]["a" if eid.startswith("A") else "b"] += 1
    matchable = {g for g, c in groups.items() if c["a"] and c["b"]}
    matchable_ledger = sum(1 for s in ledger if truth.get(s.settlement_id) in matchable)
    matched_ledger = len({m.left_id for m in matches
                          if truth.get(m.left_id) in matchable})

    return {
        "ledger_rows": len(ledger), "bank_rows": len(bank),
        "asserted_pairs": tp + fp + unlabelled,
        "tp": tp, "fp": fp, "unlabelled": unlabelled,
        "fmr_pct": round(100 * fp / (tp + fp), 3) if (tp + fp) else 0.0,
        "matchable_ledger_rows": matchable_ledger,
        "matched_ledger_rows": matched_ledger,
        "coverage_pct": round(100 * matched_ledger / matchable_ledger, 2) if matchable_ledger else 0.0,
        "exceptions": len(excs),
        "exception_mix": dict(Counter(e.reason.value for e in excs).most_common(8)),
        "fp_by_rule": dict(fp_by_rule.most_common(6)),
        "tp_by_rule": dict(tp_by_rule.most_common(6)),
        "wall_clock_s": round(elapsed, 2),
        "exposure_paise": sum(e.cash_impact_paise for e in excs),
        "withdrawn_by_threshold": len(withdrawn),
        "withdrawn_would_have_been_tp": withdrawn_would_be_tp,
        "withdrawn_would_have_been_fp": withdrawn_would_be_fp,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="benchrec")
    ap.add_argument("--data", type=Path, default=Path("/tmp/benchrec"))
    ap.add_argument("--file", default="BenchRec_cash_v1.0_train.csv")
    ap.add_argument("--year", default="2023")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--no-fuzzy", action="store_true",
                    help="disable narration similarity — isolate amount matching")
    ap.add_argument("--no-unique-assignment", action="store_true",
                    help="disable bidirectional uniqueness on the 1:1 Hungarian layer")
    ap.add_argument("--no-contingent-refusal", action="store_true",
                    help="disable the subset-sum contingency/DUPLICATE_CLAIM gate")
    ap.add_argument("--with-withholding", action="store_true",
                    help="re-enable the Razorpay-specific withholding-rate rule "
                    "(measured harmful on BenchRec: 11 of 14 matches wrong; off by default)")
    a = ap.parse_args(argv)

    ledger, bank, truth = load(a.data / a.file, a.year or None, a.cap)
    print(f"BenchRec {a.file} · year {a.year or 'all'}")
    print(f"  ledger rows {len(ledger):,} · bank rows {len(bank):,}")

    cfg = MatcherConfig(fuzzy_narration=not a.no_fuzzy,
                        plausible_withholding_bp=(10, 100, 200, 500) if a.with_withholding else (),
                        require_unique_assignment=not a.no_unique_assignment,
                        refuse_contingent_subsets=not a.no_contingent_refusal,
                        date_window_days=a.window,
                        subset_sum_lookback_days=a.window,
                        subset_sum_lookahead_days=a.window)
    r = run(ledger, bank, truth, cfg)

    print()
    print("=" * 78)
    print(f"  COVERAGE          {r['coverage_pct']:.2f}%  "
          f"({r['matched_ledger_rows']:,}/{r['matchable_ledger_rows']:,} matchable ledger rows)")
    print(f"  FALSE MATCH RATE  {r['fmr_pct']:.3f}%  "
          f"({r['fp']:,} wrong of {r['tp'] + r['fp']:,} scoreable pairs)")
    print(f"  withdrawn by threshold (< {cfg.confidence_threshold})  {r['withdrawn_by_threshold']:,}  "
          f"[of which {r['withdrawn_would_have_been_fp']:,} would have been FALSE, "
          f"{r['withdrawn_would_have_been_tp']:,} would have been correct]")
    print(f"  wall clock        {r['wall_clock_s']}s")
    print(f"  exceptions        {r['exceptions']:,}   exposure {fmt(r['exposure_paise'])}")
    print("=" * 78)
    print("  correct pairs by rule:")
    for k, v in r["tp_by_rule"].items():
        print(f"    {k:<44} {v:,}")
    if r["fp_by_rule"]:
        print("  FALSE pairs by rule:")
        for k, v in r["fp_by_rule"].items():
            print(f"    {k:<44} {v:,}")
    print("  exception mix:")
    for k, v in r["exception_mix"].items():
        print(f"    {k:<44} {v:,}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
