"""One command, cold start, deterministic output.

    python -m recon.cli --orders 300 --seed 7

Generates the data, reconciles it, evaluates against ground truth, runs the
ablation, and writes every artifact. No notebook cells, no manual steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import evaluate, generate, normalize, report
from .match import MatcherConfig, reconcile
from .money import fmt

ABLATIONS = [
    ("A_exact_ids_only", dict(exact_ids=True, tolerance_window=False,
                              fuzzy_narration=False, subset_sum=False)),
    ("B_plus_amount_date", dict(exact_ids=True, tolerance_window=True,
                                fuzzy_narration=False, subset_sum=False)),
    ("C_plus_fuzzy_narration", dict(exact_ids=True, tolerance_window=True,
                                    fuzzy_narration=True, subset_sum=False)),
    ("D_plus_subset_sum_full", dict(exact_ids=True, tolerance_window=True,
                                    fuzzy_narration=True, subset_sum=True)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ledgerlens")
    ap.add_argument("--orders", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="0.0 = pristine data, 1.0 = default noise, >1 = harsher")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--use-llm", action="store_true",
                    help="enable optional LLM re-rank on the residual (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--require-unique-assignment", type=lambda x: x.lower() != "false",
                    default=True, metavar="{True,False}",
                    help="Hungarian 1:1 layer: refuse a pairing unless it is "
                    "bidirectionally the only feasible candidate (AMBIGUOUS_ASSIGNMENT "
                    "otherwise). Default True.")
    ap.add_argument("--refuse-contingent-subsets", type=lambda x: x.lower() != "false",
                    default=True, metavar="{True,False}",
                    help="subset-sum layer: refuse a group unique only because an "
                    "earlier match consumed its twin (DUPLICATE_CLAIM otherwise). "
                    "Default True.")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    def say(*x):
        if not a.quiet:
            print(*x)

    # ---------- 1. generate ----------
    g = generate.generate(n_orders=a.orders, seed=a.seed, difficulty=a.difficulty)
    generate.write(g, a.data)
    truth = json.loads((a.data / "truth.json").read_text())
    say(f"generated  {len(g.orders)} orders · {len(g.payments)} payments · "
        f"{len(g.refunds)} refunds · {len(g.settlements)} settlements · {len(g.bank)} bank lines")

    # ---------- 2. ingest ----------
    data, rejects = normalize.load_all(a.data)
    if rejects:
        say(f"ingest rejects: {len(rejects)} (reported, not dropped)")

    # ---------- 3. reconcile ----------
    cfg = MatcherConfig(confidence_threshold=a.threshold, use_llm=a.use_llm,
                        require_unique_assignment=a.require_unique_assignment,
                        refuse_contingent_subsets=a.refuse_contingent_subsets)
    result = reconcile(data, cfg)

    # A second pass with the threshold floored, used only to draw the abstention
    # curve. The headline numbers always come from `result`, at the real threshold.
    curve_cfg = MatcherConfig(confidence_threshold=0.0, use_llm=False)
    curve_result = reconcile(data, curve_cfg)

    # ---------- 4. evaluate ----------
    ms = evaluate.score_matches(result.matches, truth)
    xs = evaluate.score_exceptions(result.exceptions, truth)
    mr = evaluate.match_rate(result, truth)
    curve = evaluate.abstention_curve(curve_result.matches, truth)

    unreconciled = sum(
        e.blocking_data.get("payout_paise", 0) or e.blocking_data.get("amount_paise", 0)
        for e in result.exceptions
        if e.reason.value in {"SETTLEMENT_NOT_IN_BANK", "UNEXPLAINED_BANK_CREDIT",
                              "DUPLICATE_BANK_CREDIT"})

    payload = {
        "schema_version": "1.0.0",   # exception taxonomy + artifact contract
        "python": sys.version.split()[0],
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": a.seed, "difficulty": a.difficulty, "dataset": str(a.data),
        "config": vars(cfg),
        "throughput": result.stats,
        "match_rate": mr,
        "match_scores": ms,
        "exception_scores": xs,
        "abstention_curve": curve,
        "ingest_rejects": list(rejects),
        "value": {
            "gross_captured": sum(p.gross_paise for p in data["payments"]),
            "payout_declared": sum(s.payout_paise for s in data["settlements"]),
            "bank_credits": sum(b.credit_paise for b in data["bank"]),
            "unreconciled_exposure": unreconciled,
            "exception_cash_at_risk": sum(e.cash_impact_paise for e in result.exceptions),
        },
        "ai": {"backend": result.ai_stats.backend,
               "residual_seen": result.ai_stats.residual_seen,
               "accepted": result.ai_stats.accepted,
               "rejected_by_constraint": result.ai_stats.rejected_by_constraint,
               "tiebreaks_invoked": result.ai_stats.tiebreaks_invoked,
               "tiebreak_margins": result.ai_stats.tiebreak_margins[:20],
               "similarity_scores_computed": result.ai_stats.proposals,
               "llm_calls": result.ai_stats.llm_calls,
               "llm_errors": result.ai_stats.llm_errors,
               "notes": result.ai_stats.notes},
    }

    # ---------- 5. ablation ----------
    if not a.no_ablation:
        rows = []
        for name, flags in ABLATIONS:
            r = reconcile(data, MatcherConfig(confidence_threshold=a.threshold, **flags))
            m = evaluate.match_rate(r, truth)
            rows.append({"config": name,
                         "layers": ",".join(k for k, v in flags.items() if v),
                         **m, "exceptions": len(r.exceptions),
                         "wall_clock_s": r.stats["wall_clock_s"]})
        payload["ablation"] = rows

    # ---------- 6. artifacts ----------
    a.out.mkdir(parents=True, exist_ok=True)
    report.write_json(a.out / "run.json", payload)
    report.write_exceptions_csv(a.out / "exceptions.csv", result.exceptions)
    report.write_matches_csv(a.out / "matches.csv", result.matches)
    report.write_html(a.out / "close_report.html", payload, result.exceptions, result.matches)

    # ---------- 7. stdout summary ----------
    l3 = ms["L3_settlement_bank"]
    say("")
    say("=" * 78)
    say(f"  SETTLEMENT MATCH RATE   {mr['settlement_match_rate_pct']:.2f}%   "
        f"({mr['settlements_correctly_reconciled']}/{mr['settlements_expected_to_land']} payouts)")
    say(f"  FALSE MATCH RATE        {mr['false_match_rate_pct']:.3f}%   "
        f"<- the number that decides whether this is usable")
    say(f"  L3 precision / recall   {l3['precision']:.4f} / {l3['recall']:.4f}")
    say(f"  L1 precision / recall   {ms['L1_order_payment']['precision']:.4f} / "
        f"{ms['L1_order_payment']['recall']:.4f}")
    say(f"  throughput              {result.stats['records_per_second']:,} rec/s  "
        f"({result.stats['records_ingested']:,} records in {result.stats['wall_clock_s']}s)")
    say(f"  unreconciled exposure   {fmt(unreconciled)}")
    say("=" * 78)
    say("  accuracy by tier (1=clean, 3=garbled narration + split/merge):")
    for k, v in ms["L3_by_difficulty_tier"].items():
        say(f"    {k}: n={v['truth_pairs']:<4} precision={v['precision']:.4f} "
            f"recall={v['recall']:.4f}  fp={v['fp']} fn={v['fn']}")
    say("  exception detection (excluding trivial ORDER_NO_PAYMENT):")
    h = xs["excluding_trivial_ORDER_NO_PAYMENT"]
    say(f"    precision={h['precision']:.4f} recall={h['recall']:.4f} "
        f"missed={h['fn']} spurious={h['fp']}")
    say(f"    abstentions routed to human: {xs['abstentions_routed_to_human']}")
    say("=" * 78)
    say(f"  open exceptions: {len(result.exceptions)}   (top 8 by severity)")
    for e in result.exceptions[:8]:
        say(f"    [{e.severity:>3}] {e.reason.value:<28} {e.entity_id:<18} {e.detail[:52]}")
    say("=" * 78)
    say(f"  artifacts -> {a.out}/run.json, exceptions.csv, matches.csv, close_report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
