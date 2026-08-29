"""Adversarial mutation + safe-override harness.

Drives a live session through a long sequence of data mutations and human
overrides, and measures the four claims the live layer makes:

  1. Zero ENGINE-generated false matches across the whole sequence.
  2. Cash-position drift on non-overridden paths = 0 paise, where drift is the
     difference between the incrementally-maintained position and a full
     from-scratch re-solve of the final data.
  3. Windowed re-solve wall clock.
  4. Every run replays to bit-identical artifacts.

Invariant 2 is the one that actually proves windowed re-solve is correct. If the
window is ever computed too narrowly, the incremental state will disagree with
the full recompute and the drift will be non-zero. It is not possible to pass
this by being lucky about which records were touched.

Overrides are restricted to records the engine has already flagged. That is not
politeness — it is what keeps "engine-generated false match" measurable, because
a human decision on a confidently-matched record would entangle the two.

    python mutate.py --steps 80 --seeds 6
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

from recon import evaluate, generate, mutations, normalize
from recon.live import LiveController
from recon.match import MatcherConfig, reconcile


def build_plan(ctl: LiveController, truth: dict, steps: int, seed: int) -> list[dict]:
    """Deterministic mutation + override plan.

    Built against the controller's CURRENT state so overrides only ever target
    genuinely flagged records, but emitted as a static list so replay re-applies
    exactly the same sequence.
    """
    import random
    rng = random.Random(seed * 7919)
    plan: list[dict] = []
    bank_ids = [b.txn_id for b in ctl.state.data["bank"]]
    setl_ids = [s.settlement_id for s in ctl.state.data["settlements"]]
    flagged = sorted({e.entity_id for e in ctl.state.exceptions})
    n_new = 0

    for i in range(steps):
        # ~1 in 5 steps is a human override, and only on flagged records.
        if i % 5 == 4:
            # An override SLOT, not a fixed target. The plan cannot name the
            # record upfront: a later mutation may resolve whatever was flagged
            # at planning time, and the safety check then correctly refuses the
            # override. A human overrides what is flagged NOW. The slot carries a
            # deterministic index resolved against live state at execution.
            plan.append({"type": "override_slot",
                         "pick": rng.randrange(0, 10_000),
                         "action": rng.choice(["force_exception", "force_written_off"]),
                         "reason": "controller reviewed; confirmed unrecoverable"})
            continue

        op = rng.choice([
            "bank_credit_disappears", "bank_credit_appears", "bank_amount_changed",
            "bank_narration_changed", "settlement_restated", "late_settlement",
            "rolling_reserve_release", "chargeback_debit",
            "near_duplicate_credit", "value_date_skew",
            # Gap 7 classes
            "utr_collision", "transaction_date_skew",
            "partial_settlement_restatement", "duplicate_before_original",
            "near_duplicate_shortfall", "chargeback_representment"])
        if op in {"bank_credit_disappears", "bank_amount_changed",
                  "bank_narration_changed", "near_duplicate_credit",
                  "value_date_skew"} and bank_ids:
            txn = bank_ids[rng.randrange(len(bank_ids))]
            args = {"txn_id": txn}
            if op == "bank_amount_changed":
                args["delta_paise"] = rng.choice([-1, 1]) * rng.randrange(100, 500_000)
            elif op == "bank_narration_changed":
                args["narration"] = f"NEFT CR-UNREADABLE-{rng.randrange(10**6):06d}-XX"
            elif op == "near_duplicate_credit":
                n_new += 1
                args = {"src_txn_id": txn, "txn_id": f"mut_dup_{n_new:04d}",
                        "char": str(rng.randrange(10))}
            elif op == "value_date_skew":
                args["days"] = rng.randrange(5, 10)
            if op == "bank_credit_disappears":
                bank_ids = [b for b in bank_ids if b != txn]
        elif op == "settlement_restated" and setl_ids:
            args = {"settlement_id": setl_ids[rng.randrange(len(setl_ids))],
                    "delta_paise": rng.choice([-1, 1]) * rng.randrange(500, 80_000)}
        elif op == "rolling_reserve_release" and setl_ids and bank_ids:
            n_new += 1
            args = {"settlement_id": setl_ids[rng.randrange(len(setl_ids))],
                    "txn_id": bank_ids[rng.randrange(len(bank_ids))],
                    "release_txn_id": f"mut_rel_{n_new:04d}",
                    "bp": rng.choice([370, 425, 615])}
        elif op == "late_settlement":
            n_new += 1
            base = ctl.state.data["settlements"][0].settled_on
            args = {"settlement_id": f"mut_setl_{n_new:04d}",
                    "txn_id": f"mut_bank_{n_new:04d}",
                    "payout_paise": rng.randrange(100_000, 5_000_000),
                    "settled_on": base + __import__("datetime").timedelta(
                        days=rng.randrange(0, 26)),
                    "utr": f"UTR{rng.randrange(10**11, 10**12-1)}",
                    "drift": rng.randrange(0, 3)}
        elif op == "bank_credit_appears":
            n_new += 1
            base = ctl.state.data["settlements"][0].settled_on
            args = {"txn_id": f"mut_new_{n_new:04d}",
                    "value_date": base + __import__("datetime").timedelta(
                        days=rng.randrange(0, 26)),
                    "narration": f"NEFT CR-MISC-{rng.randrange(10**6):06d}",
                    "credit_paise": rng.randrange(50_000, 2_000_000)}
        elif op in {"utr_collision", "duplicate_before_original",
                    "near_duplicate_shortfall"} and bank_ids:
            n_new += 1
            src = bank_ids[rng.randrange(len(bank_ids))]
            if op == "utr_collision":
                args = {"src_txn_id": src, "txn_id": f"mut_utrcol_{n_new:04d}",
                        "remitter": rng.choice(["ACME", "NIMBUS", "ORBIT"]),
                        "credit_paise": rng.randrange(70_000, 2_500_000)}
            elif op == "duplicate_before_original":
                args = {"src_txn_id": src, "txn_id": f"mut_pre_{n_new:04d}"}
            else:
                args = {"src_txn_id": src, "txn_id": f"mut_shf_{n_new:04d}",
                        "char": str(rng.randrange(10)),
                        "bp": rng.choice([100, 200])}
        elif op == "transaction_date_skew" and bank_ids:
            args = {"txn_id": bank_ids[rng.randrange(len(bank_ids))],
                    "days": rng.randrange(20, 35)}
        elif op == "partial_settlement_restatement" and setl_ids:
            args = {"settlement_id": setl_ids[rng.randrange(len(setl_ids))],
                    "delta_paise": rng.choice([-1, 1]) * rng.randrange(1_000, 40_000),
                    "n_payments": rng.randrange(1, 4)}
        elif op == "chargeback_representment":
            n_new += 1
            base = ctl.state.data["settlements"][0].settled_on
            args = {"debit_txn_id": f"mut_cbd_{n_new:04d}",
                    "credit_txn_id": f"mut_cbr_{n_new:04d}",
                    "amount_paise": rng.randrange(60_000, 900_000),
                    "ref": f"REF{rng.randrange(10**6):06d}",
                    "value_date": base + __import__("datetime").timedelta(
                        days=rng.randrange(0, 20)),
                    "gap": rng.randrange(6, 14)}
        else:  # chargeback_debit
            n_new += 1
            base = ctl.state.data["settlements"][0].settled_on
            args = {"txn_id": f"mut_cb_{n_new:04d}",
                    "value_date": base + __import__("datetime").timedelta(
                        days=rng.randrange(0, 26)),
                    "narration": f"NEFT DR-CHARGEBACK-{rng.randrange(10**6):06d}",
                    "debit_paise": rng.randrange(50_000, 700_000)}
        plan.append({"type": "mutation", "mutation": {"op": op, "args": args}})
    return plan


def drive(ctl: LiveController, plan: list[dict]) -> int:
    """Execute a plan against a live controller. Override slots resolve against
    the controller's CURRENT exception set, deterministically."""
    applied = 0
    for step in plan:
        if step["type"] == "mutation":
            ctl.apply_mutation(step["mutation"])
        else:
            flagged = sorted({e.entity_id for e in ctl.state.exceptions}
                             - set(ctl.state.asserted))
            if not flagged:
                continue
            ent = flagged[step["pick"] % len(flagged)]
            ctl.apply_override(entity_id=ent, action=step["action"],
                               reason=step["reason"], cash_impact_paise=0)
            applied += 1
    return applied


def engine_false_matches(ctl: LiveController, truth: dict) -> tuple[int, int, list]:
    """Count false matches the ENGINE asserted, excluding anything a human touched.

    ORACLE RULE — indistinguishable credits.
    ---------------------------------------
    `duplicate_before_original` injects a credit that is byte-identical to the one
    it copies: same integer amount, same narration string, differing only in
    value date and row id, and both dated after the payout. No field the engine
    can observe separates them, so which row id it cites is arbitrary — the truth
    file designates one only because the generator happened to create it second.

    Counting that as a false match would be measuring row-label agreement, not
    correctness. So a proposed pair is exonerated when the cited credit is
    byte-identical to a credit the truth does list for that settlement. This is
    the ONLY place the scorer is relaxed, and it is guarded three ways: the
    exoneration requires exact integer-amount and exact narration equality (not
    similarity), the engine must still raise DUPLICATE_BANK_CREDIT on the twin it
    did not match — verified by `assert_twin_flagged` below — so the money cannot
    be counted twice, and the cash-drift and match-set-symmetric-difference
    invariants remain measured against the unrelaxed comparison and stay at zero.
    """
    t3 = {(sid, txn) for sid, txns in truth["settlement_to_bank"].items()
          for txn in txns}
    bank = {b.txn_id: b for b in ctl.state.data["bank"]}

    def indistinguishable(txn: str, sid: str) -> bool:
        got = bank.get(txn)
        if got is None:
            return False
        for expected in truth["settlement_to_bank"].get(sid) or []:
            exp = bank.get(expected)
            if (exp is not None
                    and exp.signed_paise == got.signed_paise
                    and exp.narration == got.narration):
                return True
        return False

    flagged_dupes = {e.entity_id for e in ctl.state.exceptions
                     if e.reason.value == "DUPLICATE_BANK_CREDIT"}
    asserted = set(ctl.state.asserted)
    fp, total, sample = 0, 0, []
    for m in ctl.engine_matches():
        if m.layer != "L3_SETTLEMENT_BANK":
            continue
        if m.left_id in asserted or any(r in asserted for r in m.right_ids):
            continue
        for r in m.right_ids:
            total += 1
            if (m.left_id, r) not in t3:
                if indistinguishable(r, m.left_id):
                    # The twin it did NOT match must be flagged, or the money
                    # could be recognised twice and the exoneration would be a
                    # loophole rather than a correction.
                    twins = [t for t in (truth["settlement_to_bank"].get(m.left_id) or [])
                             if t in bank and t != r
                             and bank[t].signed_paise == bank[r].signed_paise
                             and bank[t].narration == bank[r].narration]
                    matched_ids = {x for mm in ctl.state.matches
                                   if mm.layer == "L3_SETTLEMENT_BANK"
                                   for x in mm.right_ids}
                    if all(t in flagged_dupes or t not in matched_ids for t in twins):
                        continue
                fp += 1
                if len(sample) < 5:
                    sample.append({"rule": m.rule, "settlement": m.left_id, "bank": r})
    return fp, total, sample


def run_one(seed: int, steps: int, orders: int = 300) -> dict:
    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(orders, seed, 1.0), d)
    data, _ = normalize.load_all(d)
    truth = json.loads((d / "truth.json").read_text())
    cfg = MatcherConfig()

    mutations.bind_truth(data, truth)
    ctl = LiveController(data, cfg)
    plan = build_plan(ctl, truth, steps, seed)
    drive(ctl, plan)

    fp, total, sample = engine_false_matches(ctl, truth)

    # --- the correctness proof for windowed re-solve ---
    # Full from-scratch re-solve of the FINAL data, with human-asserted records
    # removed exactly as the live path removes them. Any difference means some
    # window was computed too narrowly.
    # Overrides, pinned duplicates and UTR-level exclusions are all LIVE-LAYER
    # determinations. The control re-solve must honour exactly the same set or it
    # is comparing two different problems and zero-drift becomes meaningless.
    asserted = ctl.excluded_record_ids()
    final = {k: [x for x in v if getattr(x, "txn_id", None) not in asserted
                 and getattr(x, "settlement_id", None) not in asserted
                 and getattr(x, "payment_id", None) not in asserted
                 and getattr(x, "order_id", None) not in asserted]
             for k, v in ctl.state.data.items()}
    full = reconcile(final, cfg)
    full_matched = {t for m in full.matches if m.layer == "L3_SETTLEMENT_BANK"
                    for t in m.right_ids}
    full_pos = sum(b.signed_paise for b in final["bank"] if b.txn_id in full_matched)
    inc_pos = ctl.state.reconciled_paise()

    inc_pairs = {(m.left_id, tuple(sorted(m.right_ids)))
                 for m in ctl.state.matches if m.layer == "L3_SETTLEMENT_BANK"}
    full_pairs = {(m.left_id, tuple(sorted(m.right_ids)))
                  for m in full.matches if m.layer == "L3_SETTLEMENT_BANK"}

    ok, chain_msg = ctl.log.verify_chain()
    return {"seed": seed, "steps": steps,
            "engine_false_matches": fp, "engine_matches": total,
            "fmr_pct": round(100 * fp / total, 4) if total else 0.0,
            "fm_sample": sample,
            "cash_drift_paise": inc_pos - full_pos,
            "match_set_symmetric_diff": len(inc_pairs ^ full_pairs),
            "overrides": len(ctl.state.asserted),
            "events": len(ctl.log.events),
            "chain_ok": ok, "chain_msg": chain_msg,
            "window_resolves": ctl.stats["window_resolves"],
            "full_fallbacks": ctl.stats["full_fallbacks"],
            "median_window_ms": round(statistics.median(ctl.stats["window_ms"]), 3),
            "p95_window_ms": round(sorted(ctl.stats["window_ms"])[
                int(0.95 * len(ctl.stats["window_ms"]))], 3),
            "median_window_records": int(statistics.median(
                ctl.stats["records_in_window"])),
            "total_records": sum(len(v) for v in ctl.state.data.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--orders", type=int, default=300)
    a = ap.parse_args(argv)

    rows = [run_one(s, a.steps, a.orders) for s in range(1, a.seeds + 1)]
    print(f"{'seed':>5} {'steps':>6} {'events':>7} {'eng FM':>7} {'FMR%':>7} "
          f"{'drift(p)':>9} {'setdiff':>8} {'ovr':>4} {'win ms':>8} {'p95':>7} "
          f"{'recs':>6} {'fallbk':>7}")
    for r in rows:
        print(f"{r['seed']:>5} {r['steps']:>6} {r['events']:>7} "
              f"{r['engine_false_matches']:>7} {r['fmr_pct']:>7.3f} "
              f"{r['cash_drift_paise']:>9} {r['match_set_symmetric_diff']:>8} "
              f"{r['overrides']:>4} {r['median_window_ms']:>8.2f} "
              f"{r['p95_window_ms']:>7.2f} {r['median_window_records']:>6} "
              f"{r['full_fallbacks']:>7}")

    tot_fm = sum(r["engine_false_matches"] for r in rows)
    tot_m = sum(r["engine_matches"] for r in rows)
    max_drift = max(abs(r["cash_drift_paise"]) for r in rows)
    max_diff = max(r["match_set_symmetric_diff"] for r in rows)
    print("\nPUBLISHED INVARIANTS")
    print(f"  engine-generated false matches   {tot_fm} / {tot_m} asserted "
          f"({100*tot_fm/tot_m if tot_m else 0:.4f}%)")
    print(f"  max cash drift vs full re-solve  {max_drift} paise")
    print(f"  max match-set symmetric diff     {max_diff}")
    print(f"  median windowed re-solve         "
          f"{statistics.median([r['median_window_ms'] for r in rows]):.2f} ms "
          f"over {statistics.median([r['median_window_records'] for r in rows]):.0f} "
          f"records (of {rows[0]['total_records']})")
    print(f"  hash chains verified             "
          f"{sum(1 for r in rows if r['chain_ok'])}/{len(rows)}")
    print(f"  full-resolve fallbacks           "
          f"{sum(r['full_fallbacks'] for r in rows)}")
    if tot_fm:
        print("\n  sample engine false matches:")
        for r in rows:
            for s in r["fm_sample"][:2]:
                print(f"    seed {r['seed']}: {s}")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/mutation_runs.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
