"""Evaluation against labelled ground truth.

Design notes that matter more than the code:

1. Match rate alone is a vanity metric. A system that matches everything at 70%
   precision is far worse than one that matches 80% at 99.9%, because in
   reconciliation a false match is expensive (money booked against the wrong
   payout, a real discrepancy hidden) while an exception is merely work. So the
   headline number here is always a pair: coverage AND false-match rate.

2. Accuracy is reported per difficulty tier. A single average lets easy rows
   subsidise hard ones and is the standard way hackathon numbers lie.

3. The abstention curve is computed by sweeping the confidence threshold over a
   single reconciliation run. It answers the question a finance lead will
   actually ask: "at what level of automation does this stop being safe?"
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Match, Reason


@dataclass
class Scores:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_match_rate(self) -> float:
        return self.fp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "false_match_rate": round(self.false_match_rate, 4)}


def _l1_pairs(matches: list[Match]) -> set[tuple[str, str]]:
    return {(m.left_id, m.right_ids[0]) for m in matches if m.layer == "L1_ORDER_PAYMENT"}


def _l3_pairs(matches: list[Match]) -> set[tuple[str, str]]:
    """Flatten N:1 settlement<->bank links into pairs. A merged credit linking
    two settlements produces two pairs; a split payout produces two pairs. This
    grades partial credit honestly instead of scoring whole groups all-or-nothing."""
    out = set()
    for m in matches:
        if m.layer != "L3_SETTLEMENT_BANK":
            continue
        for r in m.right_ids:
            out.add((m.left_id, r))
    return out


def score_matches(matches: list[Match], truth: dict[str, Any]) -> dict[str, Any]:
    # ---- L1 ----
    t1 = {(o, p) for o, p in truth["order_to_payment"].items()}
    p1 = _l1_pairs(matches)
    s1 = Scores(len(p1 & t1), len(p1 - t1), len(t1 - p1))

    # ---- L3 ----
    t3 = {(sid, txn) for sid, txns in truth["settlement_to_bank"].items() for txn in txns}
    p3 = _l3_pairs(matches)
    s3 = Scores(len(p3 & t3), len(p3 - t3), len(t3 - p3))

    # ---- per-tier L3 ----
    tiers = {k: int(v) for k, v in truth["tiers"].items()}
    per_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted(set(tiers.values())):
        txns = {t for t, v in tiers.items() if v == tier}
        tt = {(s, t) for (s, t) in t3 if t in txns}
        pp = {(s, t) for (s, t) in p3 if t in txns}
        st = Scores(len(pp & tt), len(pp - tt), len(tt - pp))
        per_tier[f"tier_{tier}"] = {**st.as_dict(), "truth_pairs": len(tt)}

    return {"L1_order_payment": s1.as_dict(),
            "L3_settlement_bank": s3.as_dict(),
            "L3_by_difficulty_tier": per_tier,
            "false_matches_sample": [
                {"settlement_id": s, "bank_txn": t} for s, t in sorted(p3 - t3)[:10]],
    }


def score_exceptions(exceptions, truth: dict[str, Any]) -> dict[str, Any]:
    expected = {(e["entity_id"], e["reason"]) for e in truth["expected_exceptions"]}
    predicted = {(e.entity_id, e.reason.value) for e in exceptions
                 if e.reason is not Reason.BELOW_CONFIDENCE_THRESHOLD}
    abstentions = sum(1 for e in exceptions if e.reason is Reason.BELOW_CONFIDENCE_THRESHOLD)

    s = Scores(len(predicted & expected), len(predicted - expected), len(expected - predicted))

    # Same, excluding the trivially-detectable class. ORDER_NO_PAYMENT is just
    # "status == failed"; leaving it in the headline would flatter the number.
    triv = "ORDER_NO_PAYMENT"
    exp_h = {x for x in expected if x[1] != triv}
    pre_h = {x for x in predicted if x[1] != triv}
    sh = Scores(len(pre_h & exp_h), len(pre_h - exp_h), len(exp_h - pre_h))

    by_reason: dict[str, dict[str, int]] = {}
    for reason in sorted({r for _, r in expected} | {r for _, r in predicted}):
        e_r = {x for x in expected if x[1] == reason}
        p_r = {x for x in predicted if x[1] == reason}
        by_reason[reason] = {"expected": len(e_r), "predicted": len(p_r),
                             "correct": len(e_r & p_r),
                             "missed": len(e_r - p_r), "spurious": len(p_r - e_r)}

    return {"all_classes": s.as_dict(),
            "excluding_trivial_ORDER_NO_PAYMENT": sh.as_dict(),
            "abstentions_routed_to_human": abstentions,
            "by_reason_code": by_reason,
            "missed_sample": [{"entity": a, "reason": b} for a, b in sorted(expected - predicted)[:10]],
            "spurious_sample": [{"entity": a, "reason": b} for a, b in sorted(predicted - expected)[:10]]}


def abstention_curve(all_matches: list[Match], truth: dict[str, Any],
                     thresholds=(0.0, 0.35, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99)) -> list[dict[str, Any]]:
    """Sweep the confidence threshold over one run.

    Reads as: 'automate everything above X, send the rest to a human, and here is
    the false-match rate you are accepting for that level of automation.'
    """
    t3 = {(sid, txn) for sid, txns in truth["settlement_to_bank"].items() for txn in txns}
    t1 = {(o, p) for o, p in truth["order_to_payment"].items()}
    truth_all = t1 | t3
    rows = []
    for tau in thresholds:
        kept = [m for m in all_matches if m.confidence >= tau]
        pred = _l1_pairs(kept) | _l3_pairs(kept)
        s = Scores(len(pred & truth_all), len(pred - truth_all), len(truth_all - pred))
        rows.append({"threshold": tau, "asserted": len(pred),
                     "coverage": round(len(pred & truth_all) / len(truth_all), 4) if truth_all else 0.0,
                     "precision": round(s.precision, 5),
                     "false_match_rate": round(s.false_match_rate, 5),
                     "sent_to_human": len(all_matches) - len(kept)})
    return rows


def match_rate(result, truth: dict[str, Any]) -> dict[str, Any]:
    """The headline the track asks for, stated without spin."""
    t3 = {(sid, txn) for sid, txns in truth["settlement_to_bank"].items() for txn in txns}
    p3 = _l3_pairs(result.matches)
    settlements_truth = {s for s, _ in t3}
    settlements_matched = {s for s, _ in (p3 & t3)}
    return {
        "settlement_bank_pairs_in_truth": len(t3),
        "settlement_bank_pairs_asserted": len(p3),
        "settlement_bank_pairs_correct": len(p3 & t3),
        "settlements_expected_to_land": len(settlements_truth),
        "settlements_correctly_reconciled": len(settlements_matched),
        "settlement_match_rate_pct": round(
            100 * len(settlements_matched) / len(settlements_truth), 2) if settlements_truth else 0.0,
        "false_match_rate_pct": round(
            100 * len(p3 - t3) / len(p3), 3) if p3 else 0.0,
    }
