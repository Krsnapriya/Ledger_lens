"""The reconciliation engine.

The central claim of this project: settlement reconciliation is not a
classification problem, it is a *constrained assignment* problem.

Row-by-row greedy matching fails on the cases that actually cost finance teams
time, because those cases are N:1. A bank credit is the sum of many captures
minus fees minus GST minus refunds minus withholding; two payouts arrive
consolidated as one credit; one payout arrives split across two. Asking a model
"do these two rows match?" cannot express that, and asking it to add the numbers
up invites a hallucinated total that balances.

So the engine is layered, and the layers escalate in cost and decrease in
certainty:

  L1  order  <-> payment       identifier joins, then optimal assignment
  L2  payment fee/GST + batch arithmetic   pure integer rules, no matching
  L3  settlement <-> bank      identifier -> amount+date -> fuzzy -> subset-sum

Every asserted match carries the rule that produced it, a confidence, the
evidence, and the candidates that were rejected. That last field is what makes
this auditable: a match you cannot argue against is a match you cannot trust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import subsetsum
from .ai import SemanticResolver, cosine, extract_deterministic
from .models import BankLine, Exception_, Match, Payment, Reason, Settlement
from .money import pct_of

INFEASIBLE = 1e6


@dataclass
class MatcherConfig:
    """Every knob is explicit so the ablation can turn layers off one at a time."""

    exact_ids: bool = True
    tolerance_window: bool = True
    fuzzy_narration: bool = True
    subset_sum: bool = True
    use_llm: bool = False

    confidence_threshold: float = 0.70
    date_window_days: int = 4
    amount_tolerance_paise: int = 100          # ₹1.00 absolute
    short_credit_tolerance_bp: int = 500       # 5% — beyond this it is not "short", it is split
    # Withholding happens at rates, not at arbitrary amounts. Used as a hard
    # constraint by R3.4b instead of a tolerance band. See run-2 regression note.
    plausible_withholding_bp: tuple[int, ...] = (10, 100, 200, 500)
    max_subset_size: int = 8                   # merge bound: raised 3 -> 8 via MITM
    # Asymmetric on purpose. A payout batch CONSOLIDATING 8 settlements is
    # routine end-of-day behaviour; a single payout ARRIVING in 7 separate bank
    # credits is not. Measured: every 7-way split the engine proposed under the
    # unmodeled-anomaly suite was false, produced by scavenging duplicate and
    # unrelated credits out of a large pool. The bound encodes the asymmetry of
    # the underlying process rather than pretending the two directions are alike.
    max_split_size: int = 5
    # Single-claim / contingency rule. A subset-sum solution is only trustworthy
    # if it would still be the ONLY solution had earlier rules not consumed part
    # of the pool. Uniqueness is currently evaluated over the unmatched residual,
    # so a solution can be "unique" purely because its true partner was matched
    # first — which is exactly how a phantom credit substitutes for a real half
    # of a split. Checked against the FULL window, matched lines included.
    refuse_contingent_subsets: bool = True
    subset_sum_pool_cap: int = 40              # beyond this, refuse and except
    # A consolidated payout covers a RANGE of settlement dates, so the subset-sum
    # stage needs an asymmetric, wider look-back than the 1:1 rules. Measured:
    # with the symmetric +/-4 day window an 8-way consolidation could never be
    # found at any k, because its earliest members were 8 days out and never
    # entered the candidate pool. The cardinality bound was not the binding
    # constraint; the window was.
    subset_sum_lookback_days: int = 16
    subset_sum_lookahead_days: int = 4
    # Hop limit for the live layer's record-set closure over prior match edges.
    # Measured on the adversarial suite; beyond this the window is declared
    # unprovable and the caller falls back to a full re-solve rather than
    # reconciling a set it cannot vouch for.
    window_hop_limit: int = 4
    # A split whose tranches span more than this is flagged: it usually means a
    # rolling reserve or holdback, i.e. working capital held back.
    split_long_window_days: int = 3
    # Duplicate CLUSTER detection, run before any matching rule. Thresholds are
    # measured, not guessed: across 14 seeds, legitimate split tranches of one
    # payout reach at most 0.7317 narration similarity (each tranche carries its
    # own UTR), while a near-duplicate differing by a single character never
    # falls below 0.8800. 0.80 sits in the gap with margin on both sides.
    duplicate_similarity: float = 0.80
    duplicate_date_days: int = 1
    fuzzy_min_similarity: float = 0.55


@dataclass
class Result:
    matches: list[Match] = field(default_factory=list)
    exceptions: list[Exception_] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    ai_stats: Any = None


def _within(a: int, b: int, tol: int) -> bool:
    return abs(a - b) <= tol


def _expected_narration(s: Settlement) -> str:
    """What a clean narration for this payout would have looked like. The fuzzy
    layer scores the observed narration against this synthesised reference."""
    return f"NEFT CR-GATEWAYPAY-{s.settlement_id.upper()}-{s.utr or 'NOUTR'}-SETTLEMENT"


# --------------------------------------------------------------------------
# L1 — order <-> payment
# --------------------------------------------------------------------------
def match_orders_payments(orders, payments, cfg: MatcherConfig) -> tuple[list[Match], list[Exception_]]:
    matches: list[Match] = []
    excs: list[Exception_] = []
    by_pid = {p.payment_id: p for p in payments}
    used_p: set[str] = set()
    used_o: set[str] = set()

    captured = [o for o in orders if o.status == "captured"]

    if cfg.exact_ids:
        # R1.1 order carries the gateway payment id
        for o in captured:
            if o.gateway_payment_id and o.gateway_payment_id in by_pid and o.gateway_payment_id not in used_p:
                matches.append(Match("L1_ORDER_PAYMENT", o.order_id, [o.gateway_payment_id],
                                     "R1.1_EXACT_PAYMENT_ID", 1.0,
                                     {"gateway_payment_id": o.gateway_payment_id}))
                used_p.add(o.gateway_payment_id)
                used_o.add(o.order_id)

        # R1.2 gateway carries our order ref
        oid_set = {o.order_id for o in captured}
        for p in payments:
            if p.payment_id in used_p or not p.order_ref:
                continue
            if p.order_ref in oid_set and p.order_ref not in used_o:
                matches.append(Match("L1_ORDER_PAYMENT", p.order_ref, [p.payment_id],
                                     "R1.2_EXACT_ORDER_REF", 1.0, {"order_ref": p.order_ref}))
                used_p.add(p.payment_id)
                used_o.add(p.order_ref)

        # R1.3 truncated order ref, accepted only when the prefix is unambiguous
        rem_o = [o for o in captured if o.order_id not in used_o]
        for p in payments:
            if p.payment_id in used_p or not p.order_ref:
                continue
            hits = [o for o in rem_o if o.order_id.startswith(p.order_ref) and o.order_id not in used_o]
            if len(hits) == 1:
                matches.append(Match("L1_ORDER_PAYMENT", hits[0].order_id, [p.payment_id],
                                     "R1.3_PREFIX_ORDER_REF", 0.90,
                                     {"prefix": p.order_ref, "resolved_to": hits[0].order_id}))
                used_p.add(p.payment_id)
                used_o.add(hits[0].order_id)
            elif len(hits) > 1:
                excs.append(Exception_("payment", p.payment_id, Reason.ORDER_PAYMENT_AMBIGUOUS,
                                       f"truncated ref {p.order_ref!r} matches {len(hits)} orders",
                                       {"order_ref": p.order_ref},
                                       [{"order_id": h.order_id} for h in hits[:5]]))
                used_p.add(p.payment_id)

    # R1.4 residual: optimal assignment on (amount, date proximity)
    if cfg.tolerance_window:
        rem_o = [o for o in captured if o.order_id not in used_o]
        rem_p = [p for p in payments if p.payment_id not in used_p]
        if rem_o and rem_p:
            C = np.full((len(rem_o), len(rem_p)), INFEASIBLE)
            for i, o in enumerate(rem_o):
                for j, p in enumerate(rem_p):
                    if o.amount_paise != p.gross_paise:
                        continue
                    dd = abs((o.created_on - p.captured_on).days)
                    if dd > cfg.date_window_days:
                        continue
                    C[i, j] = dd * 0.1          # prefer same-day
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= INFEASIBLE:
                    continue
                o, p = rem_o[i], rem_p[j]
                # count equally-good alternatives — this drives the confidence
                alts = int((C[i, :] == C[i, j]).sum())
                conf = 0.88 if alts == 1 else max(0.35, 0.88 / alts)
                matches.append(Match("L1_ORDER_PAYMENT", o.order_id, [p.payment_id],
                                     "R1.4_AMOUNT_DATE_ASSIGNMENT", conf,
                                     {"amount_paise": o.amount_paise,
                                      "date_gap_days": abs((o.created_on - p.captured_on).days),
                                      "equally_good_alternatives": alts}))
                used_o.add(o.order_id)
                used_p.add(p.payment_id)

    for o in captured:
        if o.order_id not in used_o:
            excs.append(Exception_("order", o.order_id, Reason.ORDER_NO_PAYMENT,
                                   "captured order with no matching gateway capture",
                                   {"amount_paise": o.amount_paise,
                                    "gateway_payment_id": o.gateway_payment_id}))
    for p in payments:
        if p.payment_id not in used_p:
            excs.append(Exception_("payment", p.payment_id, Reason.PAYMENT_NO_ORDER,
                                   "gateway capture with no internal order",
                                   {"gross_paise": p.gross_paise, "order_ref": p.order_ref}))
    return matches, excs


# --------------------------------------------------------------------------
# L2 — arithmetic. No AI, no matching, no tolerance beyond rounding.
# --------------------------------------------------------------------------
def check_arithmetic(payments: list[Payment], settlements: list[Settlement],
                     refunds, mdr_bp: int = 200, gst_bp: int = 1800) -> list[Exception_]:
    excs: list[Exception_] = []
    for p in payments:
        exp_fee = pct_of(p.gross_paise, mdr_bp)
        if abs(p.fee_paise - exp_fee) > 1:
            excs.append(Exception_("payment", p.payment_id, Reason.FEE_RATE_MISMATCH,
                                   "fee does not equal contracted MDR",
                                   {"gross_paise": p.gross_paise, "fee_charged": p.fee_paise,
                                    "fee_expected": exp_fee,
                                    "delta_paise": p.fee_paise - exp_fee}, confidence=1.0))
        exp_gst = pct_of(p.fee_paise, gst_bp)
        if abs(p.gst_paise - exp_gst) > 1:
            excs.append(Exception_("payment", p.payment_id, Reason.GST_MISMATCH,
                                   "GST on fee is not 18%",
                                   {"fee_paise": p.fee_paise, "gst_charged": p.gst_paise,
                                    "gst_expected": exp_gst}, confidence=1.0))
        if p.net_paise != p.gross_paise - p.fee_paise - p.gst_paise:
            excs.append(Exception_("payment", p.payment_id, Reason.NET_ARITHMETIC_MISMATCH,
                                   "net != gross - fee - gst",
                                   {"gross": p.gross_paise, "fee": p.fee_paise,
                                    "gst": p.gst_paise, "net": p.net_paise}, confidence=1.0))

    for s in settlements:
        members = [p for p in payments if p.settlement_id == s.settlement_id]
        rf = [r for r in refunds if r.settlement_id == s.settlement_id]
        expected = sum(p.net_paise for p in members) - sum(r.amount_paise for r in rf)
        if members and s.payout_paise != expected:
            excs.append(Exception_("settlement", s.settlement_id, Reason.BATCH_ARITHMETIC_MISMATCH,
                                   "declared payout != sum(net) - refunds",
                                   {"declared_paise": s.payout_paise, "expected_paise": expected,
                                    "delta_paise": s.payout_paise - expected,
                                    "member_payments": len(members), "refunds_applied": len(rf)},
                                   confidence=1.0))
    return excs


# --------------------------------------------------------------------------
# L3 — settlement <-> bank. The hard layer.
# --------------------------------------------------------------------------
def match_settlements_bank(settlements: list[Settlement], bank: list[BankLine],
                           cfg: MatcherConfig, resolver: SemanticResolver
                           ) -> tuple[list[Match], list[Exception_]]:
    matches: list[Match] = []
    excs: list[Exception_] = []
    lines = [b for b in bank if b.signed_paise != 0]
    by_sid = {s.settlement_id: s for s in settlements}
    used_s: set[str] = set()
    used_b: set[str] = set()
    # identifier found in narration but the amount refused it — a strong prior
    # for the split/merge solver rather than a match
    split_hints: dict[str, str] = {}

    def in_window(s: Settlement, b: BankLine) -> bool:
        return abs((b.value_date - s.settled_on).days) <= cfg.date_window_days

    def in_consolidation_window(s: Settlement, b: BankLine) -> bool:
        """Asymmetric window for the subset-sum stage only.

        A consolidated credit lands at or after the LAST settlement in the group,
        so the group extends backwards from the credit, not symmetrically around
        it. Widening this window is safe precisely because the uniqueness gate
        refuses any target reachable more than one way — a bigger pool produces
        more abstentions, never more false matches.
        """
        gap = (b.value_date - s.settled_on).days
        return -cfg.subset_sum_lookahead_days <= gap <= cfg.subset_sum_lookback_days

    def duplicate_rank(b: BankLine) -> tuple[int, date, str]:
        """Ordering for choosing which member of a duplicate cluster to KEEP.

        "Earliest wins" was wrong. A phantom that posts a day BEFORE the credit it
        copies then becomes the survivor, and the real credit is retired — six of
        the eight false matches introduced by the expanded generator came from
        exactly this. It is not a tie-break problem, it is a missing domain rule:
        **a payout cannot land before it is instructed.** A credit naming a
        settlement but dated before that settlement's payout date cannot be that
        payout's landing, so it sorts last and loses the cluster.
        """
        ex = extract_deterministic(b.narration)
        st = by_sid.get(ex.settlement_id) if ex.settlement_id else None
        impossible = 1 if (st is not None and b.value_date < st.settled_on) else 0
        return (impossible, b.value_date, b.txn_id)

    def implied_rate_ok(declared: int, credited: int) -> int | None:
        """Does the shortfall correspond to a plausible withholding rate?

        Used as a hard gate anywhere the engine is tempted to accept a match
        whose amount does not tie out. Withholding happens at rates; coincidence
        does not. Returns the rate in basis points, or None.
        """
        delta = declared - credited
        if declared <= 0 or delta <= 0:
            return None            # withholding applies to positive payouts only
        bp = delta * 10000 / max(1, declared)
        for cand in cfg.plausible_withholding_bp:
            if abs(bp - cand) <= 2:
                return cand
        return None

    # ---- R3.0 retire duplicate CLUSTERS among still-unmatched credits ----
    # The existing duplicate pass compares an unmatched credit against credits
    # that are ALREADY matched. That is blind to the case where neither member
    # of a duplicate pair has been matched yet — which is exactly what the
    # mutation harness produces when it injects a near-duplicate whose narration
    # differs by one character, leaving the UTR intact. Both credits then claim
    # the same settlement legitimately, and whichever rule fires first picks one
    # by sort order. Eight of sixteen adversarial false matches came from this.
    #
    # So duplicates are clustered and retired BEFORE any rule runs. Grouping is
    # on exact integer amount, then a date proximity gate, then narration
    # similarity above a measured threshold. The earliest line by (date, id)
    # survives; the rest are retired as DUPLICATE_BANK_CREDIT and never enter a
    # candidate pool. Order-independent and deterministic.
    if cfg.exact_ids:
        # ---- R3.0a same-UTR re-postings, regardless of amount ----
        # A UTR is a unique bank transaction reference: two credits carrying the
        # same one are the same instrument, and a divergence in amount means one
        # of them is stale, NOT that they are different payments.
        #
        # The amount-keyed pass below cannot see this. Traced from the adversarial
        # suite: a duplicate is injected, then the ORIGINAL credit is restated by
        # a later mutation. The stale duplicate keeps the old amount, which still
        # ties out to the payout exactly, while the real credit no longer does.
        # Grouping on exact amount stopped clustering them, and the engine
        # confidently attributed the payout to a record that no longer reflects
        # any money movement. Five of the eight remaining false matches.
        by_utr: dict[str, list[BankLine]] = {}
        for b in lines:
            if b.txn_id in used_b:
                continue
            ex = extract_deterministic(b.narration)
            if ex.utr:
                by_utr.setdefault(ex.utr, []).append(b)
        for utr in sorted(by_utr):
            group = sorted(by_utr[utr], key=duplicate_rank)
            if len(group) < 2:
                continue
            keep = group[0]
            for other in group[1:]:
                if abs((other.value_date - keep.value_date).days) > cfg.duplicate_date_days:
                    continue
                excs.append(Exception_("bank_line", other.txn_id,
                                       Reason.DUPLICATE_BANK_CREDIT,
                                       "same bank UTR as an earlier unreconciled credit; "
                                       "one of the two postings is stale",
                                       {"amount_paise": other.signed_paise,
                                        "duplicate_of": keep.txn_id,
                                        "kept_amount_paise": keep.signed_paise,
                                        "amount_divergence_paise": other.signed_paise - keep.signed_paise,
                                        "shared_utr": utr,
                                        "date_gap_days": (other.value_date - keep.value_date).days,
                                        "narration": other.narration},
                                       confidence=0.97))
                used_b.add(other.txn_id)

        # ---- R3.0b same-amount near-identical narration ----
        buckets: dict[int, list[BankLine]] = {}
        for b in lines:
            if b.txn_id not in used_b:
                buckets.setdefault(b.signed_paise, []).append(b)
        for amt in sorted(buckets):
            group = sorted(buckets[amt], key=duplicate_rank)
            if len(group) < 2:
                continue
            # PAIRWISE, not star-shaped around the earliest member.
            #
            # The first version compared every line only against group[0]. When a
            # bucket holds three lines — the two legitimate halves of a split plus
            # a phantom copy of one of them — the earliest line is the OTHER half,
            # which is dissimilar to both. The one pair that actually matched was
            # never compared to itself, and the phantom survived to substitute
            # into the split. That was the last engine false match in the suite.
            for i in range(len(group)):
                keep = group[i]
                if keep.txn_id in used_b:
                    continue
                keep_ex = extract_deterministic(keep.narration)
                for j in range(i + 1, len(group)):
                    other = group[j]
                    if other.txn_id in used_b:
                        continue
                    if abs((other.value_date - keep.value_date).days) > cfg.duplicate_date_days:
                        continue
                    sim = cosine(other.narration, keep.narration)
                    if sim < cfg.duplicate_similarity:
                        continue
                    other_ex = extract_deterministic(other.narration)
                    if (keep_ex.settlement_id and other_ex.settlement_id
                            and keep_ex.settlement_id != other_ex.settlement_id):
                        continue
                    # The UTR veto is conditional, not absolute.
                    #
                    # It exists because two unrelated consolidated payouts narrate
                    # as CONSOLIDATED-<UTR> with no settlement id and read 0.9767
                    # alike; there, differing UTRs are the only discriminator.
                    #
                    # But written as an unconditional override it was too strong.
                    # A split tranche narrates as ...-PART-<UTR>, so a one-character
                    # corruption at the END of the narration lands INSIDE the UTR.
                    # The phantom then carries a different UTR, the veto fires, and
                    # the pair is never clustered — though both name the same
                    # settlement, carry the identical amount, and read 97% alike.
                    # That is corruption evidence, not evidence of a distinct payment.
                    corroborated = bool(keep_ex.settlement_id
                                        and keep_ex.settlement_id == other_ex.settlement_id)
                    if (not corroborated and keep_ex.utr and other_ex.utr
                            and keep_ex.utr != other_ex.utr):
                        continue
                    excs.append(Exception_("bank_line", other.txn_id,
                                           Reason.DUPLICATE_BANK_CREDIT,
                                           "identical amount and near-identical narration "
                                           "to an earlier unreconciled credit",
                                           {"amount_paise": other.signed_paise,
                                            "duplicate_of": keep.txn_id,
                                            "narration_similarity": round(sim, 4),
                                            "shared_settlement_id": keep_ex.settlement_id,
                                            "utr_corrupted": bool(
                                                corroborated and keep_ex.utr != other_ex.utr),
                                            "date_gap_days": (other.value_date - keep.value_date).days,
                                            "narration": other.narration},
                                           confidence=0.96))
                    used_b.add(other.txn_id)

        # ---- R3.0c near-duplicate short by a plausible withholding rate ----
        # A phantom can differ on BOTH axes at once: amount short by exactly a
        # withholding rate and one character of the UTR corrupted. That defeats
        # amount-keyed clustering (amounts differ), UTR clustering (UTRs differ),
        # and is shaped precisely to be accepted by the withholding-rate rule as
        # a legitimately-withheld payout. Clustered here on the signature itself:
        # same settlement id, high narration similarity, and an amount that is a
        # plausible rate below its sibling.
        openv = [b for b in lines if b.txn_id not in used_b and b.signed_paise > 0]
        for keep in sorted(openv, key=duplicate_rank):
            if keep.txn_id in used_b:
                continue
            k_ex = extract_deterministic(keep.narration)
            if not k_ex.settlement_id:
                continue
            for other in openv:
                if other.txn_id in used_b or other.txn_id == keep.txn_id:
                    continue
                if other.signed_paise >= keep.signed_paise:
                    continue
                if abs((other.value_date - keep.value_date).days) > cfg.duplicate_date_days:
                    continue
                o_ex = extract_deterministic(other.narration)
                if o_ex.settlement_id != k_ex.settlement_id:
                    continue
                rate = implied_rate_ok(keep.signed_paise, other.signed_paise)
                if rate is None:
                    continue
                if cosine(other.narration, keep.narration) < cfg.duplicate_similarity:
                    continue
                excs.append(Exception_("bank_line", other.txn_id,
                                       Reason.DUPLICATE_BANK_CREDIT,
                                       "near-duplicate of an earlier credit, short by a "
                                       "plausible withholding rate with a corrupted reference",
                                       {"amount_paise": other.signed_paise,
                                        "duplicate_of": keep.txn_id,
                                        "sibling_amount_paise": keep.signed_paise,
                                        "implied_rate_bp": rate,
                                        "shared_settlement_id": k_ex.settlement_id,
                                        "narration": other.narration},
                                       confidence=0.95))
                used_b.add(other.txn_id)

    # ---- R3.1 / R3.2 identifier in narration ----
    # Claims are collected first, then resolved. The original version iterated
    # bank lines and matched the first one carrying a given identifier, which is
    # first-wins on a coin flip when two lines carry the SAME identifier.
    #
    # Found by the mutation harness: a near-duplicate credit that alters the tail
    # of the narration leaves the UTR intact, so both the real credit and its
    # duplicate legitimately claim the same settlement. First-wins picked the
    # duplicate and produced the only engine false match in the suite. Now a
    # contested identifier is resolved deterministically to the earliest line and
    # the losers are retired as duplicates immediately, by identifier rather than
    # by narration similarity — which is strictly stronger evidence.
    if cfg.exact_ids:
        utr_index = {s.utr: s for s in settlements if s.utr}
        claims: dict[str, list[tuple[BankLine, str]]] = {}
        for b in lines:
            if b.txn_id in used_b:
                continue
            ex = extract_deterministic(b.narration)
            s = by_sid.get(ex.settlement_id) if ex.settlement_id else None
            rule = "R3.1_SETTLEMENT_ID_IN_NARRATION"
            if s is None and ex.utr and ex.utr in utr_index:
                s, rule = utr_index[ex.utr], "R3.2_UTR_MATCH"
            if s is None or s.settlement_id in used_s:
                continue
            claims.setdefault(s.settlement_id, []).append((b, rule))

        for sid in sorted(claims):
            s = by_sid[sid]
            contenders = sorted(claims[sid], key=lambda t: (t[0].value_date, t[0].txn_id))
            # A contender is a DUPLICATE only if it carries the identifier AND
            # ties out to the full payout. Two contenders that share the
            # identifier but each carry a fraction of it are split tranches, not
            # duplicates — retiring the second half as a duplicate cost 6 points
            # of match rate on the first attempt at this fix.
            ties = [(b, r) for b, r in contenders
                    if _within(b.signed_paise, s.payout_paise, cfg.amount_tolerance_paise)]
            non_ties = [(b, r) for b, r in contenders if (b, r) not in ties]

            if ties:
                b, rule = ties[0]
                for other, _ in ties[1:]:
                    if other.txn_id in used_b:
                        continue
                    excs.append(Exception_("bank_line", other.txn_id,
                                           Reason.DUPLICATE_BANK_CREDIT,
                                           "a second credit carries the same settlement "
                                           "identifier AND the same full payout amount",
                                           {"amount_paise": other.signed_paise,
                                            "duplicate_of": b.txn_id,
                                            "contested_settlement": sid,
                                            "narration": other.narration},
                                           confidence=0.98))
                    used_b.add(other.txn_id)
                for other, _ in non_ties:
                    split_hints[other.txn_id] = sid
                matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id, [b.txn_id],
                                     rule, 0.99,
                                     {"narration": b.narration, "amount_paise": b.signed_paise,
                                      "date_gap_days": (b.value_date - s.settled_on).days,
                                      "contested_by": len(ties) - 1}))
                used_s.add(s.settlement_id)
                used_b.add(b.txn_id)
                continue

            # No contender ties out. Try a withholding shortfall on the earliest;
            # everything else becomes a hint for the split solver.
            b, rule = contenders[0]
            delta = b.signed_paise - s.payout_paise
            if (rate := implied_rate_ok(s.payout_paise, b.signed_paise)) is not None:
                # Identifier is unambiguous but the money is short: withholding
                # applied outside the declared batch. This is a real match AND a
                # real exception; reporting only one of the two is wrong.
                matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id, [b.txn_id],
                                     rule + "_WITH_SHORTFALL", 0.95,
                                     {"declared_paise": s.payout_paise,
                                      "credited_paise": b.signed_paise, "delta_paise": delta}))
                excs.append(Exception_("settlement", s.settlement_id, Reason.AMOUNT_MISMATCH,
                                       "bank credit short of declared payout",
                                       {"declared_paise": s.payout_paise,
                                        "credited_paise": b.signed_paise, "delta_paise": delta,
                                        "implied_rate_bp": rate},
                                       confidence=0.95))
                used_s.add(s.settlement_id)
                used_b.add(b.txn_id)
                for other, _ in contenders[1:]:
                    split_hints[other.txn_id] = sid
            else:
                # Identifier present, amount does not tie out and the gap is not a
                # plausible withholding rate. Two live possibilities: a split
                # payout, or a corrupted narration whose id survived the garbling
                # as a DIFFERENT valid id ("SETL_0019" -> "SETL_0009"). An
                # identifier that survives syntactically has not necessarily
                # survived semantically, so it is downgraded to a hint and must
                # earn the match through the amount constraint instead.
                for other, _ in contenders:
                    split_hints[other.txn_id] = sid

    # ---- R3.3 / R3.4 amount + date, resolved as a constrained assignment ----
    # Narration similarity enters HERE, as a tie-break term inside the cost
    # matrix — not as a downstream rule.
    #
    # It was originally a downstream rule (R3.5) and it was measured at exactly
    # zero contribution across 80 runs. The reason is structural: that rule
    # required an exact amount match inside the date window, but any candidate
    # meeting that condition has already been consumed by this assignment step.
    # The fuzzy layer was unreachable code that would have shipped as "AI-powered
    # matching" on a slide.
    #
    # Narration only carries information when the amount key fails to
    # discriminate — i.e. when two settlements in the window share a payout
    # total. So that is where it belongs: breaking ties, not finding matches.
    if cfg.tolerance_window:
        rem_s = [s for s in settlements if s.settlement_id not in used_s]
        rem_b = [b for b in lines if b.txn_id not in used_b]
        if rem_s and rem_b:
            C = np.full((len(rem_s), len(rem_b)), INFEASIBLE)
            SIM = np.zeros((len(rem_s), len(rem_b)))
            for i, s in enumerate(rem_s):
                exp_narr = _expected_narration(s)
                for j, b in enumerate(rem_b):
                    if not _within(b.signed_paise, s.payout_paise, cfg.amount_tolerance_paise):
                        continue
                    if not in_window(s, b):
                        continue
                    gap = abs((b.value_date - s.settled_on).days)
                    if cfg.fuzzy_narration:
                        sim = resolver.similarity(b.narration, exp_narr)
                        SIM[i, j] = sim
                        C[i, j] = gap * 0.1 + (1.0 - sim) * 0.5
                    else:
                        C[i, j] = gap * 0.1
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= INFEASIBLE:
                    continue
                s, b = rem_s[i], rem_b[j]
                feas = np.where(C[i, :] < INFEASIBLE)[0]
                alts = int(len(feas))
                if alts == 1:
                    rule, conf = "R3.3_AMOUNT_DATE_UNIQUE", 0.93
                elif cfg.fuzzy_narration:
                    # Confidence is driven by the MARGIN between the chosen
                    # candidate and the runner-up, not by the raw score. A high
                    # similarity that every candidate shares is not evidence.
                    sims = sorted((float(SIM[i, k]) for k in feas), reverse=True)
                    margin = sims[0] - sims[1]
                    resolver.note_tiebreak(margin)
                    rule = "R3.4_NARRATION_TIEBREAK"
                    conf = min(0.92, 0.55 + 0.8 * margin)
                else:
                    rule, conf = "R3.4_AMOUNT_DATE_ASSIGNMENT", max(0.40, 0.93 / alts)
                matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id, [b.txn_id], rule, conf,
                                     {"amount_paise": b.signed_paise,
                                      "date_gap_days": (b.value_date - s.settled_on).days,
                                      "competing_candidates": alts,
                                      "narration_similarity": round(float(SIM[i, j]), 4),
                                      "narration": b.narration}))
                used_s.add(s.settlement_id)
                used_b.add(b.txn_id)

    def retire_duplicates() -> None:
        """Retire double-posted credits from the candidate pool.

        Ordering matters more than any individual rule here. A double-posted
        credit carries exactly the amount of a real payout, so every rule that
        widens a tolerance will happily consume it to "explain" some unrelated
        settlement.

        Found at difficulty 3.0: a payout that never landed and a duplicated
        credit differed by roughly 1%, so the withholding-rate rule linked them.
        Two real exceptions annihilated into one false match — the worst outcome
        available to this system, because it makes missing money look reconciled.

        Run TWICE: once before the speculative rules, and again afterwards, since
        a duplicate whose original was only matched by the late subset-sum stage
        is invisible on the first pass.
        """
        matched_lines = {b.txn_id: b for b in lines if b.txn_id in used_b}
        for b in [x for x in lines if x.txn_id not in used_b]:
            for mid, mb in matched_lines.items():
                if (b.signed_paise == mb.signed_paise
                        and abs((b.value_date - mb.value_date).days) <= 1
                        and cosine(b.narration, mb.narration) > 0.90):
                    excs.append(Exception_("bank_line", b.txn_id, Reason.DUPLICATE_BANK_CREDIT,
                                           "identical credit already reconciled",
                                           {"amount_paise": b.signed_paise,
                                            "duplicate_of": mid,
                                            "narration_similarity": round(cosine(b.narration, mb.narration), 4)},
                                           confidence=0.97))
                    used_b.add(b.txn_id)
                    break

    retire_duplicates()

    # ---- R3.4b near-amount assignment, gated on uniqueness ----
    # Found by run 1: a payout with no UTR, a narration that dropped the
    # settlement id, AND withholding applied outside the batch is invisible to
    # every rule above — no identifier survives and the amount is 1% short. It
    # was double-counted as SETTLEMENT_NOT_IN_BANK plus UNEXPLAINED_BANK_CREDIT,
    # which is the worst kind of wrong: two loud exceptions instead of one quiet
    # match with a flagged shortfall.
    #
    # The fix widens the amount band to the short-credit tolerance, but only
    # asserts when exactly ONE candidate is feasible in the window. With more
    # than one, confidence is deliberately driven below threshold so the match
    # is withdrawn and a human decides. Widening a tolerance without a
    # uniqueness gate is how a reconciliation engine starts inventing matches.
    # REGRESSION FIX (run 2): the first version of this rule accepted any credit
    # within a 5% band that was the unique candidate in the window. Across 40
    # seeds that single rule produced *every* false match in the system, up to
    # 9% FMR — a neighbouring payout of similar size lands inside a 5% band far
    # more often than intuition suggests. Buying recall with a wide tolerance is
    # exactly the trade this engine is supposed to refuse.
    #
    # The replacement uses a domain constraint instead of a tolerance: a shortfall
    # between a declared payout and a bank credit is not an arbitrary number. It
    # is withholding, and withholding happens at *rates*. So the delta must imply
    # a plausible rate (0.1 / 1 / 2 / 5%) to within 2bp. A coincidentally-similar
    # neighbouring payout does not differ by exactly 1.00%.
    if cfg.tolerance_window:
        rem_s = [s for s in settlements if s.settlement_id not in used_s]
        rem_b = [b for b in lines if b.txn_id not in used_b]

        for s in rem_s:
            if s.settlement_id in used_s:
                continue
            feasible = [(b, r) for b in rem_b
                        if b.txn_id not in used_b and in_window(s, b)
                        and (r := implied_rate_ok(s.payout_paise, b.signed_paise)) is not None]
            if not feasible:
                continue
            b, rate = min(feasible, key=lambda t: abs(t[0].signed_paise - s.payout_paise))
            # Bidirectional uniqueness: this credit must also have no other
            # settlement that could explain it at a plausible rate. One-way
            # uniqueness was what let the coincidences through.
            back = [s2 for s2 in rem_s if s2.settlement_id not in used_s and in_window(s2, b)
                    and implied_rate_ok(s2.payout_paise, b.signed_paise) is not None]
            delta = b.signed_paise - s.payout_paise
            unique = len(feasible) == 1 and len(back) == 1
            conf = 0.80 if unique else 0.45      # <threshold => withdrawn, human decides
            matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id, [b.txn_id],
                                 "R3.4b_WITHHOLDING_RATE_MATCH", conf,
                                 {"declared_paise": s.payout_paise,
                                  "credited_paise": b.signed_paise, "delta_paise": delta,
                                  "implied_rate_bp": rate,
                                  "forward_candidates": len(feasible),
                                  "reverse_candidates": len(back),
                                  "narration": b.narration}))
            if conf >= cfg.confidence_threshold:
                excs.append(Exception_("settlement", s.settlement_id, Reason.AMOUNT_MISMATCH,
                                       "bank credit short of declared payout (no identifier survived; matched "
                                       "on a plausible withholding rate, uniquely, within window)",
                                       {"declared_paise": s.payout_paise,
                                        "credited_paise": b.signed_paise, "delta_paise": delta,
                                        "implied_rate_bp": round(abs(delta) * 10000 / max(1, s.payout_paise))},
                                       confidence=conf))
                used_s.add(s.settlement_id)
                used_b.add(b.txn_id)

    # ---- R3.5 removed ----
    # A downstream fuzzy-narration rule used to live here. It was measured at
    # zero contribution across 80 runs and deleted rather than left in to make
    # the architecture diagram look better. Its function now lives in R3.4 as a
    # tie-break term. See the note above R3.3/R3.4.

    # ---- R3.6 / R3.7 subset-sum for merged and split payouts ----
    # Cardinality-constrained meet-in-the-middle, k <= 8. See recon/subsetsum.py.
    #
    # Two changes from the first version, both forced by measurement:
    #   1. The bound was 3. Once the generator produced realistic 3-8 way
    #      end-of-day consolidations, match rate fell to 37.74%. The bound was
    #      not a minor limitation, it was the dominant one.
    #   2. The old enumerator returned the FIRST subset that hit the target. If
    #      two different subsets both sum to a credit, that is a coin flip
    #      dressed as a match. Now a target reachable more than one way is
    #      refused outright and becomes an exception. Raising k without this
    #      gate would have bought recall with the false-match invariant.
    if cfg.subset_sum:
        # merged: one credit == sum of k settlements
        for b in [x for x in lines if x.txn_id not in used_b]:
            # Sign partition: a credit may only be explained by positive payouts
            # and a debit only by negative ones. Mixing signs lets a chargeback
            # debit be conscripted into explaining a payout, which explodes the
            # solution space and manufactures coincidences.
            sign = 1 if b.signed_paise > 0 else -1
            pool = [s for s in settlements
                    if s.settlement_id not in used_s and in_consolidation_window(s, b)
                    and (s.payout_paise > 0) == (sign > 0)]
            if len(pool) > cfg.subset_sum_pool_cap:
                # Hitting the cap is a REFUSAL, not a shrug. The first version
                # skipped the target silently, so it fell through to the generic
                # residue path and was reported as an ordinary unexplained credit
                # — an operator would have had no way to tell "no payout explains
                # this" from "there were too many candidates to decide safely".
                # Those demand completely different actions.
                excs.append(Exception_("bank_line", b.txn_id, Reason.POOL_CAP_EXCEEDED,
                                       f"{len(pool)} candidate settlements in the window "
                                       f"exceeds the safe cap of {cfg.subset_sum_pool_cap}; "
                                       "refusing rather than approximating",
                                       {"amount_paise": b.signed_paise,
                                        "candidates_in_window": len(pool),
                                        "cap": cfg.subset_sum_pool_cap,
                                        "narration": b.narration},
                                       confidence=0.0))
                used_b.add(b.txn_id)
                continue
            if not pool:
                continue
            unique, sol = subsetsum.is_unique(
                [s.payout_paise for s in pool], b.signed_paise,
                max_k=cfg.max_subset_size, tol=cfg.amount_tolerance_paise)
            if sol is None:
                continue
            group = [pool[i] for i in sol]
            if not unique:
                # Ambiguous: more than one combination explains this credit.
                excs.append(Exception_("bank_line", b.txn_id, Reason.AMBIGUOUS_SUBSET,
                                       f"{len(group)}-way consolidation is not unique; "
                                       "at least two different settlement groups sum to this credit",
                                       {"amount_paise": b.signed_paise,
                                        "one_candidate_group": [x.settlement_id for x in group],
                                        "group_size": len(group),
                                        "narration": b.narration},
                                       confidence=0.0))
                used_b.add(b.txn_id)
                continue
            # Confidence decays with group size: a 7-way coincidence is likelier
            # than a 2-way one, so a larger group is weaker evidence.
            if cfg.refuse_contingent_subsets:
                gids = {x.settlement_id for x in group}
                wanted = {x.payout_paise for x in group}
                shadows = [x for x in settlements
                           if x.settlement_id not in gids
                           and x.payout_paise in wanted
                           and in_consolidation_window(x, b)]
                if shadows:
                    excs.append(Exception_("bank_line", b.txn_id, Reason.DUPLICATE_CLAIM,
                                           f"{len(group)}-way consolidation is unique only because "
                                           f"an amount-identical settlement was already consumed",
                                           {"amount_paise": b.signed_paise,
                                            "proposed_group": [x.settlement_id for x in group],
                                            "shadow_candidates": sorted(
                                                x.settlement_id for x in shadows)[:5]},
                                           confidence=0.0))
                    used_b.add(b.txn_id)
                    continue
            conf = max(0.60, 0.95 - 0.04 * (len(group) - 2))
            for s in group:
                matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id, [b.txn_id],
                                     f"R3.6_SUBSET_SUM_MERGED_{len(group)}WAY", conf,
                                     {"merged_group": [x.settlement_id for x in group],
                                      "group_size": len(group),
                                      "group_total_paise": sum(x.payout_paise for x in group),
                                      "credit_paise": b.signed_paise,
                                      "uniqueness_verified": True}))
                used_s.add(s.settlement_id)
            used_b.add(b.txn_id)

        # split: one settlement == sum of k credits
        for s in [x for x in settlements if x.settlement_id not in used_s]:
            pool = [b for b in lines if b.txn_id not in used_b
                    and in_consolidation_window(s, b)
                    and (b.signed_paise > 0) == (s.payout_paise > 0)]
            if len(pool) > cfg.subset_sum_pool_cap:
                excs.append(Exception_("settlement", s.settlement_id,
                                       Reason.POOL_CAP_EXCEEDED,
                                       f"{len(pool)} candidate credits in the window "
                                       f"exceeds the safe cap of {cfg.subset_sum_pool_cap}; "
                                       "refusing rather than approximating",
                                       {"payout_paise": s.payout_paise,
                                        "candidates_in_window": len(pool),
                                        "cap": cfg.subset_sum_pool_cap},
                                       confidence=0.0))
                used_s.add(s.settlement_id)
                continue
            if not pool:
                continue
            hinted = [b for b in pool if split_hints.get(b.txn_id) == s.settlement_id]
            chosen, unique, source = None, False, pool
            # A narration that named this settlement but failed on amount is a
            # strong prior; try that restricted pool first.
            for cand_pool in ([hinted] if len(hinted) >= 2 else []) + [pool]:
                u, sol = subsetsum.is_unique(
                    [b.signed_paise for b in cand_pool], s.payout_paise,
                    max_k=cfg.max_split_size, tol=cfg.amount_tolerance_paise)
                if sol is not None:
                    chosen, unique, source = [cand_pool[i] for i in sol], u, cand_pool
                    break
            if chosen is None:
                continue
            if not unique:
                excs.append(Exception_("settlement", s.settlement_id, Reason.AMBIGUOUS_SUBSET,
                                       f"{len(chosen)}-way split is not unique; at least two "
                                       "different groups of credits sum to this payout",
                                       {"payout_paise": s.payout_paise,
                                        "one_candidate_group": [b.txn_id for b in chosen],
                                        "group_size": len(chosen)},
                                       confidence=0.0))
                used_s.add(s.settlement_id)
                continue
            conf = max(0.60, 0.95 - 0.06 * (len(chosen) - 2))
            matches.append(Match("L3_SETTLEMENT_BANK", s.settlement_id,
                                 [b.txn_id for b in chosen],
                                 f"R3.7_SUBSET_SUM_SPLIT_{len(chosen)}WAY", conf,
                                 {"parts": [b.txn_id for b in chosen],
                                  "group_size": len(chosen),
                                  "parts_total_paise": sum(b.signed_paise for b in chosen),
                                  "payout_paise": s.payout_paise,
                                  "used_narration_hint": source is hinted,
                                  "uniqueness_verified": True}))
            if cfg.refuse_contingent_subsets:
                chosen_ids = {b.txn_id for b in chosen}
                wanted = {b.signed_paise for b in chosen}
                shadows = [b for b in lines
                           if b.txn_id not in chosen_ids
                           and b.signed_paise in wanted
                           and in_consolidation_window(s, b)]
                if shadows:
                    excs.append(Exception_("settlement", s.settlement_id,
                                           Reason.DUPLICATE_CLAIM,
                                           f"{len(chosen)}-way split is unique only because an "
                                           f"amount-identical twin was already consumed; "
                                           f"{len(shadows)} shadow candidate(s) in window",
                                           {"payout_paise": s.payout_paise,
                                            "proposed_group": [b.txn_id for b in chosen],
                                            "shadow_candidates": sorted(b.txn_id for b in shadows)[:5],
                                            "contested_amounts": sorted(wanted)},
                                           confidence=0.0))
                    used_s.add(s.settlement_id)
                    continue
            span = (max(b.value_date for b in chosen)
                    - min(b.value_date for b in chosen)).days
            if span > cfg.split_long_window_days:
                excs.append(Exception_("settlement", s.settlement_id,
                                       Reason.SPLIT_SPANS_LONG_WINDOW,
                                       f"payout arrived in {len(chosen)} tranches spanning "
                                       f"{span} days — likely a rolling reserve or holdback",
                                       {"payout_paise": s.payout_paise,
                                        "tranche_span_days": span,
                                        "tranches": [b.txn_id for b in chosen],
                                        "tranche_amounts": [b.signed_paise for b in chosen]},
                                       confidence=0.90))
            used_s.add(s.settlement_id)
            for b in chosen:
                used_b.add(b.txn_id)

    # second duplicate pass: catches near-duplicates whose original was
    # only matched by the late subset-sum stage.
    retire_duplicates()

    # ---- residue becomes honest exceptions ----
    for s in settlements:
        if s.settlement_id not in used_s:
            excs.append(Exception_("settlement", s.settlement_id, Reason.SETTLEMENT_NOT_IN_BANK,
                                   "gateway declared a payout with no corresponding bank credit",
                                   {"payout_paise": s.payout_paise,
                                    "settled_on": s.settled_on.isoformat(), "utr": s.utr},
                                   confidence=0.90))
    for b in lines:
        if b.txn_id not in used_b:
            excs.append(Exception_("bank_line", b.txn_id, Reason.UNEXPLAINED_BANK_CREDIT,
                                   "credit could not be attributed to any payout",
                                   {"amount_paise": b.signed_paise, "narration": b.narration,
                                    "value_date": b.value_date.isoformat()},
                                   confidence=0.85))
    return matches, excs


# --------------------------------------------------------------------------
def reconcile(data: dict[str, Any], cfg: MatcherConfig) -> Result:
    t0 = time.perf_counter()
    resolver = SemanticResolver(use_llm=cfg.use_llm)

    m1, e1 = match_orders_payments(data["orders"], data["payments"], cfg)
    e2 = check_arithmetic(data["payments"], data["settlements"], data["refunds"])
    m3, e3 = match_settlements_bank(data["settlements"], data["bank"], cfg, resolver)

    matches = m1 + m3
    excs = e1 + e2 + e3

    # Abstention. A match below threshold is withdrawn and becomes an exception.
    # This is the knob that trades recall for precision, and the reason the
    # evaluation reports a curve rather than a single number.
    kept: list[Match] = []
    for m in matches:
        if m.confidence < cfg.confidence_threshold:
            excs.append(Exception_(
                "match", f"{m.left_id}->{','.join(m.right_ids)}",
                Reason.BELOW_CONFIDENCE_THRESHOLD,
                f"best candidate scored {m.confidence:.2f}, below threshold "
                f"{cfg.confidence_threshold:.2f}; withheld for human review",
                {"rule": m.rule, "layer": m.layer, **m.evidence},
                m.rejected, confidence=m.confidence))
        else:
            kept.append(m)

    elapsed = time.perf_counter() - t0
    n_records = sum(len(data[k]) for k in ("orders", "payments", "refunds", "settlements", "bank"))
    excs.sort(key=lambda e: (-e.severity, e.entity_id))

    return Result(kept, excs, {
        "records_ingested": n_records,
        "wall_clock_s": round(elapsed, 4),
        "records_per_second": round(n_records / elapsed, 1) if elapsed else None,
        "matches_asserted": len(kept),
        "matches_withdrawn_by_threshold": len(matches) - len(kept),
        "exceptions": len(excs),
        "by_rule": {r: sum(1 for m in kept if m.rule == r) for r in sorted({m.rule for m in kept})},
    }, resolver.stats)
