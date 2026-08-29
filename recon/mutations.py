"""Mutation operators.

Each operator changes the ledgers the way the real world changes them, and —
critically — updates the ground truth alongside. A mutation suite that mutates
data without mutating truth measures nothing: every change would register as a
regression.

**A mutation's data effect and its touched set must be pure functions of the
data and the recorded args — never of ground truth.** The first version let
`rolling_reserve_release` look up which bank line a settlement mapped to via
`truth`, and derived touched sets from truth as well. Replay starts from an
empty truth dict (truth is a measurement artifact, not an input), so those
operators silently became no-ops on replay and the event streams diverged at the
very first window. Truth is now written but never read.

Every operator returns the set of entity ids it touched. That set seeds the
window computation, so an operator that under-reports what it touched will
produce a window that is not closed, and the incremental-vs-full drift check
will catch it. The touched set is a claim the harness verifies, not a hint.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from .ai import RE_UTR
from .models import BankLine, Payment, Settlement
from .money import pct_of


def _bank_index(data: dict[str, Any], txn_id: str) -> int | None:
    for i, b in enumerate(data["bank"]):
        if b.txn_id == txn_id:
            return i
    return None


def _sidx(data: dict[str, Any], sid: str) -> int | None:
    for i, s in enumerate(data["settlements"]):
        if s.settlement_id == sid:
            return i
    return None


# --------------------------------------------------------------- operators
def op_bank_credit_disappears(data, truth, args) -> set[str]:
    """A credit posted and then reversed out of the statement export."""
    txn = args["txn_id"]
    i = _bank_index(data, txn)
    if i is None:
        return set()
    data["bank"].pop(i)
    for sid, txns in list(truth["settlement_to_bank"].items()):
        if txn in txns:
            truth["settlement_to_bank"][sid] = [t for t in txns if t != txn]
    truth["bank_to_settlements"].pop(txn, None)
    truth["tiers"].pop(txn, None)
    # Touched is {txn} alone. The settlement it was matched to is pulled in by
    # window closure, which walks CURRENT matches — no truth lookup needed.
    return {txn}


def op_bank_credit_appears(data, truth, args) -> set[str]:
    """A credit that was missing from yesterday's export shows up today."""
    txn = args["txn_id"]
    data["bank"].append(BankLine(txn, args["value_date"], args["narration"],
                                 args["credit_paise"], 0))
    truth["bank_to_settlements"][txn] = []
    truth["tiers"][txn] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": txn,
         "reason": "UNEXPLAINED_BANK_CREDIT", "note": "late-appearing credit"})
    return {txn}


def op_bank_amount_changed(data, truth, args) -> set[str]:
    """The bank restates an amount. Truth keeps the pairing: the credit still
    corresponds to the payout, it is now simply wrong, which the engine should
    surface rather than un-match."""
    txn, delta = args["txn_id"], args["delta_paise"]
    i = _bank_index(data, txn)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"][i] = BankLine(b.txn_id, b.value_date, b.narration,
                               max(b.credit_paise + delta, 0), b.debit_paise)
    return {txn}


def op_bank_narration_changed(data, truth, args) -> set[str]:
    txn = args["txn_id"]
    i = _bank_index(data, txn)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"][i] = BankLine(b.txn_id, b.value_date, args["narration"],
                               b.credit_paise, b.debit_paise)
    truth["tiers"][txn] = 3
    return {txn}


def op_settlement_restated(data, truth, args) -> set[str]:
    """The gateway regenerates a settlement report with a different payout."""
    sid, delta = args["settlement_id"], args["delta_paise"]
    i = _sidx(data, sid)
    if i is None:
        return set()
    s = data["settlements"][i]
    data["settlements"][i] = Settlement(s.settlement_id, s.settled_on,
                                        s.payout_paise + delta, s.utr)
    return {sid}


def op_late_settlement(data, truth, args) -> set[str]:
    """A settlement and its bank credit both arrive late, together."""
    sid, txn = args["settlement_id"], args["txn_id"]
    amt, d = args["payout_paise"], args["settled_on"]
    data["settlements"].append(Settlement(sid, d, amt, args["utr"]))
    data["bank"].append(BankLine(
        txn, d + timedelta(days=args.get("drift", 0)),
        f"NEFT CR-GATEWAYPAY-{sid.upper()}-{args['utr']}-SETTLEMENT", amt, 0))
    truth["settlement_to_bank"][sid] = [txn]
    truth["bank_to_settlements"][txn] = [sid]
    truth["tiers"][txn] = 1
    return {sid, txn}


def op_rolling_reserve_release(data, truth, args) -> set[str]:
    """Retroactively split a landed credit into a short credit plus a release
    seven days later — the unmodeled class that previously caused 84 false
    matches and turned out to be a ground-truth error, not an engine error.

    The target credit is named explicitly in args and the held amount is derived
    from that credit, not from a truth lookup, so the operator is replayable.
    """
    sid, rel, txn = args["settlement_id"], args["release_txn_id"], args["txn_id"]
    i = _bank_index(data, txn)
    if i is None or data["bank"][i].credit_paise <= 0:
        return set()
    b = data["bank"][i]
    held = pct_of(b.credit_paise, args.get("bp", 425))
    if held <= 0 or held >= b.credit_paise:
        return set()
    data["bank"][i] = BankLine(b.txn_id, b.value_date, b.narration,
                               b.credit_paise - held, b.debit_paise)
    data["bank"].append(BankLine(rel, b.value_date + timedelta(days=7),
                                 f"NEFT CR-GATEWAYPAY-RESERVE-RELEASE-{rel.upper()}",
                                 held, 0))
    # GROUND TRUTH CORRECTION (second occurrence of the failure-#11 class).
    # The operator used to attribute the release to the settlement named in args,
    # which the plan chooses independently of the credit being split. When they
    # did not correspond — almost always — the release was labelled unexplained
    # even though the payout genuinely now arrives in two tranches. The engine
    # correctly paired them and was scored wrong for it: 7 of 16 adversarial
    # false matches. The release belongs to whichever settlement actually owns
    # the credit being split. `sid` is retained only for the event payload.
    owner = next((k for k, v in truth["settlement_to_bank"].items() if txn in (v or [])),
                 None)
    if owner is not None:
        truth["settlement_to_bank"][owner] = sorted(
            set((truth["settlement_to_bank"].get(owner) or []) + [rel]))
        truth["bank_to_settlements"][rel] = [owner]
        truth.setdefault("expected_exceptions", []).append(
            {"entity_type": "settlement", "entity_id": owner,
             "reason": "SPLIT_SPANS_LONG_WINDOW", "note": "rolling reserve release"})
    else:
        truth["bank_to_settlements"][rel] = []
    truth["tiers"][rel] = 3
    return {txn, rel}


def op_chargeback_debit(data, truth, args) -> set[str]:
    txn = args["txn_id"]
    data["bank"].append(BankLine(txn, args["value_date"], args["narration"],
                                 0, args["debit_paise"]))
    truth["bank_to_settlements"][txn] = []
    truth["tiers"][txn] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": txn,
         "reason": "UNEXPLAINED_BANK_CREDIT", "note": "chargeback debit"})
    return {txn}


def op_near_duplicate_credit(data, truth, args) -> set[str]:
    """Same amount, narration one character off — exact-string dedupe misses it."""
    src, dup = args["src_txn_id"], args["txn_id"]
    i = _bank_index(data, src)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"].append(BankLine(dup, b.value_date, b.narration[:-1] + args["char"],
                                 b.credit_paise, b.debit_paise))
    truth["bank_to_settlements"][dup] = []
    truth["tiers"][dup] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": dup,
         "reason": "DUPLICATE_BANK_CREDIT", "note": "near-duplicate UTR"})
    return {dup, src}


def op_value_date_skew(data, truth, args) -> set[str]:
    txn = args["txn_id"]
    i = _bank_index(data, txn)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"][i] = BankLine(b.txn_id, b.value_date + timedelta(days=args["days"]),
                               b.narration, b.credit_paise, b.debit_paise)
    truth["tiers"][txn] = 3
    return {txn}



# ---------------------------------------------------------------------------
# Gap 7 — anomaly classes chosen to attack rules added in earlier gaps.
# Each carries ground-truth labels so the oracle can score false matches.
# ---------------------------------------------------------------------------
def op_utr_collision(data, truth, args) -> set[str]:
    """Same UTR string, genuinely DIFFERENT instruments.

    Aimed squarely at "same UTR = same instrument" (R3.0a). A UTR is unique per
    bank, not globally; two banks can mint the same reference. Unrelated amount,
    different remitter, close date — so the date gate cannot save the rule.
    """
    src, txn = args["src_txn_id"], args["txn_id"]
    i = _bank_index(data, src)
    if i is None:
        return set()
    b = data["bank"][i]
    m = RE_UTR.search(b.narration)
    if not m:
        return set()
    data["bank"].append(BankLine(
        txn, b.value_date, f"NEFT CR-OTHERBANK-{args['remitter']}-{m.group(1)}-INWARD",
        args["credit_paise"], 0))
    truth["bank_to_settlements"][txn] = []
    truth["tiers"][txn] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": txn,
         "reason": "UNEXPLAINED_BANK_CREDIT", "note": "UTR collision, different bank"})
    return {txn, src}


def op_transaction_date_skew(data, truth, args) -> set[str]:
    """Value date vs transaction date disagreeing by more than the look-back."""
    txn = args["txn_id"]
    i = _bank_index(data, txn)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"][i] = BankLine(b.txn_id, b.value_date + timedelta(days=args["days"]),
                               b.narration, b.credit_paise, b.debit_paise)
    truth["tiers"][txn] = 3
    return {txn}


def op_partial_settlement_restatement(data, truth, args) -> set[str]:
    """Only a SUBSET of a batch's captures is restated, so the batch stops footing
    while the payout and every other member stay untouched."""
    sid, delta = args["settlement_id"], args["delta_paise"]
    n = args.get("n_payments", 2)
    touched = {sid}
    changed = 0
    for i, p in enumerate(data["payments"]):
        if p.settlement_id != sid or changed >= n:
            continue
        data["payments"][i] = Payment(p.payment_id, p.order_ref, p.captured_on,
                                      p.gross_paise, p.fee_paise, p.gst_paise,
                                      p.net_paise + delta, p.settlement_id, p.method)
        touched.add(p.payment_id)
        changed += 1
    if changed:
        truth.setdefault("expected_exceptions", []).append(
            {"entity_type": "settlement", "entity_id": sid,
             "reason": "BATCH_ARITHMETIC_MISMATCH", "note": "partial restatement"})
    return touched


def op_duplicate_before_original(data, truth, args) -> set[str]:
    """A phantom that POSTS FIRST.

    Aimed at the duplicate-cluster tie-break, which keeps the earliest line by
    (date, id). If the phantom is earlier, the naive rule keeps the phantom and
    retires the real credit — losing a true match and asserting a false one.
    """
    src, dup = args["src_txn_id"], args["txn_id"]
    i = _bank_index(data, src)
    if i is None:
        return set()
    b = data["bank"][i]
    data["bank"].append(BankLine(dup, b.value_date - timedelta(days=1),
                                 b.narration, b.credit_paise, b.debit_paise))
    truth["bank_to_settlements"][dup] = []
    truth["tiers"][dup] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": dup,
         "reason": "DUPLICATE_BANK_CREDIT", "note": "duplicate posted before original"})
    return {dup, src}


def op_near_duplicate_shortfall(data, truth, args) -> set[str]:
    """Differs in BOTH axes at once: amount short by a plausible withholding rate
    AND one character of the UTR. Defeats amount clustering, UTR clustering, and
    is shaped to be accepted by the withholding-rate gate."""
    src, dup = args["src_txn_id"], args["txn_id"]
    i = _bank_index(data, src)
    if i is None or data["bank"][i].credit_paise <= 0:
        return set()
    b = data["bank"][i]
    short = b.credit_paise - pct_of(b.credit_paise, args.get("bp", 100))
    data["bank"].append(BankLine(dup, b.value_date,
                                 b.narration[:-1] + args["char"], short, 0))
    truth["bank_to_settlements"][dup] = []
    truth["tiers"][dup] = 3
    truth.setdefault("expected_exceptions", []).append(
        {"entity_type": "bank_line", "entity_id": dup,
         "reason": "DUPLICATE_BANK_CREDIT", "note": "near-dup: short amount + corrupted UTR"})
    return {dup, src}


def op_chargeback_representment(data, truth, args) -> set[str]:
    """The full lifecycle: a chargeback debit, then a representment credit that
    reverses it days later. Neither leg belongs to any payout, and the pair must
    not be conscripted into explaining one."""
    dr, cr = args["debit_txn_id"], args["credit_txn_id"]
    amt = args["amount_paise"]
    data["bank"].append(BankLine(dr, args["value_date"],
                                 f"NEFT DR-CHARGEBACK-{args['ref']}", 0, amt))
    data["bank"].append(BankLine(cr, args["value_date"] + timedelta(days=args.get("gap", 9)),
                                 f"NEFT CR-REPRESENTMENT-{args['ref']}", amt, 0))
    for t in (dr, cr):
        truth["bank_to_settlements"][t] = []
        truth["tiers"][t] = 3
        truth.setdefault("expected_exceptions", []).append(
            {"entity_type": "bank_line", "entity_id": t,
             "reason": "UNEXPLAINED_BANK_CREDIT", "note": "chargeback lifecycle leg"})
    return {dr, cr}


OPS = {
    "bank_credit_disappears": op_bank_credit_disappears,
    "bank_credit_appears": op_bank_credit_appears,
    "bank_amount_changed": op_bank_amount_changed,
    "bank_narration_changed": op_bank_narration_changed,
    "settlement_restated": op_settlement_restated,
    "late_settlement": op_late_settlement,
    "rolling_reserve_release": op_rolling_reserve_release,
    "chargeback_debit": op_chargeback_debit,
    "near_duplicate_credit": op_near_duplicate_credit,
    "value_date_skew": op_value_date_skew,
    # Gap 7 classes
    "utr_collision": op_utr_collision,
    "transaction_date_skew": op_transaction_date_skew,
    "partial_settlement_restatement": op_partial_settlement_restatement,
    "duplicate_before_original": op_duplicate_before_original,
    "near_duplicate_shortfall": op_near_duplicate_shortfall,
    "chargeback_representment": op_chargeback_representment,
}

_TRUTH: dict[int, dict] = {}


def bind_truth(data: dict[str, Any], truth: dict[str, Any]) -> None:
    """Associate a truth dict with a data dict so `apply` can update both."""
    _TRUTH[id(data)] = truth


def apply(data: dict[str, Any], mutation: dict[str, Any]) -> set[str]:
    truth = _TRUTH.get(id(data), {"settlement_to_bank": {}, "bank_to_settlements": {},
                                  "tiers": {}, "expected_exceptions": []})
    touched = OPS[mutation["op"]](data, truth, mutation.get("args", {}))
    data["bank"].sort(key=lambda b: (b.value_date, b.txn_id))
    data["settlements"].sort(key=lambda s: (s.settled_on, s.settlement_id))
    return touched
