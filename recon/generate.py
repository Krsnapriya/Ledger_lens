"""Synthetic ledger generator — the most important file in this repo.

Reported accuracy is only as credible as the data it was measured on. A generator
that emits clean rows produces a 99% match rate that means nothing. So this
generator is built as a first-class artifact: every anomaly it injects is
recorded in a ground-truth file, and every bank line is stamped with a
difficulty tier so accuracy can be reported per tier instead of as one
flattering average.

Sources emitted (5 ledgers):
    orders.csv        internal order ledger
    payments.csv      gateway capture report
    refunds.csv       gateway refund report
    settlements.csv   gateway payout report
    bank.csv          bank statement

Deliberate design choice: the generator does NOT know about the matcher. It
models a payment business and then damages the records the way real exports are
damaged. It is not a mirror of the matching rules, which would make the whole
evaluation circular.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .money import fmt, pct_of
from .models import BankLine, Order, Payment, Refund, Settlement

MDR_BP = 200        # contracted 2.00% merchant discount rate
GST_BP = 1800       # 18% GST on the fee
START = date(2026, 4, 1)

MERCHANT_NAMES = [
    "ACME TRADING PVT LTD", "NIMBUS RETAIL", "KHANNA ELECTRONICS",
    "SOUTHERN SPICE FOODS", "VELOCITY LOGISTICS", "PIXELWORKS STUDIO",
    "GREENLEAF ORGANICS", "MERIDIAN BOOKS", "TATVA WELLNESS", "ORBIT MOBILITY",
]
METHODS = ["upi", "card", "netbanking", "wallet"]


def _utr(rng: random.Random) -> str:
    return f"UTR{rng.randint(10**11, 10**12 - 1)}"


def _garble(s: str, rng: random.Random) -> str:
    """Damage a string the way an OCR'd / truncated bank narration is damaged."""
    chars = list(s)
    for _ in range(max(1, len(chars) // 12)):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    out = "".join(chars)
    if rng.random() < 0.5:
        out = out[: max(8, int(len(out) * rng.uniform(0.55, 0.85)))]
    return out


def _indian_fmt(paise: int) -> str:
    """Some bank exports ship money as lakh-grouped strings, not numbers."""
    return fmt(paise).replace("₹", "")


class Generated:
    def __init__(self) -> None:
        self.orders: list[Order] = []
        self.payments: list[Payment] = []
        self.refunds: list[Refund] = []
        self.settlements: list[Settlement] = []
        self.bank: list[BankLine] = []
        self.truth: dict[str, Any] = {
            "order_to_payment": {},
            "payment_to_settlement": {},
            "settlement_to_bank": {},
            "bank_to_settlements": {},
            "expected_exceptions": [],
            "tiers": {},
            "injected": [],
        }

    def _expect(self, entity_type: str, entity_id: str, reason: str, note: str) -> None:
        self.truth["expected_exceptions"].append(
            {"entity_type": entity_type, "entity_id": entity_id,
             "reason": reason, "note": note}
        )


def generate(n_orders: int = 300, seed: int = 7, difficulty: float = 1.0,
             days: int = 28, multiway: float = 0.0, unmodeled: float = 0.0) -> Generated:
    """Build a full noisy multi-ledger batch.

    difficulty scales anomaly injection rates. 0.0 => pristine data (used by the
    test suite to assert the matcher hits 100% when there is nothing to be
    confused by; if it cannot do that, nothing else it says is believable).

    multiway injects consolidations and splits of cardinality 3-8. Added after
    discovering the original generator NEVER produced a group larger than 2,
    which meant the documented "subset-sum bounded to 3" limitation had never
    once been exercised. A limitation you have not tested is a guess.

    unmodeled injects anomaly classes the matcher was explicitly NOT designed
    for — rolling reserves at implausible rates, chargeback debits, near-duplicate
    UTRs, credits landing before the settlement report, value-date skew beyond
    the matching window. This is the generalisation test: the engine is allowed
    to MISS these, but it must never match them wrongly. Degrading into honest
    exceptions is a pass; a confident false match is a fail.
    """
    rng = random.Random(seed)
    g = Generated()
    d = difficulty

    # ---------- orders + captures ----------
    for i in range(n_orders):
        oid = f"order_{i:05d}"
        created = START + timedelta(days=rng.randrange(0, days))
        amount = rng.randrange(19900, 4_500_000)  # ₹199 .. ₹45,000
        captured = rng.random() > 0.13

        if not captured:
            # BUG FIX (run 1): a failed order is not a reconciliation exception.
            # It was originally labelled ORDER_NO_PAYMENT, which meant ground
            # truth demanded 40 exceptions the matcher (correctly) never raised,
            # dragging exception recall down for no reason. Failed orders are
            # simply out of scope: no money moved, nothing to reconcile.
            g.orders.append(Order(oid, created, rng.choice(MERCHANT_NAMES),
                                  amount, "failed", None))
            continue

        pid = f"pay_{i:05d}{rng.choice('abcdefgh')}"
        fee = pct_of(amount, MDR_BP)
        gst = pct_of(fee, GST_BP)
        net = amount - fee - gst

        # --- anomaly: gateway fee charged at the wrong rate ---
        if rng.random() < 0.020 * d:
            fee = pct_of(amount, MDR_BP + rng.choice([25, 50, -25]))
            # BUG FIX (run 1): GST must be recomputed off the *charged* fee.
            # Leaving the old GST made every wrong-fee row also a wrong-GST row,
            # so the detector correctly raised GST_MISMATCH x6 and the evaluator
            # scored them as spurious. The detector was right; the generator was
            # emitting an unlabelled second anomaly. A gateway that overcharges
            # its fee still charges 18% GST on the fee it actually took.
            gst = pct_of(fee, GST_BP)
            net = amount - fee - gst
            g._expect("payment", pid, "FEE_RATE_MISMATCH", "fee off contracted MDR")
            g.truth["injected"].append({"kind": "fee_rate_wrong", "id": pid})
        # --- anomaly: GST not 18% of fee ---
        elif rng.random() < 0.015 * d:
            gst = pct_of(fee, GST_BP - 300)
            net = amount - fee - gst
            g._expect("payment", pid, "GST_MISMATCH", "GST off 18%")
            g.truth["injected"].append({"kind": "gst_wrong", "id": pid})

        # --- link damage between order ledger and gateway ---
        link = rng.random()
        order_pid: str | None = pid
        order_ref: str | None = oid
        if link < 0.07 * d:
            order_pid = None                       # tier 2: no id on our side
            g.truth["injected"].append({"kind": "missing_pid", "id": oid})
        elif link < 0.12 * d:
            order_ref = oid[: rng.randrange(6, 10)]  # tier 2: truncated ref
            g.truth["injected"].append({"kind": "truncated_ref", "id": pid})
        elif link < 0.155 * d:
            order_pid = pid[:-1] + rng.choice("xyz9")  # tier 3: typo'd id
            order_ref = None
            g.truth["injected"].append({"kind": "typo_pid", "id": oid})

        g.orders.append(Order(oid, created, rng.choice(MERCHANT_NAMES),
                              amount, "captured", order_pid))
        g.payments.append(Payment(pid, order_ref, created, amount, fee, gst, net,
                                  None, rng.choice(METHODS)))
        g.truth["order_to_payment"][oid] = pid

    # --- anomaly: capture with no internal order (out-of-band payment link) ---
    for k in range(int(3 * d)):
        pid = f"pay_orphan_{k}"
        amount = rng.randrange(50_000, 900_000)
        fee = pct_of(amount, MDR_BP)
        gst = pct_of(fee, GST_BP)
        g.payments.append(Payment(pid, None, START + timedelta(days=rng.randrange(0, days)),
                                  amount, fee, gst, amount - fee - gst, None,
                                  rng.choice(METHODS)))
        g._expect("payment", pid, "PAYMENT_NO_ORDER", "capture with no order row")
        g.truth["injected"].append({"kind": "orphan_payment", "id": pid})

    # ---------- refunds ----------
    refundable = [p for p in g.payments if rng.random() < 0.06]
    for j, p in enumerate(refundable):
        amt = p.gross_paise if rng.random() < 0.4 else pct_of(p.gross_paise, rng.randrange(2000, 8000))
        g.refunds.append(Refund(f"rfnd_{j:04d}", p.payment_id,
                                p.captured_on + timedelta(days=rng.randrange(1, 5)),
                                amt, None))

    # ---------- settlement batches (T+2) ----------
    by_day: dict[date, list[Payment]] = {}
    for p in g.payments:
        by_day.setdefault(p.captured_on, []).append(p)

    settled_payments: list[Payment] = []
    batches: list[tuple[Settlement, list[Payment], list[Refund]]] = []
    for k, day in enumerate(sorted(by_day)):
        group = by_day[day]
        # a slice of the last day's captures stays unsettled (legitimately pending)
        if day == max(by_day) and rng.random() < 0.8:
            continue
        sid = f"setl_{k:04d}"
        settled_on = day + timedelta(days=2)
        batch_refunds = [r for r in g.refunds if r.refunded_on <= settled_on
                         and r.settlement_id is None
                         and any(p2.payment_id == r.payment_id for p2 in g.payments)]
        batch_refunds = batch_refunds[:3]
        for r in batch_refunds:
            idx = g.refunds.index(r)
            g.refunds[idx] = Refund(r.refund_id, r.payment_id, r.refunded_on, r.amount_paise, sid)

        gross_net = sum(p.net_paise for p in group)
        refund_total = sum(r.amount_paise for r in batch_refunds)
        payout = gross_net - refund_total

        declared = payout
        # --- anomaly: gateway's own batch arithmetic does not foot ---
        if rng.random() < 0.05 * d:
            declared = payout + rng.choice([-1, 1]) * rng.randrange(500, 50_000)
            g._expect("settlement", sid, "BATCH_ARITHMETIC_MISMATCH",
                      "declared payout != sum(net) - refunds")
            g.truth["injected"].append({"kind": "batch_arith_wrong", "id": sid})

        s = Settlement(sid, settled_on, declared,
                       None if rng.random() < 0.15 * d else _utr(rng))
        batches.append((s, group, batch_refunds))
        g.settlements.append(s)
        for p in group:
            idx = g.payments.index(p)
            g.payments[idx] = Payment(p.payment_id, p.order_ref, p.captured_on,
                                      p.gross_paise, p.fee_paise, p.gst_paise,
                                      p.net_paise, sid, p.method)
            g.truth["payment_to_settlement"][p.payment_id] = sid
        settled_payments.extend(group)

    # ---------- bank statement ----------
    txn_n = 0

    def _new_txn() -> str:
        nonlocal txn_n
        txn_n += 1
        return f"bank_{txn_n:05d}"

    def _narration(s: Settlement, tier: int) -> str:
        base = f"NEFT CR-GATEWAYPAY-{s.settlement_id.upper()}-{s.utr or 'NOUTR'}-SETTLEMENT"
        if tier == 1:
            return base
        if tier == 2:
            return f"NEFT CR-GATEWAYPAY-{s.utr or 'PAYOUT'}-SETTLEMENT"  # id dropped
        return _garble(base, rng)

    i = 0
    while i < len(batches):
        s, group, brefunds = batches[i]
        roll = rng.random()

        # --- anomaly: payout never landed in the bank ---
        if roll < 0.045 * d:
            g.truth["settlement_to_bank"][s.settlement_id] = []
            g._expect("settlement", s.settlement_id, "SETTLEMENT_NOT_IN_BANK",
                      "gateway declared payout, no bank credit")
            g.truth["injected"].append({"kind": "missing_in_bank", "id": s.settlement_id})
            i += 1
            continue

        # --- anomaly: k-way consolidation (end-of-day batching), k in 3..8 ---
        if multiway and rng.random() < 0.30 * multiway and i + 2 < len(batches):
            k = min(rng.randrange(3, 9), len(batches) - i)
            group_s = [batches[i + j][0] for j in range(k)]
            txn = _new_txn()
            total = sum(x.payout_paise for x in group_s)
            vd = max(x.settled_on for x in group_s) + timedelta(days=rng.randrange(0, 2))
            g.bank.append(BankLine(txn, vd, f"NEFT CR-GATEWAYPAY-CONSOLIDATED-{k}X-{_utr(rng)}",
                                   max(total, 0), max(-total, 0)))
            for x in group_s:
                g.truth["settlement_to_bank"][x.settlement_id] = [txn]
            g.truth["bank_to_settlements"][txn] = [x.settlement_id for x in group_s]
            g.truth["tiers"][txn] = 3
            g.truth["injected"].append({"kind": f"multiway_merge_{k}", "id": txn})
            i += k
            continue

        # --- anomaly: k-way split of one payout, k in 3..5 ---
        if multiway and rng.random() < 0.20 * multiway and s.payout_paise > 0:
            k = rng.randrange(3, 6)
            base = s.payout_paise // k
            parts = [base] * (k - 1) + [s.payout_paise - base * (k - 1)]
            txns = []
            for amt in parts:
                t = _new_txn()
                g.bank.append(BankLine(t, s.settled_on + timedelta(days=rng.randrange(0, 2)),
                                       f"NEFT CR-GATEWAYPAY-{s.settlement_id.upper()}-PART-{_utr(rng)}",
                                       amt, 0))
                g.truth["bank_to_settlements"][t] = [s.settlement_id]
                g.truth["tiers"][t] = 3
                txns.append(t)
            g.truth["settlement_to_bank"][s.settlement_id] = txns
            g.truth["injected"].append({"kind": f"multiway_split_{k}", "id": s.settlement_id})
            i += 1
            continue

        # --- anomaly: two settlements arrive merged as one bank credit ---
        if roll < 0.10 * d and i + 1 < len(batches):
            s2, _, _ = batches[i + 1]
            txn = _new_txn()
            total = s.payout_paise + s2.payout_paise
            vd = max(s.settled_on, s2.settled_on) + timedelta(days=rng.randrange(0, 2))
            g.bank.append(BankLine(txn, vd,
                                   f"NEFT CR-GATEWAYPAY-CONSOLIDATED-{_utr(rng)}", total, 0))
            g.truth["settlement_to_bank"][s.settlement_id] = [txn]
            g.truth["settlement_to_bank"][s2.settlement_id] = [txn]
            g.truth["bank_to_settlements"][txn] = [s.settlement_id, s2.settlement_id]
            g.truth["tiers"][txn] = 3
            g.truth["injected"].append({"kind": "merged_payout", "id": txn})
            i += 2
            continue

        # --- anomaly: one payout split across two bank credits ---
        if roll < 0.15 * d:
            a = s.payout_paise // 2
            b = s.payout_paise - a
            t1, t2 = _new_txn(), _new_txn()
            for t, amt in ((t1, a), (t2, b)):
                g.bank.append(BankLine(t, s.settled_on + timedelta(days=rng.randrange(0, 2)),
                                       f"NEFT CR-GATEWAYPAY-{s.settlement_id.upper()}-PART-{_utr(rng)}",
                                       amt, 0))
                g.truth["bank_to_settlements"][t] = [s.settlement_id]
                g.truth["tiers"][t] = 3
            g.truth["settlement_to_bank"][s.settlement_id] = [t1, t2]
            g.truth["injected"].append({"kind": "split_payout", "id": s.settlement_id})
            i += 1
            continue

        # --- normal landing, with tiered narration/date damage ---
        r2 = rng.random()
        tier = 1 if r2 > 0.45 * d else (2 if r2 > 0.18 * d else 3)
        drift = 0 if tier == 1 else rng.randrange(0, 4)
        credit = s.payout_paise

        # --- anomaly: TDS silently withheld outside the declared batch ---
        if rng.random() < 0.05 * d and s.payout_paise > 0:
            credit = s.payout_paise - pct_of(s.payout_paise, 100)
            g._expect("settlement", s.settlement_id, "AMOUNT_MISMATCH",
                      "bank credit short of declared payout (TDS withheld)")
            g.truth["injected"].append({"kind": "silent_tds", "id": s.settlement_id})
            tier = max(tier, 2)

        txn = _new_txn()
        # A net-negative batch is a recovery DEBIT, not a credit. See the
        # signed_paise note on BankLine.
        g.bank.append(BankLine(txn, s.settled_on + timedelta(days=drift),
                               _narration(s, tier),
                               credit if credit >= 0 else 0,
                               0 if credit >= 0 else -credit))
        g.truth["settlement_to_bank"][s.settlement_id] = [txn]
        g.truth["bank_to_settlements"][txn] = [s.settlement_id]
        g.truth["tiers"][txn] = tier

        # --- anomaly: the same credit posted twice ---
        if rng.random() < 0.035 * d:
            dup = _new_txn()
            g.bank.append(BankLine(dup, s.settled_on + timedelta(days=drift),
                                   g.bank[-1].narration + " ",
                                   credit if credit >= 0 else 0,
                                   0 if credit >= 0 else -credit))
            g.truth["bank_to_settlements"][dup] = []
            g.truth["tiers"][dup] = 2
            g._expect("bank_line", dup, "DUPLICATE_BANK_CREDIT", "same payout posted twice")
            g.truth["injected"].append({"kind": "dup_credit", "id": dup})
        i += 1

    # ---------- UNMODELED damage: classes the matcher was not built for ----------
    if unmodeled:
        u = unmodeled
        landed = [(sid, txns[0]) for sid, txns in g.truth["settlement_to_bank"].items()
                  if len(txns) == 1]
        bank_by_id = {b.txn_id: idx for idx, b in enumerate(g.bank)}

        for sid, txn in landed:
            s_obj = next(x for x in g.settlements if x.settlement_id == sid)
            idx = bank_by_id.get(txn)
            if idx is None or s_obj.payout_paise <= 0:
                continue

            # 1. Rolling reserve: an implausible-rate holdback, released later as
            #    a separate credit with no settlement of its own. The withholding
            #    -rate gate is designed to REJECT this, so it must surface as an
            #    exception, not as a confident short match.
            if rng.random() < 0.10 * u:
                bp = rng.choice([370, 425, 615, 880])      # deliberately not 10/100/200/500
                held = pct_of(s_obj.payout_paise, bp)
                old = g.bank[idx]
                g.bank[idx] = BankLine(old.txn_id, old.value_date, old.narration,
                                       old.credit_paise - held, old.debit_paise)
                rel = _new_txn()
                g.bank.append(BankLine(rel, old.value_date + timedelta(days=7),
                                       f"NEFT CR-GATEWAYPAY-RESERVE-RELEASE-{_utr(rng)}", held, 0))
                # GROUND TRUTH CORRECTION. This was originally labelled
                # UNEXPLAINED_BANK_CREDIT, and the engine was scored as producing
                # 84 false matches for pairing the two credits. The engine was
                # right: the payout genuinely arrived as two tranches summing to
                # exactly P, and reconciling them together is the economically
                # correct answer. The label was wrong.
                #
                # What the engine legitimately owed and did not provide was a
                # signal that 7 days of working capital were held — so the
                # expected exception is now SPLIT_SPANS_LONG_WINDOW on the
                # settlement, not "unexplained" on the credit.
                g.truth["settlement_to_bank"][sid] = [txn, rel]
                g.truth["bank_to_settlements"][rel] = [sid]
                g.truth["bank_to_settlements"][txn] = [sid]
                g.truth["tiers"][rel] = 3
                g._expect("settlement", sid, "SPLIT_SPANS_LONG_WINDOW",
                          "payout arrived in tranches 7 days apart (rolling reserve)")
                g.truth["injected"].append({"kind": "rolling_reserve", "id": sid})

            # 2. Value-date skew beyond the matching window.
            elif rng.random() < 0.08 * u:
                old = g.bank[idx]
                g.bank[idx] = BankLine(old.txn_id, old.value_date + timedelta(days=rng.randrange(5, 9)),
                                       old.narration, old.credit_paise, old.debit_paise)
                g.truth["tiers"][old.txn_id] = 3
                g.truth["injected"].append({"kind": "value_date_skew", "id": old.txn_id})

            # 3. Bank credit lands BEFORE the settlement report is generated.
            elif rng.random() < 0.06 * u:
                old = g.bank[idx]
                g.bank[idx] = BankLine(old.txn_id, old.value_date - timedelta(days=rng.randrange(5, 8)),
                                       old.narration, old.credit_paise, old.debit_paise)
                g.truth["tiers"][old.txn_id] = 3
                g.truth["injected"].append({"kind": "credit_before_report", "id": old.txn_id})

            # 4. Near-duplicate UTR: one character off, so exact-string dedupe misses it.
            elif rng.random() < 0.05 * u:
                old = g.bank[idx]
                dup = _new_txn()
                narr = old.narration[:-1] + rng.choice("0123456789")
                g.bank.append(BankLine(dup, old.value_date, narr,
                                       old.credit_paise, old.debit_paise))
                g.truth["bank_to_settlements"][dup] = []
                g.truth["tiers"][dup] = 3
                g._expect("bank_line", dup, "DUPLICATE_BANK_CREDIT", "near-duplicate UTR, not exact string")
                g.truth["injected"].append({"kind": "near_dup_utr", "id": dup})

        # 5. Chargeback debits: money out, tied to no settlement at all.
        for _ in range(int(3 * u)):
            t = _new_txn()
            g.bank.append(BankLine(t, START + timedelta(days=rng.randrange(5, days)),
                                   f"NEFT DR-CHARGEBACK-{rng.choice(MERCHANT_NAMES)}-{_utr(rng)}",
                                   0, rng.randrange(50_000, 800_000)))
            g.truth["bank_to_settlements"][t] = []
            g.truth["tiers"][t] = 3
            g._expect("bank_line", t, "UNEXPLAINED_BANK_CREDIT", "chargeback debit, no settlement")
            g.truth["injected"].append({"kind": "chargeback_debit", "id": t})

    # --- anomaly: credits that are simply not gateway payouts ---
    for k in range(int(4 * d)):
        txn = _new_txn()
        amt = rng.randrange(100_000, 3_000_000)
        g.bank.append(BankLine(txn, START + timedelta(days=rng.randrange(3, days + 2)),
                               f"NEFT CR-{rng.choice(MERCHANT_NAMES)}-DIRECT-{_utr(rng)}", amt, 0))
        g.truth["bank_to_settlements"][txn] = []
        g.truth["tiers"][txn] = 1
        g._expect("bank_line", txn, "UNEXPLAINED_BANK_CREDIT", "direct transfer, not a payout")
        g.truth["injected"].append({"kind": "direct_neft", "id": txn})

    g.bank.sort(key=lambda b: (b.value_date, b.txn_id))
    return g


def write(g: Generated, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)

    def dump(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with (outdir / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    dump("orders.csv", [
        {"order_id": o.order_id, "created_at": o.created_on.isoformat(),
         "customer": o.customer, "amount": f"{o.amount_paise/100:.2f}",
         "status": o.status, "gateway_payment_id": o.gateway_payment_id or ""}
        for o in g.orders
    ], ["order_id", "created_at", "customer", "amount", "status", "gateway_payment_id"])

    dump("payments.csv", [
        {"payment_id": p.payment_id, "order_ref": p.order_ref or "",
         "captured_at": p.captured_on.isoformat(),
         "gross": f"{p.gross_paise/100:.2f}", "fee": f"{p.fee_paise/100:.2f}",
         "gst_on_fee": f"{p.gst_paise/100:.2f}", "net": f"{p.net_paise/100:.2f}",
         "settlement_id": p.settlement_id or "", "method": p.method}
        for p in g.payments
    ], ["payment_id", "order_ref", "captured_at", "gross", "fee", "gst_on_fee",
        "net", "settlement_id", "method"])

    dump("refunds.csv", [
        {"refund_id": r.refund_id, "payment_id": r.payment_id,
         "refunded_at": r.refunded_on.isoformat(),
         "amount": f"{r.amount_paise/100:.2f}", "settlement_id": r.settlement_id or ""}
        for r in g.refunds
    ], ["refund_id", "payment_id", "refunded_at", "amount", "settlement_id"])

    dump("settlements.csv", [
        {"settlement_id": s.settlement_id, "settled_at": s.settled_on.isoformat(),
         "payout_amount": f"{s.payout_paise/100:.2f}", "utr": s.utr or ""}
        for s in g.settlements
    ], ["settlement_id", "settled_at", "payout_amount", "utr"])

    # Bank export: mixed money formatting on purpose. Half the rows carry
    # lakh-grouped strings, which is what a real bank CSV actually looks like.
    dump("bank.csv", [
        {"txn_id": b.txn_id, "value_date": b.value_date.strftime("%d-%m-%Y"),
         "narration": b.narration,
         "credit": (_indian_fmt(b.credit_paise) if rng.random() < 0.5
                    else f"{b.credit_paise/100:.2f}") if b.credit_paise else "",
         "debit": f"{b.debit_paise/100:.2f}" if b.debit_paise else ""}
        for b in g.bank
    ], ["txn_id", "value_date", "narration", "credit", "debit"])

    (outdir / "truth.json").write_text(json.dumps(g.truth, indent=2, default=str))
