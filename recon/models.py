"""Typed records + the exception taxonomy.

The reason codes are a closed enum on purpose. An exception engine that emits
free-text reasons cannot be measured: you can't compute exception precision
against a ground truth if every run phrases the failure differently. Free text
belongs in `detail`, never in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Reason(str, Enum):
    """Closed set of unresolved-record reason codes."""

    # order <-> payment layer
    ORDER_NO_PAYMENT = "ORDER_NO_PAYMENT"           # order never captured
    PAYMENT_NO_ORDER = "PAYMENT_NO_ORDER"           # capture with no internal order
    ORDER_PAYMENT_AMBIGUOUS = "ORDER_PAYMENT_AMBIGUOUS"

    # fee / tax arithmetic layer
    FEE_RATE_MISMATCH = "FEE_RATE_MISMATCH"         # fee != contracted rate
    GST_MISMATCH = "GST_MISMATCH"                   # GST != 18% of fee
    NET_ARITHMETIC_MISMATCH = "NET_ARITHMETIC_MISMATCH"  # net != gross-fee-gst

    # settlement batch layer
    BATCH_ARITHMETIC_MISMATCH = "BATCH_ARITHMETIC_MISMATCH"

    # settlement <-> bank layer
    SETTLEMENT_NOT_IN_BANK = "SETTLEMENT_NOT_IN_BANK"    # payout never landed
    UNEXPLAINED_BANK_CREDIT = "UNEXPLAINED_BANK_CREDIT"  # money in, no source
    DUPLICATE_BANK_CREDIT = "DUPLICATE_BANK_CREDIT"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    BELOW_CONFIDENCE_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"  # abstained
    AMBIGUOUS_SUBSET = "AMBIGUOUS_SUBSET"       # >1 group sums to the same target
    AMBIGUOUS_ASSIGNMENT = "AMBIGUOUS_ASSIGNMENT"  # >1 equally valid 1:1 pairing
    DUPLICATE_CLAIM = "DUPLICATE_CLAIM"         # solution unique only because a twin was consumed
    OVERRIDE_LEAK = "OVERRIDE_LEAK"             # a human-asserted record reached the engine
    POOL_CAP_EXCEEDED = "POOL_CAP_EXCEEDED"     # too many candidates to decide safely
    SPLIT_SPANS_LONG_WINDOW = "SPLIT_SPANS_LONG_WINDOW"  # tranches far apart in time


# Severity drives the exception inbox ordering. Money-out-the-door first.
SEVERITY: dict[Reason, int] = {
    Reason.OVERRIDE_LEAK: 120,
    Reason.SETTLEMENT_NOT_IN_BANK: 100,
    Reason.DUPLICATE_BANK_CREDIT: 95,
    Reason.BATCH_ARITHMETIC_MISMATCH: 90,
    Reason.AMOUNT_MISMATCH: 80,
    Reason.UNEXPLAINED_BANK_CREDIT: 70,
    Reason.AMBIGUOUS_SUBSET: 68,
    Reason.AMBIGUOUS_ASSIGNMENT: 66,
    Reason.POOL_CAP_EXCEEDED: 69,
    Reason.DUPLICATE_CLAIM: 67,
    Reason.SPLIT_SPANS_LONG_WINDOW: 65,
    Reason.FEE_RATE_MISMATCH: 60,
    Reason.GST_MISMATCH: 55,
    Reason.NET_ARITHMETIC_MISMATCH: 55,
    Reason.PAYMENT_NO_ORDER: 50,
    Reason.ORDER_PAYMENT_AMBIGUOUS: 40,
    Reason.BELOW_CONFIDENCE_THRESHOLD: 35,
    Reason.ORDER_NO_PAYMENT: 10,
}

SUGGESTED_ACTION: dict[Reason, str] = {
    Reason.OVERRIDE_LEAK: "STOP. A human-asserted record reached the matching engine. Do not trust this run; the override isolation boundary is broken.",
    Reason.SETTLEMENT_NOT_IN_BANK: "Raise payout trace with gateway; confirm beneficiary account.",
    Reason.DUPLICATE_BANK_CREDIT: "Confirm reversal posted; do not recognise revenue twice.",
    Reason.BATCH_ARITHMETIC_MISMATCH: "Request settlement breakup file; check for unlisted deduction.",
    Reason.AMOUNT_MISMATCH: "Check for TDS/adjustment withheld outside the declared batch.",
    Reason.UNEXPLAINED_BANK_CREDIT: "Identify remitter; likely direct customer transfer, not gateway.",
    Reason.AMBIGUOUS_SUBSET: "Multiple settlement groups sum to this amount; request the gateway breakup file to disambiguate.",
    Reason.POOL_CAP_EXCEEDED: "Too many candidates in the window to decide safely. Narrow the period or obtain the gateway breakup file; do not accept a partial attribution.",
    Reason.AMBIGUOUS_ASSIGNMENT: "Several ledger rows and bank rows in this window share an amount and date; the pairing an optimiser returns is arbitrary. Obtain a reference that distinguishes them before attributing.",
    Reason.DUPLICATE_CLAIM: "A member of this group has an amount-identical twin elsewhere in the window; the grouping is only unique because the twin was already consumed. Confirm which credit belongs to this payout.",
    Reason.SPLIT_SPANS_LONG_WINDOW: "Payout arrived in tranches days apart — likely a rolling reserve or holdback. Confirm the reserve policy and the working capital held.",
    Reason.FEE_RATE_MISMATCH: "Compare against contracted MDR; raise fee dispute if confirmed.",
    Reason.GST_MISMATCH: "Verify GST on fee at 18%; check gateway tax invoice.",
    Reason.NET_ARITHMETIC_MISMATCH: "Gateway net does not equal gross-fee-GST; request correction.",
    Reason.PAYMENT_NO_ORDER: "Capture with no internal order: check for out-of-band payment link.",
    Reason.ORDER_PAYMENT_AMBIGUOUS: "Multiple equally-plausible captures; needs human tie-break.",
    Reason.BELOW_CONFIDENCE_THRESHOLD: "Best candidate below threshold; human confirm or reject.",
    Reason.ORDER_NO_PAYMENT: "Expected: abandoned/failed order. No action unless volume spikes.",
}


@dataclass(frozen=True)
class Order:
    order_id: str
    created_on: date
    customer: str
    amount_paise: int
    status: str                      # captured | failed
    gateway_payment_id: str | None   # frequently missing/typo'd in the wild


@dataclass(frozen=True)
class Payment:
    payment_id: str
    order_ref: str | None            # may be truncated
    captured_on: date
    gross_paise: int
    fee_paise: int
    gst_paise: int
    net_paise: int
    settlement_id: str | None
    method: str


@dataclass(frozen=True)
class Settlement:
    settlement_id: str
    settled_on: date
    payout_paise: int                # what the gateway says it sent
    utr: str | None                  # bank reference, sometimes absent
    descriptor: str = ""             # optional free-text (e.g. a ledger allocation
                                     # string) the fuzzy layer can match a bank
                                     # narration against when the id itself carries
                                     # no descriptive signal. Empty on synthetic
                                     # data, where the id IS the signal.


@dataclass(frozen=True)
class BankLine:
    txn_id: str
    value_date: date
    narration: str
    credit_paise: int
    debit_paise: int

    @property
    def signed_paise(self) -> int:
        """Credit positive, debit negative.

        Found in run 3: a settlement batch whose refunds exceed its captures has
        a NEGATIVE payout — the gateway recovers the shortfall by debiting the
        merchant. The engine originally filtered `credit_paise > 0` and so
        silently dropped every recovery debit. An engine that only reads credits
        under-reports exactly the money a merchant most wants explained."""
        return self.credit_paise - self.debit_paise


@dataclass(frozen=True)
class Refund:
    refund_id: str
    payment_id: str
    refunded_on: date
    amount_paise: int
    settlement_id: str | None        # batch the deduction was applied to


@dataclass
class Match:
    """One asserted link, with the evidence that produced it."""

    layer: str                # L1_ORDER_PAYMENT | L2_BATCH | L3_SETTLEMENT_BANK
    left_id: str
    right_ids: list[str]      # list because L3 is genuinely N:1
    rule: str                 # which rule fired — the audit anchor
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.layer, self.left_id, tuple(sorted(self.right_ids)))


@dataclass
class Exception_:
    """An honest 'I could not resolve this'."""

    entity_type: str
    entity_id: str
    reason: Reason
    detail: str
    blocking_data: dict[str, Any] = field(default_factory=dict)
    candidates_considered: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def severity(self) -> int:
        return SEVERITY[self.reason]

    @property
    def suggested_action(self) -> str:
        return SUGGESTED_ACTION[self.reason]

    @property
    def cash_impact_paise(self) -> int:
        """Rupee value at risk behind this exception.

        A controller does not triage by reason code, they triage by money. An
        exception list without an amount column forces them to open every row to
        find out which ones matter.
        """
        for k in ("payout_paise", "amount_paise", "delta_paise", "credited_paise",
                  "gross_paise", "declared_paise"):
            v = self.blocking_data.get(k)
            if isinstance(v, int) and v:
                return abs(v)
        return 0
