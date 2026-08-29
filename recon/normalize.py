"""Ingestion boundary. Everything crossing this line becomes typed and integral.

Two rules:
  1. Dates are parsed against an explicit list of formats. No dateutil guessing —
     05-06-2026 is genuinely ambiguous and a silent wrong guess shifts a
     reconciliation window by months.
  2. A row that cannot be parsed is collected into `rejects`, never dropped.
     Rows that vanish at ingestion are the single easiest way to inflate a
     match rate without noticing.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .money import MoneyParseError, parse_paise
from .models import BankLine, Order, Payment, Refund, Settlement

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d")


class Rejects(list):
    """Rows that failed to parse. Reported, never silently discarded."""


def parse_date(raw: str) -> date:
    s = (raw or "").strip()
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {raw!r}")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _opt(v: str | None) -> str | None:
    v = (v or "").strip()
    return v or None


def load_all(d: Path) -> tuple[dict[str, Any], Rejects]:
    rej = Rejects()

    def guard(kind: str, ident: str, fn):
        try:
            return fn()
        except (MoneyParseError, ValueError) as exc:
            rej.append({"source": kind, "id": ident, "error": str(exc)})
            return None

    orders = [o for o in (
        guard("orders", r["order_id"], lambda r=r: Order(
            r["order_id"], parse_date(r["created_at"]), r["customer"],
            parse_paise(r["amount"]), r["status"].strip().lower(),
            _opt(r["gateway_payment_id"])))
        for r in _rows(d / "orders.csv")) if o]

    payments = [p for p in (
        guard("payments", r["payment_id"], lambda r=r: Payment(
            r["payment_id"], _opt(r["order_ref"]), parse_date(r["captured_at"]),
            parse_paise(r["gross"]), parse_paise(r["fee"]), parse_paise(r["gst_on_fee"]),
            parse_paise(r["net"]), _opt(r["settlement_id"]), r["method"]))
        for r in _rows(d / "payments.csv")) if p]

    refunds = [x for x in (
        guard("refunds", r["refund_id"], lambda r=r: Refund(
            r["refund_id"], r["payment_id"], parse_date(r["refunded_at"]),
            parse_paise(r["amount"]), _opt(r["settlement_id"])))
        for r in _rows(d / "refunds.csv")) if x]

    settlements = [s for s in (
        guard("settlements", r["settlement_id"], lambda r=r: Settlement(
            r["settlement_id"], parse_date(r["settled_at"]),
            parse_paise(r["payout_amount"]), _opt(r["utr"])))
        for r in _rows(d / "settlements.csv")) if s]

    bank = [b for b in (
        guard("bank", r["txn_id"], lambda r=r: BankLine(
            r["txn_id"], parse_date(r["value_date"]), r["narration"],
            parse_paise(r["credit"]) if r["credit"].strip() else 0,
            parse_paise(r["debit"]) if r["debit"].strip() else 0))
        for r in _rows(d / "bank.csv")) if b]

    return {"orders": orders, "payments": payments, "refunds": refunds,
            "settlements": settlements, "bank": bank}, rej
