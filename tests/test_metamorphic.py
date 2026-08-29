"""Metamorphic tests.

Unit tests check outputs against known answers. Metamorphic tests check that a
transformation of the input produces the *relationship* to the output you'd
expect — which catches the bugs where you have no oracle for the right answer,
and where a wrong answer still looks plausible.

Each property below, if violated, means the engine is sensitive to something it
must be blind to: row order, entity naming, absolute dates, or the presence of
records that net to nothing.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from recon import evaluate, generate, normalize
from recon.match import MatcherConfig, reconcile


def _fresh(seed=11, difficulty=1.0, **gkw) -> Path:
    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(300, seed, difficulty, **gkw), d)
    return d


def _run(d: Path, **cfg):
    data, rejects = normalize.load_all(d)
    assert rejects == []
    truth = json.loads((d / "truth.json").read_text())
    r = reconcile(data, MatcherConfig(**cfg))
    return r, truth, evaluate.match_rate(r, truth)


def _signature(r):
    """Order-independent fingerprint of the engine's decisions."""
    return (sorted(m.key() for m in r.matches),
            sorted((e.entity_id, e.reason.value) for e in r.exceptions))


def _rewrite(d: Path, name: str, fn) -> None:
    path = d / name
    rows = list(csv.DictReader(path.open()))
    rows = fn(rows)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------- idempotence
def test_running_twice_is_identical():
    d = _fresh()
    a, _, _ = _run(d)
    b, _, _ = _run(d)
    assert _signature(a) == _signature(b)


# ------------------------------------------------------------ row order
def test_row_order_does_not_change_the_answer():
    """A reconciliation that depends on CSV row order is not reconciliation.
    This is the property most likely to be quietly violated by a greedy matcher
    or an unstable sort inside the assignment step."""
    d = _fresh()
    base, _, base_mr = _run(d)
    d2 = Path(tempfile.mkdtemp())
    shutil.copytree(d, d2, dirs_exist_ok=True)
    rng = random.Random(3)
    for f in ("orders.csv", "payments.csv", "bank.csv", "settlements.csv", "refunds.csv"):
        _rewrite(d2, f, lambda rows: rng.sample(rows, len(rows)))
    shuf, _, shuf_mr = _run(d2)
    assert base_mr["settlement_match_rate_pct"] == shuf_mr["settlement_match_rate_pct"]
    assert base_mr["false_match_rate_pct"] == shuf_mr["false_match_rate_pct"]
    assert _signature(base) == _signature(shuf)


# ------------------------------------------------------------ date translation
def test_shifting_every_date_changes_nothing():
    """Only relative date gaps matter. Sensitivity to absolute dates would mean
    a hard-coded window boundary or an epoch assumption somewhere."""
    d = _fresh()
    _, _, base_mr = _run(d)
    d2 = Path(tempfile.mkdtemp())
    shutil.copytree(d, d2, dirs_exist_ok=True)

    def shift(rows, col, fmt="%Y-%m-%d"):
        from datetime import datetime
        for r in rows:
            dt = datetime.strptime(r[col], fmt).date() + timedelta(days=97)
            r[col] = dt.strftime(fmt)
        return rows

    _rewrite(d2, "orders.csv", lambda rs: shift(rs, "created_at"))
    _rewrite(d2, "payments.csv", lambda rs: shift(rs, "captured_at"))
    _rewrite(d2, "refunds.csv", lambda rs: shift(rs, "refunded_at"))
    _rewrite(d2, "settlements.csv", lambda rs: shift(rs, "settled_at"))
    _rewrite(d2, "bank.csv", lambda rs: shift(rs, "value_date", "%d-%m-%Y"))
    _, _, mr2 = _run(d2)
    assert base_mr["settlement_match_rate_pct"] == mr2["settlement_match_rate_pct"]
    assert base_mr["false_match_rate_pct"] == mr2["false_match_rate_pct"]


# ------------------------------------------------------------ money formatting
def test_amount_formatting_is_irrelevant():
    """Lakh grouping, western grouping and bare decimals must reconcile
    identically. Guards the ingestion boundary against format-dependent parsing."""
    from recon.money import fmt
    d = _fresh()
    base, _, base_mr = _run(d)
    d2 = Path(tempfile.mkdtemp())
    shutil.copytree(d, d2, dirs_exist_ok=True)

    def reformat(rows):
        from recon.money import parse_paise
        for r in rows:
            if r["credit"].strip():
                r["credit"] = fmt(parse_paise(r["credit"])).replace("₹", "")
            if r["debit"].strip():
                r["debit"] = f"INR {parse_paise(r['debit'])/100:.2f}"
        return rows

    _rewrite(d2, "bank.csv", reformat)
    other, _, mr2 = _run(d2)
    assert _signature(base) == _signature(other)
    assert base_mr == mr2


# -------------------------------------------------------- irrelevant additions
def test_a_fully_refunded_payment_does_not_disturb_other_settlements():
    """Adding a capture that is refunded in full within the same batch nets to
    zero. Every OTHER settlement must reconcile exactly as before. If it does
    not, some rule is leaking state across candidates."""
    d = _fresh()
    base, _, _ = _run(d)
    base_l3 = {m.left_id for m in base.matches if m.layer == "L3_SETTLEMENT_BANK"}

    d2 = Path(tempfile.mkdtemp())
    shutil.copytree(d, d2, dirs_exist_ok=True)
    payments = list(csv.DictReader((d2 / "payments.csv").open()))
    victim = next(p for p in payments if p["settlement_id"])
    gross, fee, gst = "1000.00", "20.00", "3.60"
    payments.append({"payment_id": "pay_zzz_net0", "order_ref": "",
                     "captured_at": victim["captured_at"], "gross": gross,
                     "fee": fee, "gst_on_fee": gst, "net": "976.40",
                     "settlement_id": victim["settlement_id"], "method": "upi"})
    with (d2 / "payments.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(payments[0].keys()))
        w.writeheader(); w.writerows(payments)
    refunds = list(csv.DictReader((d2 / "refunds.csv").open()))
    refunds.append({"refund_id": "rfnd_zzz", "payment_id": "pay_zzz_net0",
                    "refunded_at": victim["captured_at"], "amount": "976.40",
                    "settlement_id": victim["settlement_id"]})
    with (d2 / "refunds.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(refunds[0].keys()))
        w.writeheader(); w.writerows(refunds)

    after, _, _ = _run(d2)
    after_l3 = {m.left_id for m in after.matches if m.layer == "L3_SETTLEMENT_BANK"}
    assert base_l3 == after_l3


# ------------------------------------------------------------------ scaling
def test_scaling_every_amount_preserves_the_match_structure():
    """Multiply all money by 10. Relative relationships are unchanged, so the
    same settlements must reconcile. Catches any absolute magnitude baked into
    a rule threshold."""
    from recon.money import fmt, parse_paise
    d = _fresh()
    base, _, _ = _run(d)
    base_pairs = {(m.left_id, tuple(sorted(m.right_ids)))
                  for m in base.matches if m.layer == "L3_SETTLEMENT_BANK"}
    d2 = Path(tempfile.mkdtemp())
    shutil.copytree(d, d2, dirs_exist_ok=True)

    def scale(cols):
        def go(rows):
            for r in rows:
                for c in cols:
                    if r.get(c, "").strip():
                        r[c] = f"{parse_paise(r[c]) * 10 / 100:.2f}"
            return rows
        return go

    _rewrite(d2, "orders.csv", scale(["amount"]))
    _rewrite(d2, "payments.csv", scale(["gross", "fee", "gst_on_fee", "net"]))
    _rewrite(d2, "refunds.csv", scale(["amount"]))
    _rewrite(d2, "settlements.csv", scale(["payout_amount"]))
    _rewrite(d2, "bank.csv", scale(["credit", "debit"]))
    after, _, _ = _run(d2)
    after_pairs = {(m.left_id, tuple(sorted(m.right_ids)))
                   for m in after.matches if m.layer == "L3_SETTLEMENT_BANK"}
    assert base_pairs == after_pairs


# --------------------------------------------------------- conservation of rows
def test_no_bank_line_is_both_matched_and_excepted_as_unexplained():
    """A line cannot simultaneously be reconciled and unexplained. Overlap here
    would mean the totals in the report double-count."""
    d = _fresh()
    r, _, _ = _run(d)
    matched = {t for m in r.matches if m.layer == "L3_SETTLEMENT_BANK" for t in m.right_ids}
    unexplained = {e.entity_id for e in r.exceptions
                   if e.reason.value == "UNEXPLAINED_BANK_CREDIT"}
    assert not (matched & unexplained)


@pytest.mark.parametrize("mode", [{"multiway": 1.0}, {"unmodeled": 1.0}, {}])
def test_uniqueness_gate_holds_under_every_generator_mode(mode):
    """Any subset-sum group the engine asserts must have been verified unique."""
    d = _fresh(**mode)
    r, _, _ = _run(d)
    for m in r.matches:
        if "SUBSET_SUM" in m.rule:
            assert m.evidence.get("uniqueness_verified") is True
