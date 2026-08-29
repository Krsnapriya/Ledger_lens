"""Tests that a skeptical reviewer would want to see pass.

These are not coverage theatre. Each one pins down a property that, if it broke,
would make the reported numbers a lie:

  - money never goes through a float
  - pristine data reconciles perfectly (if it doesn't, no other number is credible)
  - the same seed produces the same answer (reproducibility is a claim, not a hope)
  - no record disappears between the CSV and the report
  - raising the confidence threshold never increases false matches
  - the AI layer degrades to local scoring instead of silently failing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from recon import evaluate, generate, normalize
from recon.ai import SemanticResolver, cosine, extract_deterministic
from recon.match import MatcherConfig, reconcile
from recon.models import Reason
from recon.money import MoneyParseError, fmt, parse_paise, pct_of


# --------------------------------------------------------------------- money
@pytest.mark.parametrize("raw,expected", [
    ("1234.50", 123450),
    ("1,23,456.78", 12345678),      # Indian lakh grouping
    ("123,456.78", 12345678),       # western grouping
    ("INR 1234.50", 123450),
    ("₹1,234.50", 123450),
    ("(1234.50)", -123450),         # accounting negative
    ("1234.50 Cr", 123450),
    ("1234.50 Dr", -123450),
    ("0.005", 1),                   # half-up at the paise boundary
    ("0.004", 0),
    (12345, 12345),
])
def test_parse_paise(raw, expected):
    assert parse_paise(raw) == expected


@pytest.mark.parametrize("raw", ["", "nan", "abc", None, "-"])
def test_parse_paise_refuses_garbage(raw):
    """A silent zero in a ledger is worse than a crash."""
    with pytest.raises(MoneyParseError):
        parse_paise(raw)


def test_no_float_drift():
    """The canonical float trap, in ledger form."""
    assert parse_paise("0.10") + parse_paise("0.20") == parse_paise("0.30")


def test_pct_of_is_half_up_and_integral():
    assert pct_of(100_00, 200) == 200          # 2% of ₹100 = ₹2.00
    assert pct_of(333, 1800) == 60             # 59.94 -> 60, half-up
    assert isinstance(pct_of(12345, 217), int)


def test_fmt_indian_grouping():
    assert fmt(12345678) == "₹1,23,456.78"
    assert fmt(-123450) == "-₹1,234.50"


# ------------------------------------------------------------------- helpers
def run(seed=7, difficulty=1.0, orders=300, **cfg_kw):
    d = Path(tempfile.mkdtemp())
    g = generate.generate(n_orders=orders, seed=seed, difficulty=difficulty)
    generate.write(g, d)
    truth = json.loads((d / "truth.json").read_text())
    data, rejects = normalize.load_all(d)
    result = reconcile(data, MatcherConfig(**cfg_kw))
    return g, data, rejects, truth, result


# ------------------------------------------------------- the sanity contract
def test_pristine_data_reconciles_perfectly():
    """If the engine cannot hit 100% on data with no anomalies, every other
    number it reports is noise. This is the load-bearing test."""
    for seed in range(1, 11):
        _, _, rej, truth, r = run(seed=seed, difficulty=0.0)
        mr = evaluate.match_rate(r, truth)
        assert mr["settlement_match_rate_pct"] == 100.0, f"seed {seed}"
        assert mr["false_match_rate_pct"] == 0.0, f"seed {seed}"
        assert not rej


def test_no_false_matches_at_default_difficulty():
    """The number that decides whether this is usable. Held across seeds."""
    for seed in range(1, 16):
        _, _, _, truth, r = run(seed=seed, difficulty=1.0)
        assert evaluate.match_rate(r, truth)["false_match_rate_pct"] == 0.0, f"seed {seed}"


def test_reported_match_rate_is_not_cherry_picked():
    """Mean across seeds must stay high, and the worst seed must stay credible."""
    rates = [evaluate.match_rate(run(seed=s)[4], run(seed=s)[3])["settlement_match_rate_pct"]
             for s in range(1, 11)]
    assert sum(rates) / len(rates) > 97.0
    assert min(rates) > 85.0


# ----------------------------------------------------------- reproducibility
def test_same_seed_same_answer():
    a = run(seed=42)
    b = run(seed=42)
    ka = sorted(m.key() for m in a[4].matches)
    kb = sorted(m.key() for m in b[4].matches)
    assert ka == kb
    assert ([(e.entity_id, e.reason) for e in a[4].exceptions]
            == [(e.entity_id, e.reason) for e in b[4].exceptions])


def test_different_seeds_differ():
    """Guards against a generator that ignores its seed."""
    assert (sorted(m.key() for m in run(seed=1)[4].matches)
            != sorted(m.key() for m in run(seed=2)[4].matches))


# --------------------------------------------------------- nothing disappears
def test_every_generated_row_survives_ingestion():
    g, data, rejects, _, _ = run()
    assert len(data["orders"]) == len(g.orders)
    assert len(data["payments"]) == len(g.payments)
    assert len(data["settlements"]) == len(g.settlements)
    assert len(data["bank"]) == len(g.bank)
    assert rejects == []


def test_every_settlement_is_matched_or_excepted():
    """No settlement may vanish. It is either reconciled or it is on the ledger
    of things we could not reconcile."""
    _, data, _, _, r = run()
    matched = {m.left_id for m in r.matches if m.layer == "L3_SETTLEMENT_BANK"}
    excepted = {e.entity_id for e in r.exceptions}
    for s in data["settlements"]:
        assert s.settlement_id in matched or s.settlement_id in excepted


def test_every_bank_line_is_matched_or_excepted():
    _, data, _, _, r = run()
    matched = {t for m in r.matches if m.layer == "L3_SETTLEMENT_BANK" for t in m.right_ids}
    excepted = {e.entity_id for e in r.exceptions}
    for b in data["bank"]:
        if b.signed_paise == 0:
            continue
        assert b.txn_id in matched or b.txn_id in excepted, b.txn_id


# ------------------------------------------------------------- the safety knob
def test_raising_threshold_never_increases_false_matches():
    """Monotonicity of the abstention knob. If this fails, the confidence scores
    are not ordered by reliability and the curve in the report is meaningless."""
    _, _, _, truth, r = run(confidence_threshold=0.0)
    curve = evaluate.abstention_curve(r.matches, truth)
    fmrs = [row["false_match_rate"] for row in curve]
    assert all(b <= a + 1e-9 for a, b in zip(fmrs, fmrs[1:])), fmrs


def test_withdrawn_matches_become_visible_exceptions():
    """A withheld match must not silently disappear; it becomes human work."""
    _, _, _, _, r = run(confidence_threshold=0.999)
    withdrawn = [e for e in r.exceptions if e.reason is Reason.BELOW_CONFIDENCE_THRESHOLD]
    assert r.stats["matches_withdrawn_by_threshold"] == len(withdrawn)
    assert withdrawn, "threshold of 0.999 should withhold something"


# ----------------------------------------------------------- arithmetic layer
def test_arithmetic_exceptions_are_exact_not_fuzzy():
    """Fee/GST/batch checks must be deterministic: same input, zero variance,
    confidence exactly 1.0. These are rules, not judgements."""
    _, _, _, _, r = run()
    arith = [e for e in r.exceptions if e.reason in {
        Reason.FEE_RATE_MISMATCH, Reason.GST_MISMATCH,
        Reason.NET_ARITHMETIC_MISMATCH, Reason.BATCH_ARITHMETIC_MISMATCH}]
    assert arith
    assert all(e.confidence == 1.0 for e in arith)


def test_negative_payout_lands_as_a_debit():
    """A batch whose refunds exceed captures is money going out, and must still
    reconcile. Regression test for the credit-only filter bug."""
    found = False
    for seed in range(1, 40):
        _, data, _, truth, r = run(seed=seed, difficulty=0.0)
        neg = [s for s in data["settlements"] if s.payout_paise < 0]
        if not neg:
            continue
        found = True
        matched = {m.left_id for m in r.matches if m.layer == "L3_SETTLEMENT_BANK"}
        for s in neg:
            assert s.settlement_id in matched, f"seed {seed} {s.settlement_id}"
    assert found, "no negative-payout batch generated in 40 seeds; test is vacuous"


# --------------------------------------------------------------- the AI layer
def test_deterministic_extraction_beats_the_model_when_it_can():
    ex = extract_deterministic("NEFT CR-GATEWAYPAY-SETL_0012-UTR123456789012-SETTLEMENT")
    assert ex.settlement_id == "setl_0012"
    assert ex.utr == "UTR123456789012"


def test_cosine_is_robust_to_truncation_and_typos():
    a = "NEFT CR-GATEWAYPAY-SETL_0012-UTR123456789012-SETTLEMENT"
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, a[:30]) > 0.5           # truncated
    assert cosine(a, "RTGS DR-SALARY-PAYROLL-BATCH-9931") < 0.35


def test_llm_absent_degrades_loudly_not_silently():
    """Requesting the LLM backend with no key must fall back AND say so. A
    fallback that reports success is how a demo claims AI it never used."""
    import os
    prev = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        r = SemanticResolver(use_llm=True)
        assert r.stats.backend == "local-chargram"
        assert any("SKIPPED" in n for n in r.stats.notes)
        assert r.stats.llm_calls == 0
    finally:
        if prev:
            os.environ["ANTHROPIC_API_KEY"] = prev


def test_ai_layer_cannot_invent_a_settlement():
    """The model only ever re-ranks an offered candidate set, so the worst it can
    do is pick the wrong one of N — never conjure an id that does not exist."""
    r = SemanticResolver(use_llm=False)
    ranked = r.rank("GARBLED NARRATION XYZ", [("setl_0001", "a"), ("setl_0002", "b")])
    assert {sid for sid, _ in ranked} <= {"setl_0001", "setl_0002"}


def test_fuzzy_layer_is_amount_constrained():
    """Turning the fuzzy layer on must not introduce false matches. This is the
    guard against 'the embedding said so' overriding arithmetic."""
    for seed in range(1, 8):
        _, _, _, truth, on = run(seed=seed, fuzzy_narration=True)
        assert evaluate.match_rate(on, truth)["false_match_rate_pct"] == 0.0


# ------------------------------------------------------------------ evidence
def test_every_match_carries_its_rule_and_evidence():
    """An unauditable match is an unusable match."""
    _, _, _, _, r = run()
    for m in r.matches:
        assert m.rule and m.layer
        assert 0.0 <= m.confidence <= 1.0
        assert m.evidence


def test_every_exception_carries_a_code_and_an_action():
    _, _, _, _, r = run()
    for e in r.exceptions:
        assert isinstance(e.reason, Reason)
        assert e.detail and e.suggested_action
        assert e.severity > 0


# ==================== Gap 6 — ambiguity gate soundness ========================
def test_ambiguity_gate_survives_witness_saturation():
    """The meet-in-the-middle enumerator keeps at most `witness_cap` witnesses per
    half-sum. If a target is reachable more ways than that within one half, a
    second global solution could in principle be dropped and an ambiguous target
    would be asserted as unique — the exact failure the gate exists to prevent.

    Ten identical amounts make every same-size subset collide, saturating the
    witness cap many times over. The gate must still refuse.
    """
    from recon import subsetsum
    amounts = [10] * 10
    unique, sol = subsetsum.is_unique(amounts, 40, max_k=8, tol=0)
    assert sol is not None, "no solution found where 210 exist"
    assert not unique, "ambiguity gate missed a massively ambiguous target"


def test_ambiguity_gate_still_accepts_a_genuinely_unique_target():
    """Guard the other direction: refusing everything is not a fix."""
    from recon import subsetsum
    amounts = [3, 17, 41, 97, 233]
    unique, sol = subsetsum.is_unique(amounts, 3 + 41, max_k=4, tol=0)
    assert unique and sol is not None
    assert sorted(amounts[i] for i in sol) == [3, 41]


def test_subset_sum_returns_distinct_solutions_not_repeats():
    """Two 'solutions' that are the same index set must not be counted as an
    ambiguity, or every match would be refused."""
    from recon import subsetsum
    sols = subsetsum.subset_sum([5, 7, 11], 12, max_k=3, tol=0, solution_cap=2)
    assert len(sols) == 1
    assert len({tuple(sorted(s)) for s in sols}) == len(sols)
