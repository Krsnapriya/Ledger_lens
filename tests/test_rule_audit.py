"""Rule-audit tests.

The load-bearing one is `test_auditor_catches_the_known_withholding_bug`: it
proves the auditor would have found the R3.4b domain-mismatch automatically,
retroactively, without a human reading a by-rule table. The companion test
guards the obvious failure mode of any such tool — flagging things that are
actually fine.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from recon import generate, normalize
from recon.ai import SemanticResolver
from recon.match import MatcherConfig, match_settlements_bank
from recon.rule_audit import RuleStats, audit


def test_auditor_flags_a_synthetically_harmful_rule():
    """Direct unit test of the mechanism, independent of any real dataset: a
    rule that is right half the time on enough volume must be flagged."""
    from recon.models import Match

    matches = ([Match("L3_SETTLEMENT_BANK", f"good{i}", [f"g{i}"],
                      "GOOD_RULE", 0.9, {}) for i in range(20)]
               + [Match("L3_SETTLEMENT_BANK", f"bad{i}", [f"b{i}"],
                       "BAD_RULE", 0.9, {}) for i in range(20)])
    truth = {f"good{i}": True for i in range(20)}
    truth.update({f"bad{i}": (i % 2 == 0) for i in range(20)})   # 50% wrong

    def truth_fn(left_id, right_id):
        return truth.get(left_id)

    stats = audit(matches, truth_fn, min_n=10, min_precision=0.90)
    by_rule = {s.rule: s for s in stats}
    assert not by_rule["GOOD_RULE"].flagged
    assert by_rule["BAD_RULE"].flagged
    assert by_rule["BAD_RULE"].precision == 0.5


def test_auditor_does_not_flag_low_volume_noise():
    """One wrong match out of two is not evidence a rule is harmful; it is
    evidence of two data points. The min_n gate exists precisely for this."""
    from recon.models import Match
    matches = [Match("L3_SETTLEMENT_BANK", "x", ["y"], "RARE_RULE", 0.9, {})]
    truth = {"x": False}

    def truth_fn(left_id, right_id):
        return truth.get(left_id)

    stats = audit(matches, truth_fn, min_n=10, min_precision=0.90)
    assert not stats[0].flagged, "a single wrong match must not trip the flag"


def test_auditor_ignores_untruthed_pairs():
    """A pair with no ground truth available is excluded from the denominator,
    not silently counted as correct or wrong — either would be a fabrication."""
    from recon.models import Match
    matches = [Match("L3_SETTLEMENT_BANK", f"x{i}", [f"y{i}"], "R", 0.9, {})
              for i in range(15)]

    def truth_fn(left_id, right_id):
        return None   # no ground truth for anything

    stats = audit(matches, truth_fn, min_n=10, min_precision=0.90)
    assert stats == [], "no rule should be reported when nothing is verifiable"


def test_no_false_positive_on_clean_synthetic_data():
    """Every rule on synthetic baseline data is 100% precise. The auditor must
    not invent a problem that does not exist."""
    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(300, 7, 1.0), d)
    truth = json.loads((d / "truth.json").read_text())
    data, _ = normalize.load_all(d)
    cfg = MatcherConfig()
    raw, _ = match_settlements_bank(data["settlements"], data["bank"], cfg,
                                    SemanticResolver(use_llm=False))
    matches = [m for m in raw if m.confidence >= cfg.confidence_threshold]
    t3 = {(sid, txn) for sid, txns in truth["settlement_to_bank"].items()
          for txn in txns}

    def truth_fn(left_id, right_id):
        return (left_id, right_id) in t3

    stats = audit(matches, truth_fn)
    flagged = [s for s in stats if s.flagged]
    assert not flagged, f"false positive(s) on clean data: {flagged}"


def test_auditor_catches_the_known_withholding_bug_retroactively():
    """The actual regression this tool exists for. R3.4b's Indian withholding
    rates were found harmful on BenchRec by a human reading a printed table.
    This proves the auditor would have found the same thing automatically had
    it existed first — the exact scenario the external critique asked for."""
    import csv
    data_dir = Path("/tmp/benchrec")
    if not (data_dir / "BenchRec_cash_v1.0_eval.csv").exists():
        import pytest
        pytest.skip("BenchRec not downloaded in this environment")

    from recon.benchrec_eval import load_eval, load_solution

    solution = load_solution(data_dir / "BenchRec_cash_v1.0_solution.csv")
    ledger, bank, allocation_of = load_eval(data_dir / "BenchRec_cash_v1.0_eval.csv")
    cfg = MatcherConfig(plausible_withholding_bp=(10, 100, 200, 500))  # bug present
    raw, _ = match_settlements_bank(ledger, bank, cfg, SemanticResolver(use_llm=False))
    matches = [m for m in raw if m.confidence >= cfg.confidence_threshold]

    def truth_fn(left_id, right_id):
        t = solution.get(right_id[1:])
        p = allocation_of.get(left_id)
        return None if t is None or p is None else p == t

    stats = audit(matches, truth_fn)
    by_rule = {s.rule: s for s in stats}
    assert "R3.4b_WITHHOLDING_RATE_MATCH" in by_rule
    assert by_rule["R3.4b_WITHHOLDING_RATE_MATCH"].flagged
    assert by_rule["R3.4b_WITHHOLDING_RATE_MATCH"].precision < 0.30
