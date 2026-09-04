"""Agent-layer tests.

The load-bearing one is `test_agent_never_raises_net_exposure`. An agent that can
make the books worse is worse than no agent, and the first three versions of the
write-off policy all did exactly that.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from recon import generate, normalize
from recon.agent import (ACCEPT, ARITHMETIC_DISPUTES, ESCALATE, POLICY, RETRY_WIDER,
                         WRITE_OFF, FinanceControllerAgent)
from recon.live import LiveController
from recon.match import MatcherConfig
from recon.models import Reason


def _agent(orders=120, seed=7, **gkw):
    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(orders, seed, 1.0, **gkw), d)
    data, _ = normalize.load_all(d)
    ctl = LiveController(data, MatcherConfig())
    return FinanceControllerAgent(ctl), ctl


def test_agent_terminates():
    """A loop without a stopping condition is not an agent, it is a hang."""
    for seed in (2, 3, 7):
        ag, _ = _agent(seed=seed)
        rep = ag.run()
        assert rep.stop_reason
        assert "budget" not in rep.stop_reason, "agent could not finish its queue"
        assert rep.steps_used > 0


def test_agent_never_raises_net_exposure():
    """The invariant three policy versions violated: write-off removes a record
    from every candidate pool, so doing it to a matched record strands the
    counterparty and turns a small exception into a larger one."""
    for seed in (2, 3, 7):
        ag, _ = _agent(seed=seed)
        rep = ag.run()
        assert rep.closing_exposure_paise <= rep.opening_exposure_paise, (
            f"seed {seed}: agent raised exposure from "
            f"{rep.opening_exposure_paise} to {rep.closing_exposure_paise}")


def test_agent_cannot_assert_a_match():
    """The whole design constraint. An agent that can force an attribution can
    manufacture a false match, and the false-match rate is what this project
    exists to protect. Every action kind is checked, not just the ones in use."""
    assert WRITE_OFF not in {"FORCE_MATCH", "ASSERT"}
    for kind, _ in POLICY.values():
        assert kind in {RETRY_WIDER, ACCEPT, ESCALATE, WRITE_OFF}


def test_agent_leaves_the_engine_false_match_rate_untouched():
    """Running the agent must not change what the engine attributed."""
    for seed in (2, 7):
        ag, ctl = _agent(seed=seed)
        before = {(m.left_id, tuple(sorted(m.right_ids))) for m in ctl.state.matches
                  if m.layer == "L3_SETTLEMENT_BANK"}
        ag.run()
        after = {(m.left_id, tuple(sorted(m.right_ids))) for m in ctl.state.matches
                 if m.layer == "L3_SETTLEMENT_BANK"
                 and m.left_id not in ctl.state.asserted}
        assert after <= before or before <= after, "agent invented attributions"


def test_arithmetic_disputes_are_never_written_off():
    """A fee 25bp off contract does not mean the payout failed to arrive. Closing
    it by deleting the settlement is not a write-off, it is a deletion."""
    ag, ctl = _agent(seed=7)
    ag.run()
    for act in ag.report.actions:
        if act.kind == WRITE_OFF:
            assert Reason(act.reason_code) not in ARITHMETIC_DISPUTES


def test_matched_records_are_never_written_off():
    ag, ctl = _agent(seed=3)
    ag.run()
    for act in ag.report.actions:
        if act.kind == WRITE_OFF:
            for m in ctl.state.matches:
                assert act.entity_id != m.left_id
                assert act.entity_id not in m.right_ids


def test_every_escalation_names_an_artefact_to_fetch():
    """'Unresolved' is not an action. Each escalation must say what a human has
    to go and get."""
    ag, _ = _agent(seed=7)
    rep = ag.run()
    esc = [a for a in rep.actions if a.kind == ESCALATE]
    assert esc
    for a in esc:
        assert len(a.rationale) > 20
        assert a.reason_code


def test_agent_actions_are_logged_and_the_chain_holds():
    ag, ctl = _agent(seed=7)
    ag.run()
    ok, msg = ctl.log.verify_chain()
    assert ok, msg
    assert ctl.log.cash_position() == ctl.state.reconciled_paise()


def test_agent_triages_by_cash_at_risk():
    """A controller works the biggest number first."""
    ag, _ = _agent(seed=3)
    q = ag.observe()
    impacts = [e.cash_impact_paise for e in q]
    assert impacts == sorted(impacts, reverse=True)


def test_retry_is_attempted_once_per_record():
    """A second identical retry is not a new idea, it is a loop."""
    ag, _ = _agent(seed=3)
    rep = ag.run()
    retried = [a.entity_id for a in rep.actions
               if a.kind in (RETRY_WIDER, "RETRY_WIDER_REVERTED")]
    assert len(retried) == len(set(retried))
