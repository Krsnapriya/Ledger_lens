"""The finance controller agent — closes the books, then works its own queue.

Reconciling once is not closing the books. A controller reconciles, looks at what
did not resolve, decides which of those they can chase and which need someone
else, takes the cheap actions themselves, and escalates the rest with a specific
ask. This runs that loop autonomously to a stopping condition.

    observe → decide → act → re-observe → repeat → report

**The action space is deliberately narrow, and none of it can create a match.**
That is the whole design constraint. An agent that can force an attribution can
manufacture a false match, and the false-match rate is the number this project
exists to protect. So the agent may re-run the engine under different bounds, it
may accept an exception that is already explained, it may write off below a
materiality threshold, and it may escalate. It may not assert that a payout
landed. Every attribution decision stays with the deterministic engine.

Each action is recorded in the same hash-chained decision log as everything else,
so an agent run replays bit-identically like any other.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recon import generate, normalize, report
from recon.live import LiveController
from recon.match import MatcherConfig
from recon.models import Reason
from recon.money import fmt

# ---------------------------------------------------------------- action space
# Arithmetic complaints are DISPUTES about an amount, not failures of
# attribution. The settlement they name is usually reconciled perfectly well.
# Removing it to "close" the exception destroys a good match and strands its bank
# credit — measured at 50 orders, one such write-off raised exposure by 1.99
# lakh in a single step.
ARITHMETIC_DISPUTES = frozenset({
    Reason.FEE_RATE_MISMATCH, Reason.GST_MISMATCH,
    Reason.NET_ARITHMETIC_MISMATCH, Reason.BATCH_ARITHMETIC_MISMATCH,
    Reason.AMOUNT_MISMATCH,
})

RETRY_WIDER = "RETRY_WIDER_WINDOW"   # re-run the engine with a wider look-back
ACCEPT = "ACCEPT_EXPLAINED"          # already explained; close it, no money moves
WRITE_OFF = "WRITE_OFF_IMMATERIAL"   # below the materiality threshold
ESCALATE = "ESCALATE_TO_HUMAN"       # needs evidence the agent cannot obtain

# Reason code -> what a controller would actually do about it.
#
# The escalations are not a shrug: each one names the artefact a human has to go
# and get. "Request the gateway breakup file" is an action; "unresolved" is not.
POLICY: dict[Reason, tuple[str, str]] = {
    Reason.SETTLEMENT_NOT_IN_BANK: (
        RETRY_WIDER, "payout may have landed outside the matching window"),
    Reason.UNEXPLAINED_BANK_CREDIT: (
        RETRY_WIDER, "credit may belong to a payout outside the window"),
    Reason.AMBIGUOUS_SUBSET: (
        ESCALATE, "two settlement groups sum to this credit; request the gateway breakup file"),
    Reason.DUPLICATE_CLAIM: (
        ESCALATE, "grouping depends on an earlier decision; request the breakup file"),
    Reason.POOL_CAP_EXCEEDED: (
        ESCALATE, "too many candidates to decide safely; narrow the period with the gateway"),
    Reason.BATCH_ARITHMETIC_MISMATCH: (
        ESCALATE, "declared payout does not foot; request the settlement breakup"),
    Reason.FEE_RATE_MISMATCH: (
        ESCALATE, "fee off contracted MDR; raise a fee dispute with the gateway"),
    Reason.GST_MISMATCH: (
        ESCALATE, "GST not 18% of fee; request the corrected tax invoice"),
    Reason.NET_ARITHMETIC_MISMATCH: (
        ESCALATE, "gateway net does not equal gross minus fee minus GST"),
    Reason.AMOUNT_MISMATCH: (
        ACCEPT, "shortfall matches a plausible withholding rate; explained"),
    Reason.SPLIT_SPANS_LONG_WINDOW: (
        ACCEPT, "payout arrived in tranches; working capital held, reported"),
    Reason.DUPLICATE_BANK_CREDIT: (
        ACCEPT, "double posting identified; no money to recover"),
    Reason.BELOW_CONFIDENCE_THRESHOLD: (
        ESCALATE, "best candidate below threshold; human confirm or reject"),
    Reason.ORDER_PAYMENT_AMBIGUOUS: (
        ESCALATE, "truncated reference matches several orders; human tie-break"),
    Reason.PAYMENT_NO_ORDER: (
        ESCALATE, "capture with no internal order; check for an out-of-band link"),
    Reason.ORDER_NO_PAYMENT: (
        ACCEPT, "order never captured; no money moved"),
    Reason.OVERRIDE_LEAK: (
        ESCALATE, "isolation boundary breached; do not trust this run"),
}


@dataclass
class Action:
    step: int
    kind: str
    entity_id: str
    reason_code: str
    rationale: str
    cash_at_risk_paise: int
    recovered_paise: int = 0


@dataclass
class AgentReport:
    steps_used: int = 0
    actions: list[Action] = field(default_factory=list)
    opening_exposure_paise: int = 0
    closing_exposure_paise: int = 0
    # Per-action (before minus after) summed across the run. Actions interleave
    # on a queue that other actions also change, so this OVERLAPS and is not a
    # net figure. The headline is opening minus closing; this is reported beside
    # it as gross movement, never in place of it.
    recovered_paise: int = 0
    escalated_paise: int = 0
    written_off_paise: int = 0
    accepted_paise: int = 0
    harmful_actions: int = 0
    stop_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps_used": self.steps_used,
            "stop_reason": self.stop_reason,
            "harmful_actions": self.harmful_actions,
            "opening_exposure_paise": self.opening_exposure_paise,
            "closing_exposure_paise": self.closing_exposure_paise,
            "net_exposure_change_paise": (self.opening_exposure_paise
                                          - self.closing_exposure_paise),
            "gross_moved_by_widening_paise": self.recovered_paise,
            "escalated_paise": self.escalated_paise,
            "written_off_paise": self.written_off_paise,
            "accepted_paise": self.accepted_paise,
            "actions": [vars(a) for a in self.actions],
        }


class FinanceControllerAgent:
    """Works the exception queue until nothing is left that it may act on."""

    def __init__(self, controller: LiveController, materiality_paise: int = 100_000,
                 max_steps: int = 250, widen_to_days: int = 40) -> None:
        self.ctl = controller
        self.materiality = materiality_paise
        self.max_steps = max_steps
        self.widen_to_days = widen_to_days
        self.report = AgentReport()
        self._retried: set[str] = set()
        self._handled: set[tuple[str, str]] = set()

    # ------------------------------------------------------------- observe
    def observe(self) -> list:
        """Current queue, ordered the way a controller triages: money first."""
        return sorted(self.ctl.state.exceptions,
                      key=lambda e: (-e.cash_impact_paise, -e.severity, e.entity_id))

    def _is_matched(self, entity_id: str) -> bool:
        """Is this record currently carrying an attribution?"""
        for m in self.ctl.state.matches:
            if m.left_id == entity_id or entity_id in m.right_ids:
                return True
        return False

    def exposure(self) -> int:
        """Rupees the books cannot currently account for."""
        return sum(e.cash_impact_paise for e in self.ctl.state.exceptions
                   if e.reason in {Reason.SETTLEMENT_NOT_IN_BANK,
                                   Reason.UNEXPLAINED_BANK_CREDIT,
                                   Reason.AMBIGUOUS_SUBSET,
                                   Reason.POOL_CAP_EXCEEDED,
                                   Reason.BELOW_CONFIDENCE_THRESHOLD})

    # -------------------------------------------------------------- decide
    def decide(self, queue) -> tuple[Any, str, str] | None:
        """Highest cash at risk that the agent has an unused policy for."""
        for e in queue:
            key = (e.entity_id, e.reason.value)
            if key in self._handled:
                continue
            kind, rationale = POLICY.get(e.reason, (ESCALATE, "no policy; human review"))
            if kind == RETRY_WIDER and e.entity_id in self._retried:
                # One retry per record. A second is not a new idea, it is a loop.
                kind, rationale = ESCALATE, "still unresolved after a wider search"
            if kind == ESCALATE and e.cash_impact_paise < self.materiality:
                # Write-off removes the record from every candidate pool, so it
                # is only ever valid for a record nothing depends on. Two things
                # disqualify it, both found by measurement rather than reasoning:
                #
                #   the record is currently part of a match — removing it
                #   destroys that attribution and strands its counterparty;
                #
                #   the complaint is arithmetic — a fee 25bp off contract does
                #   not mean the payout did not arrive, and closing it by
                #   deleting the settlement is not a write-off, it is a deletion.
                #
                # Immaterial disputes are simply noted and left closed.
                if e.reason in ARITHMETIC_DISPUTES:
                    kind = ACCEPT
                    rationale = (f"dispute below materiality of {fmt(self.materiality)}; "
                                 "noted, attribution unaffected")
                elif self._is_matched(e.entity_id):
                    rationale = ("record is part of a live match; removing it "
                                 "would strand its counterparty")
                else:
                    kind = WRITE_OFF
                    rationale = (f"below materiality of {fmt(self.materiality)}; "
                                 "closed without escalation")
            return e, kind, rationale
        return None

    # ----------------------------------------------------------------- act
    def act(self, exc, kind: str, rationale: str) -> Action:
        before = self.exposure()
        action = Action(step=self.report.steps_used + 1, kind=kind,
                        entity_id=exc.entity_id, reason_code=exc.reason.value,
                        rationale=rationale, cash_at_risk_paise=exc.cash_impact_paise)

        if kind == RETRY_WIDER:
            # Re-run the SAME engine under wider date bounds. Every gate still
            # applies, so this cannot invent an attribution — it can only find one
            # that was sitting outside the window.
            #
            # BOTH knobs move. The first version widened only the consolidation
            # look-back, which governs subset-sum; a credit displaced by
            # value-date skew is missed by the 1:1 rules, which are governed by
            # date_window_days. Widening the wrong bound recovered nothing.
            self._retried.add(exc.entity_id)
            prev_lb = self.ctl.cfg.subset_sum_lookback_days
            prev_dw = self.ctl.cfg.date_window_days
            self.ctl.cfg.subset_sum_lookback_days = self.widen_to_days
            self.ctl.cfg.date_window_days = self.widen_to_days
            try:
                self.ctl.resolve_window([exc.entity_id], cause=f"agent:{RETRY_WIDER}")
            finally:
                self.ctl.cfg.subset_sum_lookback_days = prev_lb
                self.ctl.cfg.date_window_days = prev_dw

            after = self.exposure()
            if after < before:
                action.recovered_paise = before - after
                self.report.recovered_paise += before - after
            else:
                # The action did not help, and a wider window can make things
                # worse: more candidates means more ambiguity means more
                # abstention. An agent that acts without checking the outcome is
                # not closing a loop, so the widened result is discarded and the
                # record re-solved under the original bounds. The engine is
                # deterministic, so this restores the prior state exactly.
                self.ctl.resolve_window([exc.entity_id],
                                        cause=f"agent:{RETRY_WIDER}_REVERTED")
                action.kind = "RETRY_WIDER_REVERTED"
                action.rationale = ("wider search did not reduce exposure; "
                                    "reverted to the original bounds")

        elif kind == WRITE_OFF:
            try:
                self.ctl.apply_override(exc.entity_id, "agent_write_off", rationale,
                                        cash_impact_paise=0)
                self.report.written_off_paise += exc.cash_impact_paise
            except ValueError:
                action.kind = ESCALATE
                action.rationale = "not eligible for override; escalated instead"
                self.report.escalated_paise += exc.cash_impact_paise

        elif kind == ESCALATE:
            self.report.escalated_paise += exc.cash_impact_paise

        elif kind == ACCEPT:
            self.report.accepted_paise += exc.cash_impact_paise

        after_all = self.exposure()
        if after_all > before:
            # An agent that can make the books worse is worse than no agent. This
            # is a reported invariant, not an assertion, because the honest
            # response to "my action hurt" is to record it and stop taking that
            # action — not to crash a close that is otherwise sound.
            action.rationale += (f" [WARNING: raised exposure by "
                                 f"{fmt(after_all - before)}]")
            self.report.harmful_actions += 1
        self._handled.add((exc.entity_id, exc.reason.value))
        self.report.steps_used += 1
        self.report.actions.append(action)
        return action

    # ----------------------------------------------------------------- loop
    def run(self) -> AgentReport:
        self.report.opening_exposure_paise = self.exposure()
        while True:
            if self.report.steps_used >= self.max_steps:
                self.report.stop_reason = f"step budget of {self.max_steps} exhausted"
                break
            decision = self.decide(self.observe())
            if decision is None:
                self.report.stop_reason = "no exception remains that the agent may act on"
                break
            self.act(*decision)
        self.report.closing_exposure_paise = self.exposure()
        return self.report


# ------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recon.agent")
    ap.add_argument("--orders", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--materiality", type=int, default=100_000,
                    help="paise below which the agent closes without escalating")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--unmodeled", type=float, default=0.0,
                    help="inject anomaly classes the engine was not built for")
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    a = ap.parse_args(argv)

    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(a.orders, a.seed, 1.0,
                                     unmodeled=a.unmodeled), d)
    data, _ = normalize.load_all(d)
    ctl = LiveController(data, MatcherConfig())
    agent = FinanceControllerAgent(ctl, materiality_paise=a.materiality,
                                   max_steps=a.max_steps)
    rep = agent.run()

    print("=" * 78)
    print(f"  AGENT RUN   {a.orders} orders · seed {a.seed} · "
          f"materiality {fmt(a.materiality)}")
    print("=" * 78)
    net = rep.opening_exposure_paise - rep.closing_exposure_paise
    print(f"  opening unexplained exposure   {fmt(rep.opening_exposure_paise)}")
    print(f"  closing unexplained exposure   {fmt(rep.closing_exposure_paise)}")
    print(f"  NET exposure change            {fmt(net)}"
          f"{'  (reduced)' if net > 0 else ('  (unchanged)' if net == 0 else '  (increased)')}")
    print(f"  gross moved by widening        {fmt(rep.recovered_paise)}   "
          f"per-action, overlapping")
    print(f"  accepted as already explained  {fmt(rep.accepted_paise)}")
    print(f"  written off below materiality  {fmt(rep.written_off_paise)}")
    print(f"  escalated to a human           {fmt(rep.escalated_paise)}")
    print(f"  steps used                     {rep.steps_used} "
          f"({rep.stop_reason})")
    print(f"  actions that raised exposure   {rep.harmful_actions}"
          f"{'  <- investigate' if rep.harmful_actions else ''}")
    print("=" * 78)
    counts: dict[str, int] = {}
    for act in rep.actions:
        counts[act.kind] = counts.get(act.kind, 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<24} {v}")
    print("=" * 78)
    print("  escalation queue, highest cash at risk first:")
    esc = [x for x in rep.actions if x.kind == ESCALATE][:6]
    for x in esc:
        print(f"    {fmt(x.cash_at_risk_paise):>16}  {x.reason_code:<26} {x.rationale[:44]}")
    if not esc:
        print("    (empty)")
    print("=" * 78)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "agent_run.json").write_text(json.dumps(rep.as_dict(), indent=2, default=str))
    report.write_exceptions_csv(a.out / "agent_exceptions.csv", ctl.state.exceptions)
    ok, msg = ctl.log.verify_chain()
    print(f"  decision log: {len(ctl.log.events)} events · chain {msg} · "
          f"log cash == state cash: {ctl.log.cash_position() == ctl.state.reconciled_paise()}")
    print(f"  artifacts -> {a.out}/agent_run.json, agent_exceptions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
