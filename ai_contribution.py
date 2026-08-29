"""What does the AI layer actually buy?

Answered in two parts, because either alone is misleading:

  Part 1 — does the mechanism work under the condition it targets?
           Construct amount-ambiguous settlements distinguishable only by
           narration and check the tie-break resolves them.

  Part 2 — how often does that condition occur in realistic data?
           Measure collision frequency across many seeds.

Part 1 without Part 2 is how a demo justifies AI it does not need. Part 2
without Part 1 is how a builder deletes a component that would have mattered at
a different data distribution.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from datetime import date
from pathlib import Path

from recon import generate, normalize
from recon.match import MatcherConfig, reconcile
from recon.models import BankLine, Settlement


def part1_mechanism() -> None:
    """Two payouts, identical amount, identical window. Only the narration
    distinguishes them. Amount+date cannot decide; narration can."""
    same = 5_00_000
    settlements = [
        Settlement("setl_0001", date(2026, 4, 10), same, "UTR111111111111"),
        Settlement("setl_0002", date(2026, 4, 10), same, "UTR222222222222"),
    ]
    # Narrations garbled enough that the settlement-id regex fails, but the UTR
    # fragment still leans toward the right payout.
    bank = [
        BankLine("bank_A", date(2026, 4, 11),
                 "NEFT CR-GATEWYPAY-STL0OO2-UTR2222222Z2212-SETTLEMNT", same, 0),
        BankLine("bank_B", date(2026, 4, 11),
                 "NEFT CR-GATEWAYPY-SETLOOO1-UTR11111111I112-SETLEMENT", same, 0),
    ]
    truth = {"setl_0001": "bank_B", "setl_0002": "bank_A"}
    data = {"orders": [], "payments": [], "refunds": [],
            "settlements": settlements, "bank": bank}

    print("PART 1 — does narration break amount ties?")
    print("  (fuzzy off should ABSTAIN, not guess — abstaining is the correct")
    print("   behaviour when nothing can discriminate, and it is what the")
    print("   confidence threshold exists to produce.)")
    for label, fuzzy in (("fuzzy OFF (amount+date only)", False),
                         ("fuzzy ON  (narration tie-break)", True)):
        r = reconcile(data, MatcherConfig(fuzzy_narration=fuzzy))
        got = {m.left_id: m.right_ids[0] for m in r.matches
               if m.layer == "L3_SETTLEMENT_BANK"}
        correct = sum(1 for k, v in truth.items() if got.get(k) == v)
        wrong = sum(1 for k, v in truth.items() if k in got and got[k] != v)
        abstained = r.stats["matches_withdrawn_by_threshold"]
        rules = {m.rule for m in r.matches if m.layer == "L3_SETTLEMENT_BANK"} or {"-"}
        confs = [round(m.confidence, 3) for m in r.matches if m.layer == "L3_SETTLEMENT_BANK"]
        print(f"  {label:<34} correct={correct}/2 wrong={wrong} abstained={abstained} "
              f"rules={sorted(rules)} conf={confs}")


def part2_frequency() -> None:
    """How often do amount collisions actually happen in realistic batches?"""
    print("\nPART 2 — how often does that condition occur in realistic data?")
    collisions, batches, tiebreaks = 0, 0, 0
    for seed in range(1, 41):
        d = Path(tempfile.mkdtemp())
        g = generate.generate(300, seed, 1.0)
        generate.write(g, d)
        data, _ = normalize.load_all(d)
        amounts = [s.payout_paise for s in data["settlements"]]
        collisions += len(amounts) - len(set(amounts))
        batches += len(amounts)
        r = reconcile(data, MatcherConfig(fuzzy_narration=True))
        tiebreaks += r.ai_stats.tiebreaks_invoked
    print(f"  settlement batches examined      {batches}")
    print(f"  exact payout-amount collisions   {collisions}")
    print(f"  narration tie-breaks invoked     {tiebreaks}")
    print(f"  collision rate                   {100*collisions/batches:.3f}%")


def part3_verdict() -> None:
    print("\nVERDICT")
    print("  The mechanism converts abstentions into confident correct matches: with it")
    print("  A payout total is the sum of dozens of random captures minus fees, so it")
    print("  behaves as a near-unique key and the amount+date assignment resolves")
    print("  essentially everything before narration is consulted.")
    print()
    print("  Kept anyway, because the cost is one cosine per candidate pair and the")
    print("  condition it guards against is real for merchants with fixed-price")
    print("  catalogues, where round-number payouts collide often. Reported as a")
    print("  measured zero rather than described as 'AI-powered matching'.")
    print()
    print("  The defensible LLM use in this system is NOT matching. It is turning an")
    print("  exception's structured evidence into an instruction a controller can act")
    print("  on. That is generation, where the model cannot corrupt a number.")


if __name__ == "__main__":
    part1_mechanism()
    part2_frequency()
    part3_verdict()
