"""Per-rule precision auditor — the mechanism the withholding-rate bug should
have been caught by, not a human reading a table.

`R3.4b`'s Indian withholding rates fired 14 times on BenchRec's USD data and
were wrong 11 of them — a 79% failure rate. It was found by printing a
by-rule breakdown and reading it. That does not generalise: the next
domain-mismatched rule, on the next real dataset, will not announce itself.

This computes precision **per rule** against a labelled sample and flags any
rule whose measured precision falls below a bar, with enough volume to be a
real signal rather than noise from three unlucky matches. It does not disable
anything automatically — a rule earning a false-positive flag on a small,
noisy sample and getting silently switched off is worse than a human missing
a real one, so the output is a recommendation with the evidence attached, not
an action.

    python -m recon.rule_audit --data /tmp/benchrec        # against BenchRec
    python -m recon.rule_audit --synthetic                 # sanity: no flags
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RuleStats:
    rule: str
    n: int
    correct: int
    wrong: int

    @property
    def precision(self) -> float:
        return self.correct / self.n if self.n else 1.0

    @property
    def flagged(self) -> bool:
        return self.n >= self.min_n and self.precision < self.min_precision

    min_n: int = 10
    min_precision: float = 0.90


def audit(matches, truth_pair_fn, min_n: int = 10, min_precision: float = 0.90
          ) -> list[RuleStats]:
    """matches: engine Match objects, already confidence-filtered (this audits
    what was ASSERTED, which is the only thing that can hurt anyone).

    truth_pair_fn(left_id, right_id) -> True | False | None. None means
    "no ground truth available for this pair" and it is excluded from the
    denominator rather than counted either way — an untruthed pair is not
    evidence of harm.
    """
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # rule -> [correct, wrong]
    for m in matches:
        for r in m.right_ids:
            verdict = truth_pair_fn(m.left_id, r)
            if verdict is None:
                continue
            agg[m.rule][0 if verdict else 1] += 1

    out = []
    for rule, (c, w) in sorted(agg.items()):
        st = RuleStats(rule, c + w, c, w, min_n=min_n, min_precision=min_precision)
        out.append(st)
    return out


def report(stats: list[RuleStats]) -> str:
    lines = [f"{'rule':<44}{'n':>6}{'precision':>11}{'flag':>8}"]
    lines.append("-" * 69)
    any_flag = False
    for s in stats:
        flag = "HARMFUL" if s.flagged else ""
        any_flag = any_flag or s.flagged
        lines.append(f"{s.rule:<44}{s.n:>6}{s.precision:>10.1%}{flag:>9}")
    lines.append("-" * 69)
    if any_flag:
        lines.append("Rule(s) above the flag threshold are candidates for disabling on "
                     "THIS distribution. This is a recommendation with evidence attached, "
                     "not an automatic action — verify before disabling, the same "
                     "discipline this repo applies to every other claim.")
    else:
        lines.append("No rule flagged: every rule with enough volume to judge clears "
                     f"{RuleStats.min_precision:.0%} precision on this sample.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rule_audit")
    ap.add_argument("--data", default="/tmp/benchrec")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on the synthetic generator instead of BenchRec, "
                    "as a no-false-positive sanity check")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-precision", type=float, default=0.90)
    a = ap.parse_args(argv)

    from pathlib import Path
    from recon.match import MatcherConfig, match_settlements_bank
    from recon.ai import SemanticResolver

    if a.synthetic:
        import json
        import tempfile
        from recon import generate, normalize
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
        print(f"synthetic sanity check, {len(matches)} asserted matches\n")
    else:
        from recon.benchrec_eval import load_solution, load_eval
        solution = load_solution(Path(a.data) / "BenchRec_cash_v1.0_solution.csv")
        ledger, bank, allocation_of = load_eval(Path(a.data) / "BenchRec_cash_v1.0_eval.csv")
        cfg = MatcherConfig(plausible_withholding_bp=(10, 100, 200, 500))  # WITH the
        # rate rule enabled, deliberately — this is the retroactive test of
        # whether the auditor would have caught it without a human reading a
        # table by hand.
        raw, _ = match_settlements_bank(ledger, bank, cfg, SemanticResolver(use_llm=False))
        matches = [m for m in raw if m.confidence >= cfg.confidence_threshold]

        def truth_fn(left_id, right_id):
            truth = solution.get(right_id[1:])
            pred = allocation_of.get(left_id)
            if truth is None or pred is None:
                return None
            return pred == truth
        print(f"BenchRec, withholding rule ENABLED (retroactive test), "
              f"{len(matches)} asserted matches\n")

    stats = audit(matches, truth_fn, min_n=a.min_n, min_precision=a.min_precision)
    print(report(stats))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
