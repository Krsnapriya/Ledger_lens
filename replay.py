"""Replay verifier — the hard gate.

Starts from the original batch, re-applies every INPUT event from the decision
log in order, and requires the reconstructed final state to be **bit-identical**
to the live run: same matches.csv, same exceptions.csv, same cash position, same
derived event sequence, same log head hash.

Deliberately, replay does not copy the recorded outcomes. It re-derives them by
running the same engine over the same inputs. Copying outputs would verify
nothing but the file writer; re-deriving them verifies that the engine is
deterministic, that the window computation is not order-dependent, and that
nothing about the live path smuggled in state the log does not capture.

    python replay.py                    # runs a live session then replays it
    python replay.py --steps 60         # longer mutation sequence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from recon import generate, mutations, normalize, report
from recon.events import DecisionLog
from recon.live import LiveController
from recon.match import MatcherConfig


def artifact_hashes(controller: LiveController, out: Path) -> dict[str, str]:
    """Write the two artifacts a controller is judged on and hash the bytes."""
    out.mkdir(parents=True, exist_ok=True)
    report.write_matches_csv(out / "matches.csv",
                             sorted(controller.state.matches, key=lambda m: m.key()))
    report.write_exceptions_csv(out / "exceptions.csv",
                                sorted(controller.state.exceptions,
                                       key=lambda e: (e.entity_type, e.entity_id,
                                                      e.reason.value)))
    return {f: hashlib.sha256((out / f).read_bytes()).hexdigest()
            for f in ("matches.csv", "exceptions.csv")}


def replay(original_data, log: DecisionLog, cfg: MatcherConfig) -> LiveController:
    """Rebuild from inputs alone."""
    import copy
    data = copy.deepcopy(original_data)
    truth: dict = {"settlement_to_bank": {}, "bank_to_settlements": {},
                   "tiers": {}, "expected_exceptions": []}
    mutations.bind_truth(data, truth)
    ctl = LiveController(data, cfg)
    for ev in log.inputs():
        if ev.kind == "BATCH_INGESTED":
            continue                      # the constructor already did this
        if ev.kind == "MUTATION_APPLIED":
            ctl.apply_mutation({"op": ev.payload["op"], "args": ev.payload["args"]})
        elif ev.kind == "HUMAN_ASSERTED":
            ctl.apply_override(ev.payload["entity_id"], ev.payload["action"],
                               ev.payload["reason"],
                               cash_impact_paise=ev.cash_delta_paise,
                               counterparty=ev.payload.get("counterparty"))
    return ctl


def compare(live: LiveController, rep: LiveController,
            live_h: dict[str, str], rep_h: dict[str, str]) -> list[str]:
    fails = []
    # The gate that was missing. The verifier previously compared state to state,
    # so both sides could be equally wrong relative to the log and still report
    # VERIFIED. A 1-paise mutation desynchronised the immutable log from the
    # ledger it claims to record and passed every check.
    for name, ctl in (("live", live), ("replay", rep)):
        lg, st = ctl.log.cash_position(), ctl.state.reconciled_paise()
        if lg != st:
            fails.append(f"{name} log/state desync: log={lg} state={st} "
                         f"drift={lg - st} paise")
        bad = [e.seq for e in ctl.log.events if e.kind == "MATCH_RETIRED"
               and not e.payload.get("reason_code")]
        if bad:
            fails.append(f"{name}: {len(bad)} MATCH_RETIRED without reason_code "
                         f"(first at seq {bad[0]})")
        floats = [e.seq for e in ctl.log.events
                  if not isinstance(e.cash_delta_paise, int)
                  or isinstance(e.cash_delta_paise, bool)]
        if floats:
            fails.append(f"{name}: non-integer cash delta at seq {floats[0]}")
    if live.log.cash_position() != rep.log.cash_position():
        fails.append(f"log cash position differs: {live.log.cash_position()} "
                     f"vs {rep.log.cash_position()}")
    lv = {(e.rule_version, e.thresholds_hash) for e in live.log.events}
    rv = {(e.rule_version, e.thresholds_hash) for e in rep.log.events}
    if lv != rv:
        fails.append(f"rule_version/thresholds_hash mismatch: {lv} vs {rv}")
    if live_h != rep_h:
        for k in live_h:
            if live_h[k] != rep_h[k]:
                fails.append(f"artifact bytes differ: {k}")
    if live.state.reconciled_paise() != rep.state.reconciled_paise():
        fails.append(f"cash position differs: {live.state.reconciled_paise()} "
                     f"vs {rep.state.reconciled_paise()}")
    ls, rs = live.snapshot(), rep.snapshot()
    for key in ("matches", "exceptions", "asserted"):
        if ls[key] != rs[key]:
            fails.append(f"state differs: {key}")
    ld = [(e.kind, json.dumps(e.payload, sort_keys=True, default=str),
           e.cash_delta_paise) for e in live.log.derived()]
    rd = [(e.kind, json.dumps(e.payload, sort_keys=True, default=str),
           e.cash_delta_paise) for e in rep.log.derived()]
    if ld != rd:
        first = next((i for i, (a, b) in enumerate(zip(ld, rd)) if a != b), min(len(ld), len(rd)))
        fails.append(f"derived event stream diverges at index {first} "
                     f"({len(ld)} live vs {len(rd)} replayed)")
    if live.log.head != rep.log.head:
        fails.append("log head hash differs")
    ok, msg = live.log.verify_chain()
    if not ok:
        fails.append(f"live hash chain broken: {msg}")
    ok, msg = rep.log.verify_chain()
    if not ok:
        fails.append(f"replay hash chain broken: {msg}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--orders", type=int, default=300)
    a = ap.parse_args(argv)

    import copy
    from mutate import build_plan, drive

    d = Path(tempfile.mkdtemp())
    generate.write(generate.generate(a.orders, a.seed, 1.0), d)
    data, _ = normalize.load_all(d)
    truth = json.loads((d / "truth.json").read_text())
    original = copy.deepcopy(data)
    cfg = MatcherConfig()

    mutations.bind_truth(data, truth)
    live = LiveController(data, cfg, log_path=Path("artifacts/decisions.jsonl"))
    plan = build_plan(live, truth, a.steps, a.seed)
    drive(live, plan)
    live.log.close()

    live_h = artifact_hashes(live, Path(tempfile.mkdtemp()))
    rep = replay(original, live.log, cfg)
    rep_h = artifact_hashes(rep, Path(tempfile.mkdtemp()))

    fails = compare(live, rep, live_h, rep_h)
    print(f"events        {len(live.log.events)} "
          f"({len(live.log.inputs())} input / {len(live.log.derived())} derived)")
    print(f"hash chain    {live.log.verify_chain()[1]}  head={live.log.head[:16]}…")
    print(f"cash position {live.state.reconciled_paise()} paise "
          f"(log {live.log.cash_position()})")
    print(f"retirements   {sum(1 for e in live.log.events if e.kind=='MATCH_RETIRED')} "
          f"all with reason_code: "
          f"{all(e.payload.get('reason_code') for e in live.log.events if e.kind=='MATCH_RETIRED')}")
    print(f"matches.csv   sha256 {live_h['matches.csv'][:16]}…")
    print(f"exceptions.csv sha256 {live_h['exceptions.csv'][:16]}…")
    if fails:
        print("\nREPLAY FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print("\nREPLAY VERIFIED — bit-identical artifacts, cash position, and event stream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
