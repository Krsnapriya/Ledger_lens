"""Append-only decision log with a SHA-256 hash chain.

Every engine decision and every human action becomes an immutable event. The
log is the source of truth: `replay.py` reconstructs final state from it and
the reconstruction must be bit-identical, or the test suite fails.

Two design decisions worth defending:

**Logical clock, not wall clock.** Determinism is load-bearing once replay is a
hard gate, and a wall-clock timestamp makes bit-identical replay impossible by
construction. Each event carries `ts`, derived deterministically from a base
epoch recorded in the first event plus the sequence number. Real wall-clock time
is kept in `wall` — outside the hash and outside the replay comparison — because
it is operationally useful and cryptographically irrelevant.

**Canonical serialisation.** The hash is computed over JSON with sorted keys and
no whitespace, so it cannot drift with dict insertion order or formatting.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RULE_VERSION = "L1-L3/2026.08.21"
GENESIS = "0" * 64
BASE_EPOCH = 1_767_225_600  # 2026-01-01T00:00:00Z, fixed so `ts` is reproducible

# Input events replay re-applies; derived events replay re-computes and verifies.
INPUT_KINDS = {"BATCH_INGESTED", "MUTATION_APPLIED", "HUMAN_ASSERTED"}
DERIVED_KINDS = {"WINDOW_RESOLVED", "MATCH_ASSERTED", "MATCH_RETIRED",
                 "EXCEPTION_RAISED", "EXCEPTION_CLEARED", "FULL_RESOLVE_FALLBACK",
                 "DUPLICATE_PINNED", "DUPLICATE_UNPINNED", "OVERRIDE_LEAK"}


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def thresholds_hash(cfg: Any) -> str:
    """Fingerprint of every knob that could change a decision.

    Recorded on every event so a log can never be replayed under different
    thresholds without the mismatch being visible.
    """
    d = {k: v for k, v in vars(cfg).items()}
    return hashlib.sha256(canonical(d).encode()).hexdigest()[:16]


@dataclass
class Event:
    seq: int
    kind: str
    ts: str                        # deterministic logical time
    rule_version: str
    thresholds_hash: str
    payload: dict[str, Any]
    cash_delta_paise: int
    prev_hash: str
    hash: str = ""
    wall: str = ""                 # excluded from the chain on purpose

    def body(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "ts": self.ts,
                "rule_version": self.rule_version,
                "thresholds_hash": self.thresholds_hash,
                "payload": self.payload,
                "cash_delta_paise": self.cash_delta_paise,
                "prev_hash": self.prev_hash}

    def compute_hash(self) -> str:
        return hashlib.sha256(canonical(self.body()).encode()).hexdigest()


class DecisionLog:
    """Append-only. There is no update and no delete, by design."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w")

    @property
    def head(self) -> str:
        return self.events[-1].hash if self.events else GENESIS

    def append(self, kind: str, payload: dict[str, Any], cfg: Any,
               cash_delta_paise: int = 0) -> Event:
        if not isinstance(cash_delta_paise, int):
            raise TypeError("cash deltas are integer paise; no floats in the ledger")
        seq = len(self.events)
        ev = Event(seq=seq, kind=kind,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime(BASE_EPOCH + seq)),
                   rule_version=RULE_VERSION,
                   thresholds_hash=thresholds_hash(cfg),
                   payload=payload, cash_delta_paise=cash_delta_paise,
                   prev_hash=self.head,
                   wall=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        ev.hash = ev.compute_hash()
        self.events.append(ev)
        if self._fh:
            self._fh.write(canonical(asdict(ev)) + "\n")
            self._fh.flush()
        return ev

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ verify
    def verify_chain(self) -> tuple[bool, str]:
        """Recompute every hash and every link. Any tamper breaks it here."""
        prev = GENESIS
        for ev in self.events:
            if ev.prev_hash != prev:
                return False, f"broken link at seq {ev.seq}"
            if ev.compute_hash() != ev.hash:
                return False, f"hash mismatch at seq {ev.seq}"
            prev = ev.hash
        return True, "ok"

    def inputs(self) -> list[Event]:
        return [e for e in self.events if e.kind in INPUT_KINDS]

    def derived(self) -> list[Event]:
        return [e for e in self.events if e.kind in DERIVED_KINDS]

    def cash_position(self) -> int:
        return sum(e.cash_delta_paise for e in self.events)

    @staticmethod
    def load(path: Path) -> "DecisionLog":
        log = DecisionLog(None)
        for line in path.read_text().splitlines():
            if line.strip():
                log.events.append(Event(**json.loads(line)))
        return log
