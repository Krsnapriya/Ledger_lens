"""The live layer: windowed re-solve, override isolation, decision logging.

What this is
------------
A controller that keeps a reconciled state and updates it incrementally as data
mutates and humans intervene, re-running the *existing, unmodified* L1-L3 engine
over the minimal affected date window instead of the whole batch.

What this is NOT
----------------
Not fine-grained incremental maintenance of a global constraint graph. The unit
of recomputation is a closed date window, not an individual match. And it does
not claim that arbitrary human overrides can never create downstream
inconsistency — it claims something narrower and testable: overrides are
*isolated*, meaning an asserted record is removed from every candidate pool and
can never become evidence for an engine decision.

Window closure
--------------
A mutation seeds a window from the touched record's date, widened by the
engine's own look-back and look-ahead. Any previously matched group that
intersects that window is pulled in whole, which can widen it again, so the
computation iterates to a fixpoint. Groups are bounded in span, so it converges
quickly — but not always, so there is a hard iteration cap and an honest
fallback to full re-solve, recorded as `FULL_RESOLVE_FALLBACK`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .ai import extract_deterministic
from .events import DecisionLog
from .match import MatcherConfig, reconcile
from .models import Exception_, Match, Reason

MAX_CLOSURE_ITERATIONS = 10

# Structured retirement reason codes. A retirement carrying only a free-text
# cause is a silent invalidation as far as an auditor is concerned.
RETIRE_RECORD_REMOVED = "RECORD_REMOVED"
RETIRE_AMOUNT_CHANGED = "AMOUNT_CHANGED"
RETIRE_MEMBERSHIP_CHANGED = "MEMBERSHIP_CHANGED"
RETIRE_HUMAN_ASSERTED = "HUMAN_ASSERTED"
RETIRE_WINDOW_INVALIDATED = "WINDOW_INVALIDATED"

# Sticky duplicate determinations. A duplicate is only recognisable while the
# credit it copies is in front of the engine. Once a later mutation deletes the
# original, a single-pass engine has no evidence the survivor was ever a
# phantom — it looks like a perfectly good unmatched credit and gets matched.
#
# This is not closable by a static rule, and pushing it into the matcher would
# have destroyed the property that makes engine FMR measurable at all. It
# belongs to the live layer, which is the only component with memory across
# windows. The determination is PINNED when made and survives re-solves.
DUP_PINNED = "DUPLICATE_PINNED"
DUP_UNPINNED = "DUPLICATE_UNPINNED"


def _entity_dates(data: dict[str, Any]) -> dict[str, date]:
    """One flat id -> date map so window logic never branches on record type."""
    d: dict[str, date] = {}
    for o in data["orders"]:
        d[o.order_id] = o.created_on
    for p in data["payments"]:
        d[p.payment_id] = p.captured_on
    for s in data["settlements"]:
        d[s.settlement_id] = s.settled_on
    for b in data["bank"]:
        d[b.txn_id] = b.value_date
    return d


class _Claims:
    """Exact per-match cash attribution.

    A bank line can be claimed by SEVERAL matches at once — a merged payout
    produces one Match per settlement, all pointing at the same credit — so
    naively attributing the line's value to each match would multiply-count it.
    The position counts a line once while at least one match claims it, so the
    cash delta of retiring or asserting a match is the value of the lines whose
    claim count crosses zero. That makes the per-event deltas sum exactly to the
    position, which is the property the log/state assertion now enforces.
    """

    def __init__(self, matches, bank_by_id: dict[str, Any]) -> None:
        self.count: Counter = Counter()
        self.bank = bank_by_id
        for m in matches:
            if m.layer == "L3_SETTLEMENT_BANK":
                for t in m.right_ids:
                    self.count[t] += 1

    def _value(self, txn: str) -> int:
        b = self.bank.get(txn)
        return b.signed_paise if b is not None else 0

    def retire(self, m: Match) -> int:
        if m.layer != "L3_SETTLEMENT_BANK":
            return 0
        delta = 0
        for t in m.right_ids:
            self.count[t] -= 1
            if self.count[t] == 0:
                delta -= self._value(t)
        return delta

    def assert_(self, m: Match) -> int:
        if m.layer != "L3_SETTLEMENT_BANK":
            return 0
        delta = 0
        for t in m.right_ids:
            self.count[t] += 1
            if self.count[t] == 1:
                delta += self._value(t)
        return delta


@dataclass
class LiveState:
    data: dict[str, Any]
    matches: list[Match] = field(default_factory=list)
    exceptions: list[Exception_] = field(default_factory=list)
    asserted: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> override
    # txn_id -> {duplicate_of, amount_paise, reason, seq}
    duplicates: dict[str, dict[str, Any]] = field(default_factory=dict)
    # utr -> entity_id whose override caused the exclusion. A human writing off a
    # credit must not have that money resurrected by a later re-posting of the
    # same bank reference, so the instrument is excluded, not merely the row.
    excluded_utrs: dict[str, str] = field(default_factory=dict)

    def reconciled_paise(self) -> int:
        """Cash position: signed value of every bank line the engine has
        attributed to a payout. Integer paise throughout."""
        matched = {t for m in self.matches if m.layer == "L3_SETTLEMENT_BANK"
                   for t in m.right_ids}
        return sum(b.signed_paise for b in self.data["bank"] if b.txn_id in matched)


class OverrideLeak(RuntimeError):
    """A human-asserted record reached the matching engine.

    This is not recoverable and is never downgraded to a warning. If an override
    can be seen by any rule it can satisfy arithmetic, a subset sum, or an
    assignment — which is precisely the poison path the live layer exists to
    prevent, and it would silently invalidate the engine-FMR measurement that
    every other claim rests on.
    """


class LiveController:
    def __init__(self, data: dict[str, Any], cfg: MatcherConfig | None = None,
                 log_path: Path | None = None) -> None:
        self.cfg = cfg or MatcherConfig()
        self.log = DecisionLog(log_path)
        self.state = LiveState(data)
        self.stats: dict[str, Any] = {"window_resolves": 0, "full_fallbacks": 0,
                                      "window_days": [], "window_ms": [],
                                      "records_in_window": []}
        self._bootstrap()

    # ------------------------------------------------------------- bootstrap
    def _bootstrap(self) -> None:
        r = reconcile(self.state.data, self.cfg)
        self.state.matches = list(r.matches)
        self.state.exceptions = list(r.exceptions)
        self._pin_new_duplicates(r.exceptions)
        pos = self.state.reconciled_paise()
        self.log.append("BATCH_INGESTED", {
            "records": sum(len(self.state.data[k]) for k in
                           ("orders", "payments", "refunds", "settlements", "bank")),
            "matches": len(r.matches), "exceptions": len(r.exceptions),
        }, self.cfg, cash_delta_paise=pos)

    # ---------------------------------------------------------- window logic
    def _match_span(self, m: Match, dates: dict[str, date]) -> list[date]:
        ds = [dates[i] for i in ([m.left_id] + list(m.right_ids)) if i in dates]
        return ds

    def _amount_peers(self, seed_ids: list[str]) -> set[str]:
        """Bank lines sharing an amount with a seeded bank line, at any date.

        Duplicate detection is a GLOBAL property: the engine can only recognise a
        credit as a duplicate if the credit it duplicates is in front of it. A
        window chosen purely by date can exclude the original, and then the
        duplicate looks like a perfectly good unmatched credit — which is exactly
        how the mutation harness drove engine FMR to 13%: a near-duplicate landing
        outside its original's window got matched by identifier, by amount+date,
        and by subset-sum.

        So the window is widened along a second axis. Any bank line whose amount
        ties out with a seeded bank line is pulled in regardless of date. This is
        cheap (one pass, exact integer equality) and it restores the invariant
        that a duplicate is always evaluated alongside its original.
        """
        seeded = {b.txn_id: b for b in self.state.data["bank"] if b.txn_id in seed_ids}
        # Callers pass the whole L1 neighbourhood, not just the seeds. A
        # duplicate injected in one window can be re-examined in a later,
        # unrelated window that excludes its original — and then it stands
        # unopposed and matches by UTR. Peering over everything the window
        # already contains closes that hole.
        amounts = {b.signed_paise for b in seeded.values()}
        if not amounts:
            return set()
        return {b.txn_id for b in self.state.data["bank"]
                if b.signed_paise in amounts}

    def closed_record_set(self, seed_ids: list[str],
                          seed_dates: list[date] | None = None
                          ) -> tuple[set[str], bool, dict[str, Any]]:
        """The minimal closed SET OF RECORDS to re-solve. Single source of truth.

        The previous implementation returned a date interval, and the slice was
        then "everything between lo and hi". That is both over-inclusive in the
        middle and structurally fragile at the edges: a prior match whose bank leg
        sits inside the interval and whose settlement leg sits outside is only
        pulled in because extending the interval happens to sweep it up. Closure
        was a side effect of arithmetic on dates rather than a property of the
        graph, so it could not be stated, tested, or bounded.

        This computes the set directly:

          L0  the touched records themselves
          L1  every record inside the candidate neighbourhood of a touched
              record — this is the only place NEW matches can come from, so the
              neighbourhood uses the engine's own look-back/look-ahead
          L2+ edge closure: any prior match with a member in the set contributes
              ALL of its members, recursively, to a hard hop limit
          +   amount peers, because duplicate detection is a global property and
              a window that excludes the original cannot see the duplicate

        Closure means: no prior match is ever half-inside. Either every member of
        a match is in the set (so it can be retired and re-derived intact) or none
        of it is (so it is untouched). That is the property the caller relies on
        when it retires matches, and it is now asserted rather than assumed.
        """
        dates = _entity_dates(self.state.data)
        seeds = {i for i in seed_ids if i in dates}
        stats: dict[str, Any] = {"seeds": len(seeds), "hops_used": 0,
                                 "l1_neighbourhood": 0, "amount_peers": 0}
        # A mutation may DELETE its own seed — `bank_credit_disappears` is the
        # obvious case. The record then has no date to window around and the
        # first version fell back to a full re-solve, which is safe but not
        # minimal: 10% of all windows were whole-batch for this reason alone.
        # The caller snapshots the dates before mutating, so the window can still
        # be built around where the record used to be.
        anchor_dates = [dates[i] for i in seeds] + list(seed_dates or [])
        if not anchor_dates:
            return set(dates), False, stats

        back = max(self.cfg.subset_sum_lookback_days, self.cfg.date_window_days)
        fwd = max(self.cfg.subset_sum_lookahead_days, self.cfg.date_window_days)
        reach = max(back, fwd)   # conservative: a record may be a candidate in
                                 # either direction depending on its type

        # ---- L1: candidate neighbourhood of the seeds ----
        selected = set(seeds)
        for rid, d in dates.items():
            if any(abs((d - sd).days) <= reach for sd in anchor_dates):
                selected.add(rid)
        stats["l1_neighbourhood"] = len(selected) - len(seeds)

        # ---- amount peers (global duplicate property) ----
        peers = self._amount_peers(sorted(selected))
        stats["amount_peers"] = len(peers - selected)
        selected |= peers

        # ---- L2+: edge closure over prior matches ----
        # Closure is computed over records that still EXIST. A match may reference
        # a record a mutation deleted; the closure cannot pull in what is gone, and
        # demanding it would make the window permanently unprovable. Such a match
        # is stale by definition and must be retired, so its surviving members are
        # pulled in and it is flagged for retirement as RECORD_REMOVED.
        present = set(dates)
        stale = {m.key() for m in self.state.matches
                 if not ({m.left_id} | set(m.right_ids)) <= present}
        stats["stale_matches"] = len(stale)
        converged = False
        for hop in range(1, self.cfg.window_hop_limit + 1):
            added: set[str] = set()
            for m in self.state.matches:
                members = ({m.left_id} | set(m.right_ids)) & present
                if members & selected or m.key() in stale:
                    added |= members - selected
            if not added:
                converged = True
                stats["hops_used"] = hop - 1
                break
            selected |= added
            stats["hops_used"] = hop
        else:
            # Hop limit reached with the set still growing: the window cannot be
            # proven closed, so the caller must fall back rather than reconcile a
            # window it cannot vouch for.
            converged = False

        stats["records"] = len(selected)
        if selected:
            ds = [dates[i] for i in selected if i in dates]
            stats["span_days"] = (max(ds) - min(ds)).days if ds else 0
        return selected, converged, stats

    def assert_window_is_closed(self, selected: set[str]) -> None:
        """No prior match may be half-inside the window.

        If one is, retiring it would alter records the re-solve never sees, which
        is precisely the silent inconsistency that cash-drift and FMR checks can
        miss when the mismatch is small.
        """
        present = set(_entity_dates(self.state.data))
        for m in self.state.matches:
            members = ({m.left_id} | set(m.right_ids)) & present
            inside = members & selected
            if inside and not members <= selected:
                raise AssertionError(
                    f"window not closed: match {m.left_id}->{sorted(m.right_ids)} "
                    f"({m.rule}) has {sorted(inside)} inside and "
                    f"{sorted(members - selected)} outside")

    def _slice(self, selected: set[str]) -> tuple[dict[str, Any], set[str]]:
        """Build the engine's input from the closed record set.

        Human-asserted records and pinned duplicates are removed outright — that
        is the whole of override isolation and duplicate stickiness.
        """
        d = self.state.data
        blocked = set(self.state.asserted) | set(self.state.duplicates)

        settlements = [s for s in d["settlements"]
                       if s.settlement_id in selected
                       and not self._is_excluded(s.settlement_id, s)]
        sids = {s.settlement_id for s in settlements}
        bank = [b for b in d["bank"]
                if b.txn_id in selected and not self._is_excluded(b.txn_id, b)]
        # A settlement's batch arithmetic needs ALL its member payments, even ones
        # captured outside the window; otherwise the footing check fires
        # spuriously on a partial batch. This is completeness of an aggregate, not
        # window closure, so it is applied after the set is fixed.
        payments = [p for p in d["payments"]
                    if (p.payment_id in selected or p.settlement_id in sids)
                    and p.payment_id not in blocked]
        pids = {p.payment_id for p in payments}
        prior_order = {m.left_id: m.right_ids[0] for m in self.state.matches
                       if m.layer == "L1_ORDER_PAYMENT"}
        orders = [o for o in d["orders"]
                  if (o.order_id in selected or prior_order.get(o.order_id) in pids)
                  and o.order_id not in blocked]
        refunds = [r for r in d["refunds"] if r.settlement_id in sids]

        ids = ({o.order_id for o in orders} | pids | sids | {b.txn_id for b in bank})
        return ({"orders": orders, "payments": payments, "refunds": refunds,
                 "settlements": settlements, "bank": bank}, ids)

    def _excluded_utr_of(self, record: Any) -> str | None:
        """The bank reference an overridden record carries, if any."""
        if hasattr(record, "narration"):
            return extract_deterministic(record.narration).utr
        return getattr(record, "utr", None)

    def _is_excluded(self, entity_id: str, record: Any = None) -> bool:
        if entity_id in self.state.asserted or entity_id in self.state.duplicates:
            return True
        if record is not None:
            utr = self._excluded_utr_of(record)
            if utr and utr in self.state.excluded_utrs:
                return True
        return False

    def excluded_record_ids(self) -> set[str]:
        """Every record the LIVE LAYER holds out of the engine's reach.

        Single source of truth for overrides, pinned duplicates and UTR-level
        exclusions. Any control re-solve must honour exactly this set, or it is
        solving a different problem and the zero-drift invariant is meaningless.
        """
        out = set(self.state.asserted) | set(self.state.duplicates)
        if self.state.excluded_utrs:
            for b in self.state.data["bank"]:
                u = self._excluded_utr_of(b)
                if u and u in self.state.excluded_utrs:
                    out.add(b.txn_id)
            for st in self.state.data["settlements"]:
                if st.utr and st.utr in self.state.excluded_utrs:
                    out.add(st.settlement_id)
        return out

    def assert_no_override_leak(self, sliced: dict[str, Any] | None = None,
                                matches: list[Match] | None = None) -> None:
        """Checked on the engine's INPUT and on its OUTPUT.

        Isolation is enforced by construction at the slice, which means a future
        refactor could break it without any test noticing. So it is also
        detected: nothing excluded may appear in what the engine is handed, and
        nothing excluded may appear in what the engine returns.
        """
        blocked = set(self.state.asserted)
        if sliced is not None:
            leaked = []
            for key, id_attr in (("orders", "order_id"), ("payments", "payment_id"),
                                 ("settlements", "settlement_id"), ("bank", "txn_id")):
                for rec in sliced.get(key, []):
                    rid = getattr(rec, id_attr)
                    if rid in blocked:
                        leaked.append((rid, "id"))
                    else:
                        utr = self._excluded_utr_of(rec)
                        if utr and utr in self.state.excluded_utrs:
                            leaked.append((rid, f"utr:{utr}"))
            if leaked:
                self.log.append("OVERRIDE_LEAK",
                                {"stage": "engine_input", "leaked": sorted(map(str, leaked))[:10]},
                                self.cfg)
                raise OverrideLeak(
                    f"human-asserted record(s) reached the engine input: {sorted(map(str, leaked))[:5]}")
        if matches is not None:
            bad = [m for m in matches
                   if m.left_id in blocked or (set(m.right_ids) & blocked)]
            if bad:
                self.log.append("OVERRIDE_LEAK",
                                {"stage": "engine_output",
                                 "leaked": [f"{m.left_id}->{sorted(m.right_ids)}" for m in bad[:5]]},
                                self.cfg)
                raise OverrideLeak(
                    f"engine asserted {len(bad)} match(es) using human-asserted records")

    def _retire_reason(self, m: Match, seeds: set[str], cause: str) -> str:
        """Why this specific match is being retired. Deterministic, no free text."""
        present = set(_entity_dates(self.state.data))
        touched = {m.left_id} | set(m.right_ids)
        if touched & set(self.state.asserted):
            return RETIRE_HUMAN_ASSERTED
        if not touched <= present:
            return RETIRE_RECORD_REMOVED
        if touched & seeds:
            if cause.startswith("override:"):
                return RETIRE_HUMAN_ASSERTED
            if "amount" in cause or "reserve" in cause or "restated" in cause:
                return RETIRE_AMOUNT_CHANGED
            return RETIRE_MEMBERSHIP_CHANGED
        return RETIRE_WINDOW_INVALIDATED

    def _refresh_duplicate_pins(self) -> None:
        """Release a pin whose record has materially changed or disappeared.

        A pin must not outlive the evidence for it. If the bank restates the
        amount of a line previously judged a duplicate, that judgement was made
        about a different record and has to be re-earned; if the line is gone,
        the pin is moot. Without this, a stale pin could hide real money, which
        is a worse failure than the one the pin exists to prevent.
        """
        bank = {b.txn_id: b for b in self.state.data["bank"]}
        for txn in sorted(self.state.duplicates):
            pin = self.state.duplicates[txn]
            b = bank.get(txn)
            reason = None
            if b is None:
                reason = "record removed"
            elif b.signed_paise != pin["amount_paise"]:
                reason = "amount restated since determination"
            if reason:
                del self.state.duplicates[txn]
                self.log.append(DUP_UNPINNED,
                                {"txn_id": txn, "released_because": reason,
                                 "was_duplicate_of": pin["duplicate_of"]}, self.cfg)

    def _pin_new_duplicates(self, exceptions) -> None:
        """Make every fresh duplicate determination survive future windows."""
        bank = {b.txn_id: b for b in self.state.data["bank"]}
        for e in sorted(exceptions, key=lambda x: x.entity_id):
            if e.reason is not Reason.DUPLICATE_BANK_CREDIT:
                continue
            if e.entity_id in self.state.duplicates:
                continue
            b = bank.get(e.entity_id)
            if b is None:
                continue
            self.state.duplicates[e.entity_id] = {
                "duplicate_of": e.blocking_data.get("duplicate_of"),
                "amount_paise": b.signed_paise,
                "detail": e.detail,
                "seq": len(self.log.events)}
            self.log.append(DUP_PINNED,
                            {"txn_id": e.entity_id,
                             "duplicate_of": e.blocking_data.get("duplicate_of"),
                             "amount_paise": b.signed_paise,
                             "detail": e.detail}, self.cfg)

    def _sticky_duplicate_exceptions(self) -> list[Exception_]:
        """Re-emit pinned duplicates so they stay visible on the exception ledger
        even though the engine no longer sees the records."""
        bank = {b.txn_id: b for b in self.state.data["bank"]}
        out = []
        for txn in sorted(self.state.duplicates):
            pin = self.state.duplicates[txn]
            b = bank.get(txn)
            if b is None:
                continue
            out.append(Exception_("bank_line", txn, Reason.DUPLICATE_BANK_CREDIT,
                                  pin["detail"] + " (pinned across re-solves)",
                                  {"amount_paise": b.signed_paise,
                                   "duplicate_of": pin["duplicate_of"],
                                   "pinned_at_event": pin["seq"],
                                   "narration": b.narration},
                                  confidence=0.96))
        return out

    # ------------------------------------------------------------- re-solve
    def resolve_window(self, seed_ids: list[str], cause: str,
                       seed_dates: list[date] | None = None) -> dict[str, Any]:
        import time as _t
        t0 = _t.perf_counter()
        self._refresh_duplicate_pins()
        selected, converged, wstats = self.closed_record_set(seed_ids, seed_dates)
        if not converged:
            self.stats["full_fallbacks"] += 1
            self.log.append("FULL_RESOLVE_FALLBACK",
                            {"cause": cause, "seed_ids": sorted(seed_ids),
                             "reason": "record-set closure did not converge inside "
                                       f"{self.cfg.window_hop_limit} hops",
                             "records_at_abort": wstats.get("records")},
                            self.cfg)
            selected = set(_entity_dates(self.state.data))
        # Closure is a property the caller depends on, so it is checked, not assumed.
        self.assert_window_is_closed(selected)

        before = self.state.reconciled_paise()
        sliced, ids = self._slice(selected)
        bank_by_id = {b.txn_id: b for b in self.state.data["bank"]}
        claims = _Claims(self.state.matches, bank_by_id)
        seeds = set(seed_ids)

        # Retire every match touching the window, with its cash delta, before
        # anything is re-derived. A retirement is an event, not a deletion.
        present_ids = set(_entity_dates(self.state.data))
        excluded_now = self.excluded_record_ids()
        retired, kept = [], []
        for m in self.state.matches:
            members = {m.left_id} | set(m.right_ids)
            stale = not members <= present_ids
            # A match predating an exclusion must go even if it sits outside the
            # window. Exclusion is a global determination; leaving an older match
            # alive is exactly how an overridden record stays reconciled while the
            # control re-solve treats it as gone.
            touches_excluded = bool(members & excluded_now)
            if (stale or touches_excluded or m.left_id in ids
                    or any(r in ids for r in m.right_ids)):
                retired.append(m)
            else:
                kept.append(m)
        retired_total = 0
        for m in sorted(retired, key=lambda x: x.key()):
            delta = claims.retire(m)
            retired_total += delta
            self.log.append("MATCH_RETIRED",
                            {"layer": m.layer, "left_id": m.left_id,
                             "right_ids": sorted(m.right_ids), "rule": m.rule,
                             "reason_code": self._retire_reason(m, seeds, cause),
                             "cause": cause}, self.cfg,
                            cash_delta_paise=delta)
        self.state.matches = kept
        self.state.exceptions = [e for e in self.state.exceptions
                                 if e.entity_id not in ids]

        self.assert_no_override_leak(sliced=sliced)
        r = reconcile(sliced, self.cfg)
        self.assert_no_override_leak(matches=r.matches)
        asserted_total = 0
        for m in sorted(r.matches, key=lambda x: x.key()):
            delta = claims.assert_(m)
            asserted_total += delta
            self.log.append("MATCH_ASSERTED",
                            {"layer": m.layer, "left_id": m.left_id,
                             "right_ids": sorted(m.right_ids), "rule": m.rule,
                             "confidence": round(m.confidence, 6),
                             "evidence": m.evidence}, self.cfg,
                            cash_delta_paise=delta)
        for e in sorted(r.exceptions, key=lambda x: (x.entity_type, x.entity_id, x.reason.value)):
            self.log.append("EXCEPTION_RAISED",
                            {"entity_type": e.entity_type, "entity_id": e.entity_id,
                             "reason": e.reason.value, "detail": e.detail},
                            self.cfg, cash_delta_paise=0)
        self._pin_new_duplicates(r.exceptions)
        self.state.matches.extend(r.matches)
        self.state.exceptions.extend(r.exceptions)
        # Records the engine can no longer see must still appear on the ledger.
        seen = {(e.entity_id, e.reason) for e in self.state.exceptions}
        self.state.exceptions.extend(
            e for e in self._sticky_duplicate_exceptions()
            if (e.entity_id, e.reason) not in seen)
        self.state.exceptions.sort(key=lambda e: (-e.severity, e.entity_id))

        # Final gate: the ACCUMULATED match set, not just this window's output.
        # Checking only new assertions missed a match that predated the override.
        self.assert_no_override_leak(matches=self.state.matches)

        after = self.state.reconciled_paise()
        ms = (_t.perf_counter() - t0) * 1000
        span_days = wstats.get("span_days", -1)
        self.stats["window_resolves"] += 1
        self.stats["window_days"].append(span_days)
        self.stats["window_ms"].append(ms)
        self.stats["records_in_window"].append(len(ids))
        # Zero-delta SUMMARY event. The movement is already carried by the
        # individual MATCH_RETIRED and MATCH_ASSERTED deltas; carrying it again
        # here would double-count and break sum(deltas) == position.
        self.log.append("WINDOW_RESOLVED",
                        {"cause": cause, "converged": converged,
                         "records_in_window": len(ids),
                         "closed_set_size": wstats.get("records"),
                         "hops_used": wstats.get("hops_used"),
                         "l1_neighbourhood": wstats.get("l1_neighbourhood"),
                         "amount_peers": wstats.get("amount_peers"),
                         "span_days": span_days,
                         "matches_retired": len(retired),
                         "matches_asserted": len(r.matches),
                         "retired_cash_paise": retired_total,
                         "asserted_cash_paise": asserted_total,
                         "net_cash_paise": after - before},
                        self.cfg, cash_delta_paise=0)
        self.stats.setdefault("hops", []).append(wstats.get("hops_used", 0))
        return {"converged": converged, "records": len(ids), "ms": ms,
                "hops": wstats.get("hops_used"), "cash_delta": after - before}

    # ------------------------------------------------------------- mutations
    def apply_mutation(self, mutation: dict[str, Any]) -> dict[str, Any]:
        """Apply a data change, then re-solve only what it can have affected."""
        from . import mutations
        # The position must be sampled BEFORE the data changes. The original code
        # sampled it inside resolve_window, i.e. AFTER mutations.apply had already
        # rewritten an already-matched line, so the mutation's own effect on the
        # position fell into a gap no event covered and the log under-reported by
        # exactly the mutation delta — down to a single paise.
        pos_before = self.state.reconciled_paise()
        pre_dates = _entity_dates(self.state.data)
        touched = mutations.apply(self.state.data, mutation)
        # Dates of anything the mutation removed, captured before it vanished.
        vanished = [pre_dates[i] for i in sorted(touched)
                    if i in pre_dates and i not in _entity_dates(self.state.data)]
        delta = self.state.reconciled_paise() - pos_before
        self.log.append("MUTATION_APPLIED",
                        {"op": mutation["op"], "args": mutation.get("args", {}),
                         "touched": sorted(touched)}, self.cfg,
                        cash_delta_paise=delta)
        return self.resolve_window(sorted(touched), cause=f"mutation:{mutation['op']}",
                                   seed_dates=vanished)

    # -------------------------------------------------------------- override
    def apply_override(self, entity_id: str, action: str, reason: str,
                       cash_impact_paise: int = 0,
                       counterparty: str | None = None) -> dict[str, Any]:
        """Record a human decision and isolate it.

        Safety rule enforced here, not merely documented: an override is only
        accepted on a record the engine has itself flagged for review or left as
        an exception. Overriding a record the engine confidently matched would
        make "engine-generated false match" unmeasurable, because engine and
        human decisions would be entangled in the same record.
        """
        flagged = {e.entity_id for e in self.state.exceptions}
        if entity_id not in flagged:
            raise ValueError(
                f"unsafe override: {entity_id} is not in REVIEW/EXCEPTION state")
        if not isinstance(cash_impact_paise, int):
            raise TypeError("cash impact is integer paise")

        by_id: dict[str, Any] = {}
        for b in self.state.data["bank"]:
            by_id[b.txn_id] = b
        for st in self.state.data["settlements"]:
            by_id[st.settlement_id] = st
        utrs = []
        for eid in [entity_id] + ([counterparty] if counterparty else []):
            rec = by_id.get(eid)
            utr = self._excluded_utr_of(rec) if rec is not None else None
            if utr and utr not in self.state.excluded_utrs:
                self.state.excluded_utrs[utr] = eid
                utrs.append(utr)
        self.state.asserted[entity_id] = {
            "action": action, "reason": reason,
            "cash_impact_paise": cash_impact_paise, "counterparty": counterparty,
            "excluded_utrs": utrs}
        if counterparty:
            self.state.asserted[counterparty] = {
                "action": action, "reason": reason, "cash_impact_paise": 0,
                "counterparty": entity_id}
        # cash_delta_paise is 0 by design. The window re-solve that follows
        # removes these records from the candidate pool and the resulting
        # position movement is carried by the retirement/assertion events.
        # Adding the impact here as well double-counts it.
        self.log.append("HUMAN_ASSERTED",
                        {"entity_id": entity_id, "action": action, "reason": reason,
                         "counterparty": counterparty,
                         "excluded_utrs": sorted(utrs),
                         "declared_cash_impact_paise": cash_impact_paise},
                        self.cfg, cash_delta_paise=0)
        # A UTR exclusion is a GLOBAL determination, so every record carrying an
        # excluded reference must be re-solved — not only the ones that happen to
        # fall inside the override's own date window. Without this, a credit
        # matched in an earlier window keeps its match while the control re-solve
        # excludes it, and the two disagree by the whole value of that credit.
        seeds = [entity_id] + ([counterparty] if counterparty else [])
        if utrs:
            for b in self.state.data["bank"]:
                if self._excluded_utr_of(b) in self.state.excluded_utrs:
                    seeds.append(b.txn_id)
            for st in self.state.data["settlements"]:
                if st.utr and st.utr in self.state.excluded_utrs:
                    seeds.append(st.settlement_id)
        return self.resolve_window(sorted(set(seeds)), cause=f"override:{action}")

    # ---------------------------------------------------------------- output
    def engine_matches(self) -> list[Match]:
        """Matches the ENGINE asserted. Human assertions are not in here — they
        are events, not matches, which is what keeps the two measurable apart."""
        return self.state.matches

    def assert_log_state_consistent(self) -> None:
        """The log IS the ledger. If summed deltas disagree with the position,
        the log is fiction and every downstream artifact is unverified."""
        lg, st = self.log.cash_position(), self.state.reconciled_paise()
        if lg != st:
            raise AssertionError(f"log/state desync: log={lg} state={st} "
                                 f"drift={lg - st} paise")

    def snapshot(self) -> dict[str, Any]:
        return {
            "reconciled_paise": self.state.reconciled_paise(),
            "matches": sorted(m.key() for m in self.state.matches),
            "exceptions": sorted((e.entity_id, e.reason.value)
                                 for e in self.state.exceptions),
            "asserted": sorted(self.state.asserted),
            "pinned_duplicates": sorted(self.state.duplicates),
            "excluded_utrs": sorted(self.state.excluded_utrs),
            "log_head": self.log.head,
            "log_cash_position": self.log.cash_position(),
        }
