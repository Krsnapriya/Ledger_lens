"""The AI layer — and, more importantly, its boundary.

Where AI is used:
    Resolving a garbled bank narration to a settlement when no identifier
    survived. "NEFT CR-GATEWYPAY-SETL_0O12-STTLEMENT" against a list of
    candidate settlement ids is genuine fuzzy judgement: there is no exact key,
    the corruption is unstructured, and a human does this by eye.

Where AI is deliberately NOT used:
    - Any arithmetic. Fees, GST, batch totals, balances are integer rules.
    - Any exact identifier join. A regex is faster, free and cannot hallucinate.
    - The final assignment decision. The model proposes a similarity score; a
      constrained optimiser and a confidence threshold decide.

The model never emits a number that enters the ledger. It emits a *candidate
ranking*, which is then subject to the same amount and date constraints as
every other candidate. A hallucinated settlement id simply fails the amount
check and is discarded.

Two implementations. The local one is the default and requires no network, no
API key and no model download, so the headline numbers in this repo are
reproducible by anyone. The LLM one is optional enrichment and is measured
separately so its contribution is visible rather than assumed.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

# Deterministic extraction. Runs first, on every narration.
RE_SETTLEMENT = re.compile(r"\b(setl[_-]?\d{3,5})\b", re.IGNORECASE)
RE_UTR = re.compile(r"\b(UTR\d{10,14})\b", re.IGNORECASE)


def _grams(s: str, n: int = 3) -> Counter:
    s = re.sub(r"[^A-Z0-9]", "", s.upper())
    if len(s) < n:
        return Counter([s]) if s else Counter()
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def cosine(a: str, b: str) -> float:
    """Character-trigram cosine. Robust to truncation and single-char corruption,
    which is exactly the damage profile of a truncated bank narration."""
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    common = set(ga) & set(gb)
    if not common:
        return 0.0
    dot = sum(ga[g] * gb[g] for g in common)
    na = math.sqrt(sum(v * v for v in ga.values()))
    nb = math.sqrt(sum(v * v for v in gb.values()))
    return dot / (na * nb)


@dataclass
class Extraction:
    settlement_id: str | None = None
    utr: str | None = None
    source: str = "regex"


def extract_deterministic(narration: str) -> Extraction:
    sid = RE_SETTLEMENT.search(narration)
    utr = RE_UTR.search(narration)
    return Extraction(
        settlement_id=sid.group(1).lower().replace("-", "_") if sid else None,
        utr=utr.group(1).upper() if utr else None,
    )


@dataclass
class AIStats:
    """Everything needed to answer 'what did the AI actually buy you?'"""

    enabled: bool = False
    backend: str = "local-chargram"
    residual_seen: int = 0        # narrations the deterministic layer could not key
    proposals: int = 0            # candidate rankings produced
    accepted: int = 0             # survived amount/date constraints + threshold
    rejected_by_constraint: int = 0   # model proposed, arithmetic vetoed it
    llm_calls: int = 0
    llm_errors: int = 0
    tiebreaks_invoked: int = 0        # amount-ambiguous groups narration resolved
    tiebreak_margins: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SemanticResolver:
    """Ranks candidate settlements for a narration that carries no usable key."""

    def __init__(self, use_llm: bool = False, model: str = "claude-sonnet-4-6") -> None:
        self.stats = AIStats()
        self.model = model
        key = os.environ.get("ANTHROPIC_API_KEY")
        if use_llm and not key:
            self.stats.notes.append(
                "LLM backend requested but ANTHROPIC_API_KEY is unset — "
                "fell back to local char-trigram. Reported as SKIPPED, not as success."
            )
            use_llm = False
        self.use_llm = use_llm
        self._key = key
        self.stats.enabled = True
        self.stats.backend = "anthropic-llm+local" if use_llm else "local-chargram"

    def similarity(self, narration: str, expected: str) -> float:
        """Score one narration against one synthesised reference narration."""
        self.stats.proposals += 1
        return cosine(narration, expected)

    def note_tiebreak(self, margin: float) -> None:
        """Record that narration actually decided between competing candidates.
        Counting this is the only way to answer 'what did the AI buy you?' with
        a number instead of an assertion."""
        self.stats.tiebreaks_invoked += 1
        self.stats.residual_seen += 1
        self.stats.accepted += 1
        self.stats.tiebreak_margins.append(round(float(margin), 4))

    def rank(self, narration: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """candidates: (settlement_id, synthetic_expected_narration).
        Returns (settlement_id, score in 0..1) sorted desc. Score is a
        *similarity*, explicitly not a probability of correctness."""
        self.stats.residual_seen += 1
        local = sorted(
            ((sid, cosine(narration, text)) for sid, text in candidates),
            key=lambda t: -t[1],
        )
        self.stats.proposals += len(local)
        if not self.use_llm or not local:
            return local
        boosted = self._llm_rerank(narration, local[:8])
        return boosted if boosted else local

    def _llm_rerank(self, narration: str, top: list[tuple[str, float]]):
        """Ask the model to re-rank the local shortlist. Strictly a re-rank of an
        already-constrained candidate set: the model cannot invent an id that
        was not offered to it, which bounds the blast radius of a hallucination
        to 'picked the wrong one of N', never 'picked something that does not exist'."""
        prompt = (
            "You are matching a corrupted bank statement narration to a settlement id.\n"
            f"Narration: {narration!r}\n"
            "Candidate settlement ids:\n"
            + "\n".join(f"- {sid}" for sid, _ in top)
            + "\n\nReturn ONLY a JSON array of objects [{\"id\":..., \"score\":0..1}] "
              "ranking the candidates. No prose. If none is plausible, return []."
        )
        body = json.dumps({
            "model": self.model, "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json",
                     "x-api-key": self._key or "",
                     "anthropic-version": "2023-06-01"})
        try:
            self.stats.llm_calls += 1
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            text = "".join(b.get("text", "") for b in data.get("content", []))
            parsed = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
            allowed = {sid for sid, _ in top}
            # Hard filter: anything not in the offered set is dropped, silently
            # and by construction.
            return [(o["id"], float(o["score"])) for o in parsed if o.get("id") in allowed]
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
            self.stats.llm_errors += 1
            self.stats.notes.append(f"llm call failed, used local score: {type(exc).__name__}")
            return None
