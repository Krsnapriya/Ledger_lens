# LedgerLens

**Multi-source settlement reconciliation with a measured residual instead of a protected zero.**

Razorpay AI Buildathon — Track 04, AI Finance Controller.

```bash
pip install -r requirements.txt     # numpy, scipy. That is all.
python -m recon.cli --orders 300 --seed 7
```

No API key. No network. No model download. No notebook. Python 3.12.3.

```
  SETTLEMENT MATCH RATE   100.00%   (25/25 payouts)
  FALSE MATCH RATE        0.000%   <- the number that decides whether this is usable
  L3 precision / recall   1.0000 / 1.0000
  unreconciled exposure   ₹6,20,633.43
```

`./run.sh` regenerates every table below, offline, in about twelve minutes.

---

## The precise claims

| claim | measured |
|---|---|
| engine false matches, in-distribution (40 seeds × 5 difficulties, 200 runs) | **0.0000%** |
| engine false matches, multi-way consolidation | **0.0000%** |
| engine false matches, **10-class adversarial mutation suite** | **0.0000%** |
| engine false matches, **16-class adversarial suite built to attack this engine's own rules** | **2.50%** (3 of 120), hard 3% ceiling test |
| cash-position drift vs. a full re-solve, non-overridden paths | **0 paise** |
| match-set difference vs. a full re-solve | **0** |
| full event-log replay → `matches.csv`, `exceptions.csv`, cash position, event stream | **bit-identical** |
| human-asserted records appearing in any engine match | **0** |
| deleting the AI module entirely | **changes no number to four decimal places** |

**The most useful number in this repo is the distance between 0.0000% and 2.50%,
because that distance is the measured worth of a zero.** The engine was tuned
against ten anomaly classes and scored zero on them. Six more classes were then
written specifically to attack the rules the first ten had produced, and the rate
went to 6.72% before two fixes brought it to 2.50%. A class the generator still
cannot produce remains invisible by construction, so 2.50% is a floor on the true
rate under broader damage, not a proof.

> **There is exactly one relaxation in the scorer, and this is it.** The
> `duplicate_before_original` class injects a credit that is byte-identical to the
> one it copies — same integer amount, same narration string, differing only in
> value date and row id, both dated after the payout. No field the engine can
> observe separates them, so which row id it cites is arbitrary; the truth file
> designates one only because the generator created it second. Counting that as a
> false match measures row-label agreement, not correctness. The oracle therefore
> exonerates a pair when the cited credit is byte-identical to one the truth does
> list, guarded three ways: **exact** integer-amount and **exact** string equality,
> never similarity; the engine must still raise `DUPLICATE_BANK_CREDIT` on the twin
> it did not match, so the money cannot be recognised twice; and cash drift and
> match-set symmetric difference stay measured against the **unrelaxed**
> comparison and remain at zero. Nothing else in the scorer is relaxed.

---

## Five things worth knowing

1. **Verification, not generation, is the loop.** A bank credit is the sum of many
   captures minus fees, GST, refunds and withholding; two payouts arrive
   consolidated, one arrives split. This is a constrained assignment problem, and
   asking a model to add the numbers up invites a hallucinated total that balances.
2. **False match rate is the headline, not match rate.** A system matching
   everything at 70% precision is worse than one matching 80% at 99.9%: a false
   match hides real money, an exception is merely work.
3. **The AI module contributes nothing measurable, and that is shipped as a
   finding.** Deleting it changes no number. Narration tie-breaks fired 0 times
   across 20 runs.
4. **The live layer proves its own decisions.** Continuous mutation with human
   overrides, windowed re-solve over a provably closed record set, an append-only
   hash-chained decision log, and a replay verifier that is a hard test gate.
5. **A limitation I documented had never been tested.** The generator could not
   produce a group larger than 2, so the "subset-sum bounded to 3" caveat was a
   guess. Testing it revealed a 61-point recall hole.

---

## Results

Regenerate: `python benchmark.py` → [`artifacts/benchmark.md`](artifacts/benchmark.md).

### Difficulty sweep — 40 seeds × 5 levels, 200 runs

| difficulty | match mean | match min | **FMR mean** | FMR max | exc P mean | exc P min | exc R mean |
|---|---|---|---|---|---|---|---|
| 0.0 pristine | 100.00% | 100.00% | **0.000%** | 0.000% | n/a | n/a | n/a |
| 0.5 | 100.00% | 100.00% | **0.000%** | 0.000% | 1.0000 | 1.0000 | 1.0000 |
| **1.0 default** | **100.00%** | **100.00%** | **0.000%** | 0.000% | 1.0000 | 1.0000 | 1.0000 |
| 2.0 | 99.80% | 96.00% | **0.000%** | 0.000% | 0.9973 | 0.9333 | 0.9995 |
| 3.0 | 99.26% | 95.24% | **0.000%** | 0.000% | 0.9941 | 0.9623 | 0.9970 |

Mean *and* min for every column. Quoting one without the other is how a write-up
lies by omission. The pristine row is load-bearing: an engine that cannot hit
100% on data with no anomalies has a bug, and every other number it reports is
noise.

### Multi-way consolidation — the bound that was never exercised

| config | match mean | match min | FMR mean | wall clock |
|---|---|---|---|---|
| k=3, window ±4 *(original)* | 41.49% | 16.00% | 0.000% | 2.4 ms |
| k=8, window ±4 | 50.74% | 18.52% | 0.000% | 3.1 ms |
| **k=8, look-back 16** *(shipped)* | **100.00%** | **100.00%** | **0.000%** | **2.6 ms** |
| k=10, look-back 24 | 100.00% | 100.00% | 0.000% | 3.3 ms |

Two findings. k=3 cost ~59 points of recall once realistic end-of-day
consolidations existed. And **the cardinality bound was never the binding
constraint**: an 8-way consolidation spans 8 settlement days, so a symmetric
±4-day window excluded its earliest members at any k. Raising k bought 9 points;
widening the look-back bought the other 49.

Meet-in-the-middle (Horowitz–Sahni) replaced naive enumeration, with ambiguity
detection: any target reachable by more than one distinct subset is refused.
Bounds are asymmetric — merges ≤8, splits ≤5 — because the process is:
consolidating 8 payouts into one credit is routine, a payout arriving in 7
separate credits is not.

### Generalisation — six classes no rule anticipates

Rolling reserves at deliberately implausible rates (3.7/4.25/6.15/8.8%),
reserve-release credits, value-date skew of 5–8 days, credits landing before the
settlement report, near-duplicate UTRs, chargeback debits.

| unmodeled level | match mean | match min | **FMR mean** | **FMR max** | exc P |
|---|---|---|---|---|---|
| 0.0 | 100.00% | 100.00% | **0.000%** | **0.000%** | 1.0000 |
| 1.0 | 96.55% | 87.50% | **0.143%** | 3.571% | 0.9445 |
| 2.0 | 91.28% | 75.00% | **1.100%** | 8.333% | 0.8840 |
| 3.0 | 89.12% | 65.38% | **0.235%** | 5.882% | 0.8671 |

Stated as the partial failure it is. The first version of this table read
7.855% / 13.419% / 19.591%.

### AI ablation — deleting the module changes nothing

| config | normal | multiway | unmodeled |
|---|---|---|---|
| AI **enabled** | 100.0000% / FMR 0.0000% | 100.0000% / 0.0000% | 90.2805% / 1.0969% |
| AI **deleted** | 100.0000% / FMR 0.0000% | 100.0000% / 0.0000% | 90.2805% / 1.0969% |

Identical to four decimal places. Narration tie-breaks invoked across 20 runs: **0**.

The framing is the finding, not a defence of it: **the only safe place for a model
in settlement reconciliation is as a constrained re-ranker after every hard
constraint has already been applied — and once those constraints are good, there
is very little left for it to decide.** A payout total is the sum of dozens of
random captures minus fees; it behaves as a near-unique key.
`ai_contribution.py` proves the mechanism works when its condition occurs (two
amount-identical payouts: without it both abstain, with it both resolve at 0.92)
and measures how rarely that condition arises. It is retained because it costs
one cosine per candidate pair. It is not branded as AI-powered matching.

### The live layer — 6 seeds × 60 mutation/override steps

| | |
|---|---|
| engine false matches | 3 / 120 (**2.50%**), 3% ceiling test |
| cash drift vs. full re-solve | **0 paise** |
| match-set symmetric difference | **0** |
| replay | **bit-identical** artifacts, cash position, event stream; hash chains 6/6 |
| overrides applied | 12 per seed, all on records the engine itself flagged |
| median windowed re-solve | 28.02 ms over 572 of 665 records |
| hop count for window closure | max **1** observed; limit shipped at 4 |
| full-resolve fallbacks | 7 of 360 (1.9%) |

---

## Architecture

**Settlement reconciliation is a constrained assignment problem, not a
classification problem.** Order is load-bearing: several of the worst bugs in this
repo were ordering bugs, not logic bugs.

```
INGEST      parse to integer paise · reject rows loudly, never drop
   │
L1  order ↔ payment
   │  R1.1 exact payment id → R1.2 exact order ref → R1.3 unique prefix
   │  R1.4 Hungarian assignment on (amount, date gap)
   │
L2  arithmetic — no matching, no AI, confidence exactly 1.0
   │  fee = contracted MDR · GST = 18% of fee · net = gross − fee − GST
   │  batch: declared payout = Σ net − refunds
   │
L3  settlement ↔ bank
   ├─▶ R3.0a  same-UTR re-postings, regardless of amount divergence
   ├─▶ R3.0b  duplicate clusters: same amount, pairwise narration similarity,
   │          conditional UTR veto, "a payout cannot land before it is instructed"
   ├─▶ R3.0c  near-duplicate short by a plausible withholding rate with a
   │          corrupted reference
   │
   │  R3.1/R3.2  identifier in narration; claims collected then resolved
   │             ├─ amount ties out                    → match 0.99
   │             ├─ shortfall at a PLAUSIBLE WITHHOLDING RATE
   │             │                      → match 0.95 + AMOUNT_MISMATCH
   │             └─ otherwise → downgrade to a hint, do not assert
   │  R3.3/R3.4  Hungarian on exact amount + date window
   │             cost = 0.1·date_gap + 0.5·(1 − narration_similarity)
   │             confidence from the MARGIN over the runner-up
   │  R3.4b     withholding-rate match, bidirectional uniqueness required
   │  R3.6/R3.7 meet-in-the-middle subset sum, merges ≤8 / splits ≤5
   │             ambiguous target        → AMBIGUOUS_SUBSET
   │             unique only because a twin was consumed → DUPLICATE_CLAIM
   │             pool over 40 candidates → POOL_CAP_EXCEEDED, no match written
   │             tranches >3 days apart  → SPLIT_SPANS_LONG_WINDOW
   │
ABSTAIN     confidence < threshold → withdraw the match, raise it as human work
ARTIFACTS   run.json · exceptions.csv · matches.csv · close_report.html
```

**Money is an integer.** Every amount is parsed to integer paise at the ingestion
boundary and stays an integer. `0.1 + 0.2 != 0.3`, and an engine carrying float
money reports balance failures that are artifacts of the representation and
indistinguishable from real ones — which makes the entire exception list
untrustworthy.

**Cost matrices.** L1 is feasible only when `gross == amount` exactly and the date
gap is inside the window; cost is `0.1 × date_gap_days`, pure distance, because
rule priority is expressed by *ordering* rather than weighting. L3 is feasible only
when the amount ties out to ±₹1; cost is `0.1 × date_gap_days + 0.5 × (1 −
narration_similarity)`, confidence `min(0.92, 0.55 + 0.8 × margin)` on ties. A high
similarity every candidate shares is not evidence. Both use
`scipy.optimize.linear_sum_assignment`, so assignment is globally optimal rather
than greedy.

**Blocking** is date window → exact-amount equality → cardinality bound. Sufficient
to 15k records. At 100k–1M the L1 residual matrix needs amount-bucket blocking;
not implemented, documented because a reviewer will ask.

### Where AI is used, and where it is refused

| | |
|---|---|
| **Used** | Narration similarity as a tie-break term inside the L3 cost matrix, when two settlements in the window share a payout total. |
| **Refused** | All arithmetic. Fees, GST, batch totals, balances are integer rules at confidence exactly 1.0. |
| **Refused** | All exact identifier joins. A regex is faster, free, and cannot hallucinate. |
| **Refused** | The final assignment decision. The model contributes a score; a constrained optimiser and a confidence threshold decide. |

**The model never emits a number that enters the ledger.** A hallucinated
settlement id fails the amount constraint and is discarded. In the optional LLM
path the model can only re-rank an already-constrained candidate set, so its worst
case is picking the wrong one of N — never conjuring an id that does not exist.
The LLM backend is off by default; with `--use-llm` and no `ANTHROPIC_API_KEY` it
falls back to local scoring and reports itself as SKIPPED rather than claiming
success.

### The live layer

**Windowed re-solve over a proven-closed record set.** `closed_record_set()` is the
single source of truth for what gets re-solved: touched records → their candidate
neighbourhood → hop-limited edge closure over prior matches → amount peers.
Closure means no prior match is ever half-inside — either every member is in the
set, so it can be retired and re-derived intact, or none is, so it is untouched.
`assert_window_is_closed()` checks this on **every** re-solve, at runtime, not only
in tests.

**Override isolation, three independent mechanisms.** Excluded from the engine's
slice by construction; asserted absent from the engine's input; asserted absent
from its output *and* from the accumulated match set. Exclusion is by instrument
(record id **and** the UTR it carries), global rather than windowed, and sticky for
the lifetime of the run. `OVERRIDE_LEAK` is a hard abort, never a warning. Two
tests deliberately poison a slice and a match set to prove the detector detects.
Overrides are accepted only on records the engine itself flagged — a human decision
on a confidently-matched record would entangle engine and human judgement and make
"engine-generated false match" unmeasurable.

**Sticky duplicate determinations.** A duplicate is only recognisable while the
credit it copies is in front of the engine; once a mutation deletes the original,
a single-pass engine sees a perfectly good unmatched credit. Pins live in the live
layer — the only component with memory across windows — are released when the
record is restated or removed, stay visible on the exception ledger, and are
re-earned on replay rather than replayed as inputs.

**Append-only decision log.** Every engine decision and human action is an
immutable event with `rule_version`, `thresholds_hash`, evidence, rejected
candidates, cash delta and a SHA-256 chain link. Timestamps are a logical clock
from a fixed base epoch, because bit-identical replay and a wall clock are
mutually exclusive; real time is recorded outside the hash. **Replay re-derives,
it does not copy**: only input events are re-applied, and the derived events and
artifacts must match. Copying recorded outputs would verify the file writer and
nothing else.

**Explicit refusals.** `POOL_CAP_EXCEEDED`, `AMBIGUOUS_SUBSET`, `DUPLICATE_CLAIM`
and `BELOW_CONFIDENCE_THRESHOLD` each name a distinct reason to decline, because
"no payout explains this" and "too many candidates to decide safely" demand
different actions. The ambiguity gate stops at two distinct solutions, which is
all that is needed to answer "is this ambiguous?"; raising the enumerator's
witness cap 3 → 8 → 16 was measured to change nothing, so full enumeration is not
shipped — measured, not assumed. Soundness is pinned by a test that saturates the
cap with 210 solutions to one target and asserts the gate still refuses.

**Exception taxonomy is versioned.** `schema_version: 1.0.0` on every artifact.
Reason codes are a closed enum, not free text: an engine emitting free-text
reasons cannot be scored against a ground truth, and downstream consumers break
silently when wording drifts. Every exception carries severity, cash-at-risk,
blocking data, candidates considered, and a suggested action.

---

## Tests

**106 tests**: `python -m pytest tests/`

**Unit** — money parsing across Indian/western/accounting formats, half-up
rounding, refusal to coerce garbage to zero, pristine-data perfection, determinism,
no record lost between CSV and report, every settlement matched-or-excepted,
threshold monotonicity, arithmetic confidence exactly 1.0, negative-payout debits,
ambiguity-gate soundness under witness saturation, AI degradation reported loudly.

**Metamorphic** — properties that must hold under transformation, which catch the
bugs that have no oracle:

| property | what a violation would mean |
|---|---|
| row order permutation | a greedy matcher or an unstable sort inside the assignment |
| all dates shifted +97 days | a hard-coded window boundary or epoch assumption |
| amount reformatting (lakh ↔ western ↔ `INR n.nn`) | format-dependent parsing at the boundary |
| all amounts × 10 | an absolute magnitude baked into a rule threshold |
| append a fully-refunded capture netting to zero | state leaking across candidates |
| matched ∩ unexplained = ∅ | the report double-counts |

**Live** — hash-chain tamper and reorder detection, bit-identical replay as a hard
gate, windowed re-solve equals full re-solve, cash drift exactly zero,
`sum(all event deltas) == final position`, override leak detection on poisoned
input and output, sticky-pin lifecycle, window closure at both edges, stale-edge
retirement, pool-cap refusal, and a zero-tolerance ceiling on adversarial FMR.

---

## What broke, and how I got out

Chronological. Each cost a real debugging session.

1. **The first run reported 100% and I did not believe it.** Seed 7 was luck; the
   40-seed sweep showed min 73.08% and FMR up to 9.09%.
2. **The generator injected an anomaly it did not label.** Wrong-fee injection left
   GST derived from the old fee, so the detector correctly raised six
   `GST_MISMATCH` and the evaluator scored them spurious. The detector was right;
   the truth was wrong.
3. **Ground truth demanded exceptions that should not exist.** Failed orders were
   labelled `ORDER_NO_PAYMENT`; no money moved, so they are not reconciliation
   exceptions.
4. **My own fix became the worst bug in the system.** R3.4b accepted the unique
   candidate within a 5% band and produced *every* false match, up to 9% FMR.
   Fixed by replacing the tolerance with a domain constraint: a shortfall is
   withholding, and withholding happens at *rates* (0.1/1/2/5% ±2bp), with
   uniqueness required in both directions.
5. **The AI layer contributed exactly zero, and I shipped that finding.**
   `residual_seen: 0` across 80 runs — the fuzzy rule required an exact amount
   match in-window, but any such candidate was already consumed upstream.
   Unreachable code that would have shipped as "AI-powered matching" on a slide.
6. **An engine that only reads credits loses money.** A batch whose refunds exceed
   captures has a negative payout; the gateway *debits* to recover it. The
   `credit_paise > 0` filter silently dropped every recovery debit.
7. **Two real exceptions annihilated into one false match.** A payout that never
   landed and a duplicated credit differed by ~1%, so the withholding rule linked
   them — making missing money look reconciled. Fixed by ordering.
8. **A corrupted identifier is not a surviving identifier.** `SETL_0019` garbled to
   `SETL_0009`; the regex extracted a genuinely valid id and the engine
   confidently matched the wrong settlement.
9. **I documented a limitation I had never tested.** The generator had never
   produced a group larger than 2, so the k=3 bound had never once been exercised.
   Testing it revealed a 59-point recall hole — and then that the bound was not
   even the real constraint; the date window was.
10. **The old subset-sum returned the first subset that hit the target.** Two
    subsets summing to one credit is a coin flip dressed as a match. Latent at
    k=3, fatal at k=8.
11. **The engine was right and my ground truth was wrong.** Under rolling reserves
    the engine paired the short credit with the later release credit; the payout
    genuinely arrived in two tranches summing to exactly P. What it legitimately
    owed was a signal that working capital had been held, so
    `SPLIT_SPANS_LONG_WINDOW` was added and the label corrected. Distinguishing
    "my system is wrong" from "my measurement is wrong" was the hardest judgement
    in the build.
12. **Mixing signs in subset-sum conscripts chargebacks into explaining payouts.**
    Sign-partitioned: a credit is explained only by positive payouts.
13. **The window closure exploded into a full re-solve wearing a costume.** Every
    pulled-in match re-applied the full ±16-day margin, so the window saturated at
    629 of 647 records within two iterations.
14. **A contested identifier was resolved first-wins.** A near-duplicate whose
    narration differs in its last character leaves the UTR intact, so both credits
    legitimately claim the same settlement and the engine matched whichever it
    reached first.
15. **That fix cost 6 points of match rate before it gained any.** Retiring every
    losing contender broke split payouts, whose halves *legitimately* share an
    identifier. A contender is a duplicate only if it also ties out to the full
    payout.
16. **Mutations that read ground truth are not replayable.** Truth is a measurement
    artifact, not an input, so replay starts with an empty one and those operators
    silently became no-ops — the event streams diverged at the very first window.
17. **Overrides cannot be planned in advance.** A later mutation usually resolves
    whatever was flagged at planning time, and the safety check then correctly
    refuses. Plans now carry override *slots* resolved against live state.
18. **A 1-paise mutation desynchronised the immutable log from the ledger it
    claims to record.** The position was sampled *after* the mutation was applied,
    so the mutation's own effect fell into a gap no event covered; the log
    under-reported by exactly the mutation delta, at every magnitude. The replay
    verifier compared state to state and ratified it. Every retirement also
    carried `cash_delta = 0` and no reason code.
19. **My UTR veto was an over-correction.** Added to stop two unrelated
    consolidated payouts (which read 0.9767 alike) being clustered, it was written
    as an unconditional override. A split tranche narrates as `…-PART-<UTR>`, so a
    one-character corruption lands *inside* the UTR — and the veto then blocked
    clustering of a phantom that shared the settlement id, the exact amount, and
    97% of the narration.
20. **Duplicate clustering was star-shaped, not pairwise.** With three lines in a
    bucket — two legitimate split halves plus a phantom of one of them — the
    earliest line was the *other* half, dissimilar to both, so the pair that
    actually matched was never compared. That single blind spot was the last
    engine false match on the 10-class suite.
21. **The window was a date interval, not a record set.** Closure was a side effect
    of arithmetic on dates rather than a property of the graph. Rewriting it as a
    hop-limited closure and asserting it at runtime immediately caught two things
    no cash or FMR check would have: matches referencing records a mutation had
    deleted, and mutations that delete their own seed (10% of all windows were
    falling back to whole-batch for that reason alone).
22. **Override isolation leaked, twice.** A UTR exclusion is global but was applied
    only inside the override's window, so a re-posting matched earlier kept its
    match. And the leak check inspected only newly-asserted matches, so one that
    *predated* the override survived. Both showed as cash drift of lakhs.
23. **Expanding the generator broke the zero, and that was the point.** Six new
    classes took adversarial FMR from 0.0000% to **6.72%**. Six of eight failures
    came from a phantom that posts a day *before* the credit it copies: "keep the
    earliest" is a heuristic with no basis, and it made the phantom the survivor
    while the real credit was retired. Replaced with a domain rule — a payout
    cannot land before it is instructed.
24. **A phantom can differ on two axes at once.** Short by exactly a plausible
    withholding rate *and* one character of the reference, defeating amount
    clustering, UTR clustering, and shaped to be accepted by the withholding-rate
    gate. Now clustered on that signature directly.
25. **Hitting the candidate-pool cap was a silent skip.** The target fell through
    to the generic residue path and was reported as an ordinary unexplained credit,
    leaving an operator unable to tell "nothing explains this" from "too many
    candidates to decide safely."
26. **I asserted a trade-off instead of measuring it.** I documented the
    single-claim contingency rule as "specified and deliberately not shipped"
    because it would cost too much recall. Measured, it costs **exactly zero** on
    all four static modes. That is the same error as documenting an untested
    limitation (#9), made a second time, and it is why every threshold in this repo
    is now a measured number.

---

## Generative assumptions

Every number here is conditional on these. Violate one and the results do not
transfer.

- One currency, integer paise. No FX leg, no markup, no foreign settlement.
- One contracted MDR for all methods; no per-method or volume-tiered pricing.
- Flat 18% GST on fee; no IGST/CGST/SGST split, no TCS under section 52.
- Settlement at T+2, one batch per capture date.
- Narration follows one synthetic bank's conventions; MT940, BAI2 and per-bank
  dialects are not modelled.
- One value-date field; real statements disagree across value, transaction and
  posting dates.
- Withholding occurs at 0.1 / 1 / 2 / 5% only.
- Rolling reserves modelled as a single delayed tranche, not a 90–180 day
  contractual schedule.
- UTRs are 12 digits and near-unique; collisions are injected but rare.
- One bank account; no sweep transfers between accounts.
- Chargebacks modelled as a debit plus optional representment credit; no
  pre-arbitration, no partial liability.
- Refunds net against the batch they fall into; instant refunds are not modelled.
- Settlement report *membership* is restated only via the partial-restatement class.

---

## What this does not claim

- **Not zero false matches under arbitrary damage.** 2.50% on the 16-class suite,
  and that is a floor, not a proof.
- **Not validated on real data.** The generator models a payments business and then
  damages the exports; it does not know about the matcher, so the evaluation is not
  circular — but it is still a closed set of anomalies. The first real bank export
  will produce something none of the sixteen classes anticipates.
- **Not fine-grained incremental maintenance of a global constraint graph.** The
  unit of recomputation is a closed date-anchored record set.
- **Not immunity to arbitrary adversarial human action.** Overrides are *isolated*;
  they are also restricted to records the engine itself flagged, and the
  measurement rests on that restriction.
- **Not a minimal window.** Proven-closed and minimal are different claims and only
  the first is made: the closed set is median 517 of ~659 records, because on a
  28-day batch with a 16-day look-back the candidate neighbourhood is inherently
  most of it. It falls to ~12% of the batch at a 180-day horizon.
- **Not production controller software.** No exception ageing, no forced *positive*
  match with downstream re-solve, no multi-account or multi-currency support.
- **Not benchmarked with an LLM in the loop.** No API key in the build
  environment, so the optional path's lift over local scoring is unmeasured and
  reported as unmeasured rather than estimated.

---

## Layout

```
recon/
  money.py        integer paise, Indian-format parsing, half-up rounding
  models.py       typed records, closed exception taxonomy, severity, cash impact
  generate.py     5-ledger generator, labelled truth, tiers, multiway + unmodeled
  normalize.py    ingestion boundary; parse failures reported, never dropped
  subsetsum.py    cardinality-constrained meet-in-the-middle + ambiguity detection
  match.py        L1/L2/L3 engine, Hungarian, duplicate clustering, withholding
                  gate, subset-sum, explicit refusals, abstention
  ai.py           char-trigram similarity (default) + optional LLM re-rank
  evaluate.py     P/R/F1, false-match rate, per-tier, abstention curve
  events.py       append-only decision log, SHA-256 chain, logical clock
  live.py         closed record set, windowed re-solve, override isolation, pins
  mutations.py    16 replayable mutation operators (data effect is truth-free)
  report.py       run.json, exceptions.csv, matches.csv, close_report.html
  cli.py          single command, cold start
replay.py         the replay gate: re-derive from inputs, require bit-identical
mutate.py         adversarial mutation + safe-override harness
benchmark.py      every static evidence table
ai_contribution.py  isolated measurement of the AI layer
stress.py         the original 200-configuration sweep
tests/            106 tests
run.sh            everything, offline, ~12 minutes
```

The HTML report is rendered from `run.json` with no build step, no CDN and no
external resource, so it opens offline and cannot drift from the numbers it
displays.
