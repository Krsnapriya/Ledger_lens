# What broke, and how I got out

Chronological. Each entry cost a real debugging session. Kept in full here so
[README.md](README.md) can stay skimmable; nothing is summarised or softened
by moving it — every entry below is verbatim.

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
26. **Disabling a safety flag crashed the engine, in a path that had been
    unreachable for the life of the repo.** An external document proposed that
    the uniqueness-certificate flags were the project's core algorithmic
    advance and asked for a certified-vs-uncertified head-to-head. Building it
    required actually running with `require_unique_assignment=False`, which
    crashed on `sims[1]` in the narration-tiebreak branch: a settlement with
    exactly one feasible candidate, contested by another settlement, has
    nothing to compare a narration margin against, and no code path handled
    it because the certificate had always refused that shape before this line
    was reached. Fixed with a dedicated degrade path. The head-to-head itself
    then came back identical on every distribution tested — synthetic
    baseline, unmodeled, the full adversarial suite, and BenchRec, both flags
    independently — because the general confidence-threshold abstention
    already withdraws everything either certificate would refuse. The
    document's causal claim (that this mechanism drove the earlier
    0.220%→0.134% BenchRec improvement) was wrong; that improvement was
    entirely the withholding-rule disable. Corrected rather than left standing,
    the same way #9 and #26 (old numbering) were.
27. **A record was simultaneously reconciled and reported as unexplained.** Found
    by an external review, not by me. The merged subset-sum branch refuses a
    contingent group *before* appending the match; the split branch appended
    first, then ran the same check and `continue`d without marking the tranche
    lines used. The already-asserted match survived while its bank lines fell
    through to residue, so one run produced a record that was reconciled in
    `matches.csv` and reported as money with no source in `exceptions.csv`. Every
    guard was blind to it: the windowed and full re-solve paths run the same
    engine, so cash drift and match-set symmetric difference both agreed while
    both were wrong, and the generator rarely emits the split-with-twin shape.
    Fixed by mirroring the merged branch, and the underlying invariant — no record
    may be both matched and reported without a counterpart — is now asserted at
    runtime inside `reconcile()` and swept across 30 seed/mode combinations.
28. **A withheld match still emitted the residue it contradicted.** When R3.4b
    drives confidence to 0.45 so abstention withdraws a non-unique short-credit
    match, the entities were never marked used, so residue fired too. An operator
    saw three statements about one situation: "found a likely short credit", "no
    bank credit at all", and "credit with no source" — the last two contradicting
    the first. Suppressed at the reporting layer rather than by consuming the
    entities in the matcher, which would have blocked later rules from matching
    them for real. Measured at zero recall cost across all four static modes.
29. **I asserted a trade-off instead of measuring it.** I documented the
    single-claim contingency rule as "specified and deliberately not shipped"
    because it would cost too much recall. Measured, it costs **exactly zero** on
    all four static modes. That is the same error as documenting an untested
    limitation (#9), made a second time, and it is why every threshold in this repo
    is now a measured number.

---



30. **The uniqueness-certificate claim, tested and rejected.** An external
    document proposed that this engine's core algorithmic advance is
    "contingent uniqueness certification" and that disabling it on BenchRec
    would show measurably worse FMR — crediting part of the earlier
    0.220%→0.134% improvement to it. Tested directly, both flags
    (`require_unique_assignment`, `refuse_contingent_subsets`) independently,
    on every distribution in the repo: BenchRec, synthetic unmodeled, the full
    6×60 adversarial suite. Every cell came back byte-identical, certified or
    not. The 0.220%→0.134% improvement was 100% the withholding-rule disable;
    both flags were `True` throughout that comparison, and crediting them was
    wrong. Mechanism: both certificates refuse by driving confidence to a
    value the *general* confidence-threshold abstention (0.70, independent of
    either flag) already withdraws. Disabling them doesn't admit a
    wrongly-confident match — it re-routes the same refusal through a generic
    withdrawal instead of a named reason code. The measured safety property
    belongs to confidence-gated abstention, the actual headline invariant
    since the first difficulty sweep, not to either named mechanism. What the
    exercise *did* produce: disabling `require_unique_assignment` crashed the
    engine — `IndexError` in the narration-tiebreak branch, which assumes at
    least two candidates to compare a margin against. Unreachable for the
    project's entire life because the certificate always refused that shape
    first. A settlement with exactly one feasible candidate, contested by a
    different settlement, has nothing to compare narration against. Fixed
    with a dedicated degrade path, verified with a full regression pass before
    being trusted. Testing someone else's unverified claim about this system
    found the bug the claim was never checking for.

31. **A rule that sounded free wasn't: the causality discard.** Proposed: a
    credit naming a settlement but dated before that settlement's instruction
    date cannot be that instruction's landing, so discard it outright.
    Measured directly with a minimal repro before adding it: the generator's
    `credit_before_report` class shifts a credit 5–8 days *before* its
    settlement, deliberately outside the default ±4-day window — and the
    agent's `RETRY_WIDER_WINDOW` action widens to 40 days specifically to
    recover cases like it. At the default window, the case already abstains
    correctly with no discard needed. At the agent's widened window, an
    unconditional discard would silently convert a working recovery
    mechanism into a permanent block. Declined — not because the idea is
    wrong, but because "before" is only evidence of impossibility at the
    window the credit was actually found *outside*, and a rule that helps at
    one scope can break a mechanism at another.

32. **A third of a real dataset's unmatched records had no exception naming
    them at all.** External critique: "certificates are audit, not safety" —
    correct, already established (#30) — prompted a closer look at whether
    the audit trail itself was actually complete. It wasn't.
    `AMBIGUOUS_ASSIGNMENT` refuses a contested pairing and logs the exception
    against the SETTLEMENT side only; the specific bank credit the optimiser
    would have chosen is marked used and removed from the pool, but never
    given an exception of its own. Measured precisely on BenchRec's real held
    -out set: 10,119 bank credits — 31.6% of the entire scored set — had
    neither a prediction nor any exception explaining why, confirmed exactly
    by summing the three reason codes that DO carry bank-side ids
    (`DUPLICATE_BANK_CREDIT` + `POOL_CAP_EXCEEDED` + `UNEXPLAINED_BANK_CREDIT`
    = 6,759 = precisely the non-silent count). "Every exception is auditable"
    is a claim this project makes repeatedly; it was false for a third of a
    real dataset's unmatched records. Fixed by filing a second, linked
    exception on the specific bank credit consumed by the refusal. Changes no
    match, no FMR, no coverage number — verified identical before and after —
    only makes the exception ledger exhaustive, which is a correctness
    requirement on its own, not a nice-to-have.

33. **No automatic detector existed for "this rule is net-harmful on this
    distribution."** External critique, correctly: the withholding-rate bug
    (#23) was found by printing a by-rule table and reading it, which does
    not generalise to the next domain-mismatched rule on the next real
    dataset. Built `recon/rule_audit.py`: per-rule precision against a
    labelled sample, flagged when precision falls below a bar with enough
    volume to be signal rather than three unlucky matches. It recommends, it
    does not auto-disable — a false-positive flag silently switching off a
    real rule is worse than a human missing one. Verified two ways before
    trusting it: run retroactively on BenchRec with the withholding rule
    re-enabled, it flags `R3.4b_WITHHOLDING_RATE_MATCH` automatically (0.0%
    precision, n=18) — proof it would have caught the exact bug without
    anyone reading a table; run on clean synthetic data, it flags nothing —
    proof it does not invent problems that are not there.

34. **Tried to buy back real-data coverage with narration; it didn't work,
    and I'm keeping the negative result.** The 47% BenchRec coverage decomposes
    (measured) into 1,980 credits whose true ledger partner isn't in the eval
    pool at all (irreducible) and ~14,900 that are refused as
    `AMBIGUOUS_ASSIGNMENT` because amount+date can't break the tie. A probe on
    the residual candidate pool showed narration similarity picks the true
    partner 74.3% of the time in ambiguous groups — promising enough to build a
    gated escalation (`R3.4c`): when amount+date ties, let a decisive narration
    margin resolve it. It required adding a `descriptor` field to Settlement so
    the fuzzy layer could compare a bank narration against the ledger row's real
    allocation text instead of a synthesised Razorpay-format string — a genuine
    missing capability, correctly identified. But live, the rule promoted 8
    matches and **all 8 were wrong: 0% precision.** The 74.3% probe measured the
    full residual pool; by the time control reaches this branch the Hungarian
    assignment has already consumed the easy amount-equal candidates, leaving a
    hard rump where narration misleads. Coverage flat, FMR worse
    (0.105%→0.152%). Reverted to default-off. The rule and the descriptor field
    are kept (config-gated, harmless off) because they document a tested,
    measured, rejected direction — and because the `rule_audit.py` built two
    findings ago **automatically flags R3.4c as harmful (0% precision) if it is
    ever enabled**, which is the safety net catching the very thing I built to
    test the coverage critique. Coverage on this dataset is closer to a real
    floor than a fixable gap: the ambiguity is in the data, not the engine.
