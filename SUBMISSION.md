# Buildathon submission — copy-paste answers

Not part of the project. Delete this file before making the repo public, or keep
it; it does no harm. Track 04, AI Finance Controller.

---

## Project name

LedgerLens

## What it solves

Finance teams reconcile gateway settlements against bank statements by hand. A
bank credit is not one payment — it is dozens of captures minus fees, GST,
refunds and withholding, and payouts arrive merged or split. LedgerLens closes
that loop over five ledgers, reports a measured match rate and false-match rate,
and emits an auditable exception ledger with a cash-at-risk amount and a
suggested action for every record it could not resolve. It then survives 60 steps
of continuous data mutation and human overrides per run, with an append-only
hash-chained decision log that replays to bit-identical artifacts.

## What broke, and how you got out

> This is the field they say they read first. One story, told with the mechanism
> and the number. Do not list several.

Under rolling reserves, my engine paired a short bank credit with a release
credit that arrived seven days later, and my test data scored that as 84 false
matches. The engine was right and my ground truth was wrong — the payout
genuinely did arrive in two tranches summing to exactly the declared amount, and
reconciling them together is the correct answer. What the engine legitimately
owed and had not provided was a signal that seven days of working capital had
been held, so I added SPLIT_SPANS_LONG_WINDOW and corrected the label. Telling
"my system is wrong" apart from "my measurement is wrong", without using the
second as an excuse for the first, was the hardest judgement in the build — I had
to prove it by showing the cash position was identical either way before I was
willing to touch the ground truth.

### Alternate, if you want the arc instead of the judgement call

My engine had zero false matches across 200 runs. I did not trust that, so I
wrote six new anomaly classes designed specifically to attack the rules the first
ten had produced — duplicates that post before the original, phantoms short by a
plausible withholding rate with a corrupted reference, UTR collisions across
banks. The false-match rate went from 0.0000% to 6.72% immediately. Six of the
eight failures traced to one bad assumption: my duplicate resolver kept the
earliest credit, which is a heuristic with no basis, so a phantom posting a day
early became the survivor and the real credit was retired. I replaced it with a
domain rule — a payout cannot land before it is instructed — and shipped 2.50%
with a hard ceiling test. The distance between 0.0000% and 2.50% is the measured
worth of a zero, and it is the most useful number in the repo.

---

## Video script beats — 5:00 hard cap

- **0:00** Open cold on the reversal. No introduction. "Zero false matches across
  200 runs. Then I attacked my own rules and it went to 6.72%. This is about why
  the second number is the one worth trusting."
- **0:25** The problem in a controller's language. Five CSVs on screen, three
  seconds, no column narration.
- **1:10** One technical claim: constrained assignment, not classification.
  Integer paise. Hungarian plus meet-in-the-middle. Show the rule-ordering
  diagram — the only architecture visual.
- **2:00** The AI answer as strength: built it, measured it, it contributed
  exactly zero, deleting it changes no number to four decimal places.
- **2:40** One failure story, in full.
- **3:40** Live layer, fast: mutation, hash-chained log, bit-identical replay,
  zero cash drift, override isolation.
- **4:20** Close on the residual and the thirteen generative assumptions.

On-screen text at the moment you say each: **0.000%** · **6.72%** · **2.50%**.

Cut: your introduction, install steps, code walkthrough, tech-stack list, feature
tour. If a sentence has no number and no decision in it, cut it.

---

## If asked about forecasting

"I picked multi-source reconciliation and closed it fully rather than
half-building a forecaster on top. The cash position I report is the reconciled
one, not a projection." Choosing depth over breadth is defensible; claiming
breadth you do not have is not.

## If asked whether it is really an agent

"The agent is the controller loop. It reconciles, triages its own exception queue
by cash at risk, decides which exceptions it can chase and which need a human,
takes bounded actions, checks whether they helped, reverts them if they did not,
and stops when nothing remains it may act on. What it deliberately cannot do is
assert a match — an agent that can force an attribution can manufacture a false
match, and that is the number the whole project exists to protect."

## Panel prep — be able to answer these cold

1. Why is `WINDOW_RESOLVED` a zero-delta event?
2. Why does merged-payout cash attribution need a claim counter rather than
   summing each match's bank lines?
3. Why is the UTR veto conditional on settlement-id corroboration rather than
   absolute?
4. Why are the subset-sum bounds asymmetric — merges 8, splits 5?
5. Why is the ambiguity gate sound at witness_cap 3? What does the saturation
   test prove?
6. Why is closure computed over a record set rather than a date interval?
7. What are the three independent mechanisms enforcing override isolation, and
   why is one of them not enough?
8. Why is the oracle relaxed for byte-identical credits, and what three guards
   stop that being special pleading?
9. Where does the engine refuse rather than guess, and why is each refusal a
   distinct reason code?
10. What is the single most likely thing a real bank export breaks first?
11. Why can the agent not assert a match, and what would break if it could?
12. Why is a BATCH_ARITHMETIC_MISMATCH never written off?
