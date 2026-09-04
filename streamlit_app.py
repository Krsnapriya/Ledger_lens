"""
LedgerLens — Demo Console (Streamlit)
Read-only presentation shell over existing artifacts.
Does NOT re-implement matching. Does NOT call external APIs.

Run from repo root:
  streamlit run streamlit_app.py

Expected layout (all optional — app degrades gracefully):
  artifacts/run.json
  artifacts/matches.csv
  artifacts/exceptions.csv
  artifacts/benchmark.json
  artifacts/close_report.html
  FAILURE_LOG.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "artifacts",
    ROOT,
    ROOT.parent / "artifacts",
    Path.cwd() / "artifacts",
    Path.cwd(),
]


def find_file(*names: str) -> Path | None:
    for base in CANDIDATES:
        for name in names:
            p = base / name
            if p.is_file():
                return p
    return None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback headline numbers (published README — used only when artifacts missing)
# ---------------------------------------------------------------------------

FALLBACK_RUN = {
    "settlement_match_rate_pct": 100.0,
    "false_match_rate_pct": 0.0,
    "l3_precision": 1.0,
    "l3_recall": 1.0,
    "unreconciled_exposure_inr": 620633.43,
    "records_ingested": 635,
    "matches_asserted": 285,
    "exceptions_count": 27,
    "seed": 7,
    "orders": 300,
    "difficulty": 1.0,
}

FALLBACK_BENCHREC = [
    {"system": "MatcherByChatGPT (external ML)", "coverage": 61.79, "fmr": 4.798},
    {"system": "This engine, naive (threshold=0)", "coverage": 68.45, "fmr": 13.251},
    {"system": "This engine, certified (default)", "coverage": 47.29, "fmr": 0.105},
]

FALLBACK_CLAIMS = [
    ("Engine FMR — in-distribution (200 runs)", "0.0000%"),
    ("Engine FMR — multi-way consolidation", "0.0000%"),
    ("Engine FMR — 10-class adversarial mutation", "0.0000%"),
    ("Engine FMR — 16-class adversarial suite", "2.50% (3/120)"),
    ("Cash-position drift (non-overridden)", "0 paise"),
    ("Match-set vs full re-solve", "0"),
    ("Event-log replay", "bit-identical"),
    ("Human-asserted in engine matches", "0"),
    ("AI module deleted", "no change to 4 decimals"),
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LedgerLens — Finance Controller",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #0f1419;
        border: 1px solid #243044;
        border-radius: 10px;
        padding: 12px 14px;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; }
    .ll-hero {
        background: linear-gradient(135deg, #0b1220 0%, #132337 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .ll-hero h1 { margin: 0 0 0.35rem 0; font-size: 1.75rem; color: #f1f5f9; }
    .ll-hero p { margin: 0; color: #94a3b8; font-size: 0.95rem; }
    .ll-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 999px;
        margin-right: 6px;
    }
    .ll-warn {
        background: #1c1917;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #fde68a;
        margin: 0.75rem 0;
    }
    .ll-ok {
        background: #052e16;
        border-left: 4px solid #22c55e;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #bbf7d0;
        margin: 0.75rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

run_path = find_file("run.json")
matches_path = find_file("matches.csv")
exceptions_path = find_file("exceptions.csv")
benchmark_path = find_file("benchmark.json")
failure_log_path = find_file("FAILURE_LOG.md")
html_report_path = find_file("close_report.html")

raw_run = load_json(run_path) or {}
matches_df = load_csv(matches_path)
exceptions_df = load_csv(exceptions_path)
benchmark = load_json(benchmark_path)


def flatten_run(raw: dict[str, Any]) -> dict[str, Any]:
    """run.json is nested by section; the pages read flat keys."""
    flat: dict[str, Any] = {}
    for key in ("seed", "difficulty", "run_id", "generated_at", "python"):
        if key in raw:
            flat[key] = raw[key]

    throughput = raw.get("throughput", {})
    for src, dst in (
        ("records_ingested", "records_ingested"),
        ("matches_asserted", "matches_asserted"),
        ("exceptions", "exceptions_count"),
    ):
        if src in throughput:
            flat[dst] = throughput[src]

    match_rate = raw.get("match_rate", {})
    for key in ("settlement_match_rate_pct", "false_match_rate_pct"):
        if key in match_rate:
            flat[key] = match_rate[key]

    l3 = raw.get("match_scores", {}).get("L3_settlement_bank", {})
    if "precision" in l3:
        flat["l3_precision"] = l3["precision"]
    if "recall" in l3:
        flat["l3_recall"] = l3["recall"]

    # value section is in integer paise
    value = raw.get("value", {})
    if "unreconciled_exposure" in value:
        flat["unreconciled_exposure_inr"] = value["unreconciled_exposure"] / 100
    if "exception_cash_at_risk" in value:
        flat["exception_cash_at_risk_inr"] = value["exception_cash_at_risk"] / 100

    return flat


run = flatten_run(raw_run)
run_is_live = bool(run_path) and "settlement_match_rate_pct" in run

for k, v in FALLBACK_RUN.items():
    run.setdefault(k, v)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### LedgerLens")
    st.caption("Track 04 · AI Finance Controller")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Claims & Residual",
            "Exception Ledger",
            "Matches",
            "BenchRec External",
            "Live / Mutation",
            "Agent",
            "Failure Log",
            "Architecture",
            "Reproduce",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Artifacts detected")
    st.markdown(
        f"""
| file | status |
|------|--------|
| run.json | {'✓' if run_is_live else '— fallback'} |
| matches.csv | {'✓' if matches_path else '—'} |
| exceptions.csv | {'✓' if exceptions_path else '—'} |
| benchmark.json | {'✓' if benchmark_path else '—'} |
| FAILURE_LOG.md | {'✓' if failure_log_path else '—'} |
"""
    )
    st.markdown("---")
    st.caption("Read-only shell. Engine is not invoked from this UI.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ll-hero"><span class="ll-tag">LedgerLens</span>'
        f'<span class="ll-tag">Track 04</span>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta=delta)


# ===========================================================================
# PAGES
# ===========================================================================

if page == "Overview":
    hero(
        "Multi-source settlement reconciliation",
        "Measured residual instead of a protected zero. High precision under "
        "adversarial mutation and on real held-out data — with explicit coverage cost.",
    )

    fmr = run.get("false_match_rate_pct", run.get("false_match_rate", 0.0))
    match_rate = run.get("settlement_match_rate_pct", run.get("match_rate_pct", 100.0))
    exposure = run.get("unreconciled_exposure_inr", run.get("unreconciled_exposure", 0))
    records = run.get("records_ingested", run.get("n_records", 635))

    metric_row(
        [
            ("Settlement match rate", f"{float(match_rate):.2f}%", None),
            ("False match rate", f"{float(fmr):.3f}%", "headline invariant"),
            ("Unreconciled exposure", f"₹{float(exposure):,.2f}", None),
            ("Records in batch", f"{int(records)}", f"seed {run.get('seed', 7)}"),
        ]
    )

    st.markdown("---")
    c1, c2 = st.columns((1.2, 1))

    with c1:
        st.subheader("Thesis")
        st.markdown(
            """
- **False match rate** is the number that decides whether this is usable — not match rate.
- A system matching everything at 70% precision is worse than one matching 80% at 99.9%.
- **Refuse rather than guess.** Confidence-gated abstention is the primary safety invariant.
- The gap between **0.0000%** (tuned) and **2.50%** (16-class adversary) is the measured worth of a zero.
            """
        )
        st.markdown(
            '<div class="ll-ok"><b>Not claimed:</b> full straight-through processing. '
            "Coverage cost on real data is stated, not hidden.</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.subheader("One-command run")
        st.code(
            "pip install -r requirements.txt\n"
            "python -m recon.cli --orders 300 --seed 7\n"
            "./run.sh   # full offline regeneration ~12 min",
            language="bash",
        )
        st.caption("No API key · No network · No model download · Python 3.12")

    if html_report_path:
        st.info(f"Full offline HTML report available at `{html_report_path}`")

elif page == "Claims & Residual":
    hero("Precise claims", "Falsifiable numbers. Residual published, not protected.")

    st.subheader("Hard claims (synthetic)")
    st.dataframe(
        pd.DataFrame(FALLBACK_CLAIMS, columns=["Claim", "Measured"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("The residual arc")
    st.markdown(
        """
| Stage | Engine FMR | What happened |
|-------|------------|---------------|
| Tuned on 10 anomaly classes | **0.0000%** | In-distribution perfection |
| Expanded to 16 classes built to attack rules | **6.72%** | Generator adversary broke the zero |
| Domain fixes + ceiling | **2.50%** (3/120) | Hard 3% ceiling test; residual bounded |
        """
    )
    st.markdown(
        '<div class="ll-warn">A class the generator still cannot produce remains '
        "invisible by construction — 2.50% is a <b>floor</b> on the true rate under "
        "broader damage, not a proof.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Scorer relaxation (only one)")
    st.markdown(
        """
The `duplicate_before_original` class injects a **byte-identical** credit (same amount,
same narration; only value date + row id differ). No observable field separates them.
Oracle exonerates when cited credit is byte-identical to a truth row, guarded by:

1. Exact integer amount + exact string equality (never similarity)
2. Engine must still raise `DUPLICATE_BANK_CREDIT` on the unmatched twin
3. Cash drift & match-set diff stay measured on the **unrelaxed** comparison → 0
        """
    )

elif page == "Exception Ledger":
    hero("Exception ledger", "Every refusal has a reason code, severity, and cash impact.")

    if exceptions_df is not None and len(exceptions_df) > 0:
        st.caption(f"{len(exceptions_df)} exceptions loaded from artifacts")
        cols = list(exceptions_df.columns)
        money_cols = [c for c in cols if "cash" in c.lower() or "impact" in c.lower() or "inr" in c.lower()]
        reason_cols = [c for c in cols if "reason" in c.lower() or "code" in c.lower()]
        sev_cols = [c for c in cols if "sev" in c.lower()]

        c1, c2, c3 = st.columns(3)
        with c1:
            if reason_cols:
                reasons = ["(all)"] + sorted(exceptions_df[reason_cols[0]].astype(str).unique().tolist())
                reason_f = st.selectbox("Reason code", reasons)
            else:
                reason_f = "(all)"
        with c2:
            if sev_cols:
                sev_vals = exceptions_df[sev_cols[0]].unique().tolist()
                numeric = pd.to_numeric(pd.Series(sev_vals), errors="coerce")
                if not numeric.isna().any():
                    # severity is a numeric priority; string sort would order 100 before 50
                    sev_vals = [v for _, v in sorted(zip(numeric, sev_vals), reverse=True)]
                else:
                    sev_vals = sorted(str(v) for v in sev_vals)
                sevs = ["(all)"] + [str(v) for v in sev_vals]
                sev_f = st.selectbox("Severity", sevs)
            else:
                sev_f = "(all)"
        with c3:
            q = st.text_input("Search text", "")

        view = exceptions_df.copy()
        if reason_f != "(all)" and reason_cols:
            view = view[view[reason_cols[0]].astype(str) == reason_f]
        if sev_f != "(all)" and sev_cols:
            view = view[view[sev_cols[0]].astype(str) == sev_f]
        if q:
            mask = view.astype(str).apply(
                lambda r: q.lower() in " ".join(r.values).lower(), axis=1
            )
            view = view[mask]

        st.dataframe(view, use_container_width=True, hide_index=True, height=420)

        if money_cols:
            try:
                total = pd.to_numeric(view[money_cols[0]], errors="coerce").sum()
                st.metric("Cash at risk (filtered)", f"₹{total:,.2f}")
            except Exception:
                pass
    else:
        st.warning("exceptions.csv not found — showing taxonomy only.")
        st.markdown(
            """
| Reason code | Meaning |
|-------------|---------|
| `UNEXPLAINED_BANK_CREDIT` | No feasible settlement subset |
| `SETTLEMENT_NOT_IN_BANK` | Settlement has no bank counterpart |
| `AMBIGUOUS_SUBSET` | Multiple distinct subsets reach target |
| `DUPLICATE_CLAIM` | Unique only because twin already consumed |
| `POOL_CAP_EXCEEDED` | Too many candidates — hard refuse |
| `AMOUNT_MISMATCH` | Shortfall not at plausible withholding rate |
| `BELOW_CONFIDENCE_THRESHOLD` | Withdrawn by abstention gate |
| `OVERRIDE_LEAK` | Human-asserted record in engine path — abort |
            """
        )

elif page == "Matches":
    hero("Asserted matches", "Every match carries rule, confidence, and evidence.")

    if matches_df is not None and len(matches_df) > 0:
        st.caption(f"{len(matches_df)} matches loaded")
        rule_cols = [c for c in matches_df.columns if "rule" in c.lower()]
        conf_cols = [c for c in matches_df.columns if "conf" in c.lower()]

        c1, c2 = st.columns(2)
        with c1:
            if rule_cols:
                rules = ["(all)"] + sorted(matches_df[rule_cols[0]].astype(str).unique().tolist())
                rule_f = st.selectbox("Rule", rules)
            else:
                rule_f = "(all)"
        with c2:
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        view = matches_df.copy()
        if rule_f != "(all)" and rule_cols:
            view = view[view[rule_cols[0]].astype(str) == rule_f]
        if conf_cols:
            view = view[
                pd.to_numeric(view[conf_cols[0]], errors="coerce").fillna(0) >= min_conf
            ]

        st.dataframe(view, use_container_width=True, hide_index=True, height=420)

        if rule_cols:
            st.subheader("Matches by rule")
            st.bar_chart(view[rule_cols[0]].value_counts())
    else:
        st.warning("matches.csv not found. Run `python -m recon.cli --orders 300 --seed 7` first.")

elif page == "BenchRec External":
    hero(
        "Real data — BenchRec (ICAIF 2023)",
        "External head-to-head on the competition held-out set. Same scoring rule for all systems.",
    )

    st.dataframe(
        pd.DataFrame(FALLBACK_BENCHREC),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("vs external ML", "46× lower FMR", "0.105% vs 4.798%")
    c2.metric("vs own naive", "126× lower FMR", "0.105% vs 13.251%")
    c3.metric("Coverage cost", "47.29%", "vs 68% naive / 62% ML")

    st.markdown(
        """
**Findings**
1. First scoring attempt against `targetAllocation` gave 0/32,048 — column was JSON-wrapped; `A_allocation` is the comparable field.
2. Naive version of *this* engine (no certificates, threshold=0) is **worse** than the external ML baseline.
3. Confidence-gated abstention is the precision lever — certificates are audit, not safety, on measured distributions.
4. Coverage attack (narration tie-break) scored 0% precision on promoted matches → reverted; residual closer to ambiguity floor.
        """
    )
    st.markdown(
        '<div class="ll-ok">Thesis validated externally: refuse rather than guess, '
        "with coverage cost stated.</div>",
        unsafe_allow_html=True,
    )

elif page == "Live / Mutation":
    hero(
        "Live layer",
        "Windowed re-solve · sticky exclusions · override isolation · hash-chained log · replay gate",
    )

    metric_row(
        [
            ("Adversarial engine FMR", "2.50%", "3/120 · 3% ceiling"),
            ("Cash drift", "0 paise", "vs full re-solve"),
            ("Match-set diff", "0", "symmetric"),
            ("Replay", "bit-identical", "6/6 chains"),
        ]
    )

    st.subheader("Invariants enforced every resolve")
    st.markdown(
        """
| Invariant | Enforcement |
|-----------|-------------|
| Closed record set | No prior match half-inside window |
| Override isolation | ID + UTR sticky; three independent checks; `OVERRIDE_LEAK` aborts |
| Cash log ↔ state | `log.cash_position() == state.reconciled_paise()` |
| Claim attribution | Line value moves only when claim count crosses zero |
| Fail-fast mutation | `--fail-fast` stops on first drift / FMR violation |
        """
    )

    st.subheader("Agent cannot create a match")
    st.markdown(
        """
Action space is deliberately narrow: `RETRY_WIDER_WINDOW`, `ACCEPT_EXPLAINED`,
`WRITE_OFF_IMMATERIAL`, `ESCALATE_TO_HUMAN`. Attribution stays with the deterministic engine.
Measured: 27 steps, net exposure change **₹0.00**, harmful actions **0**.
        """
    )

elif page == "Agent":
    hero("Controller agent", "Observe → triage → act → verify → stop. Never invents attribution.")

    st.markdown(
        """
```
observe  →  triage queue by cash at risk
decide   →  highest-value exception with an unused policy
act      →  one of four bounded actions
observe  →  did exposure fall? if not, revert
repeat   →  until nothing remains the agent may act on
```
        """
    )

    st.dataframe(
        pd.DataFrame(
            [
                ("RETRY_WIDER_WINDOW", "Re-run engine with wider date bounds; all gates still apply"),
                ("ACCEPT_EXPLAINED", "Close explained shortfall/duplicate; no money moves"),
                ("WRITE_OFF_IMMATERIAL", "Only if no live attribution and not arithmetic dispute"),
                ("ESCALATE_TO_HUMAN", "Names the artefact a human must fetch"),
            ],
            columns=["Action", "Effect"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="ll-warn">Write-off of a bank credit strands its settlement '
        "(and vice versa). Arithmetic mismatches are disputes about amounts, not "
        "attribution failures — writing them off destroyed a good match (+₹1.99L exposure). "
        "Rule that survived: write-off only for records with no live attribution "
        "and non-arithmetic complaints.</div>",
        unsafe_allow_html=True,
    )

elif page == "Failure Log":
    hero("What broke, and how we got out", "Chronological. Including times the measurement was wrong.")

    if failure_log_path and failure_log_path.is_file():
        text = failure_log_path.read_text(encoding="utf-8", errors="replace")
        st.caption(f"Loaded {failure_log_path} ({len(text):,} chars)")
        q = st.text_input("Filter log (substring)", "")
        lines = text.splitlines()
        if q:
            lines = [ln for ln in lines if q.lower() in ln.lower()]
        st.code("\n".join(lines[:400]) or "(no matching lines)", language="markdown")
        if len(lines) > 400:
            st.caption(f"Showing first 400 of {len(lines)} matching lines.")
    else:
        st.markdown(
            """
**Highlights (see FAILURE_LOG.md in repo for full chronology)**

| # | Finding |
|---|---------|
| 9 | Documented limitation (k≤2) had never been tested — cost 59 pts of recall once realistic consolidations existed |
| 11 | Engine was right; ground truth was wrong (rolling reserves) — corrected labels, not code |
| 23–25 | Domain-mismatched withholding rule on USD BenchRec: 11/14 wrong → disabled → FMR 0.220%→0.134% |
| 26 | Testing contingent-uniqueness claim: certificates redundant with abstention; surfaced latent crash |
| 30–31 | Certificates do not explain BenchRec residual; attribution corrected |
| 32 | 31.6% of held-out bank credits had no exception naming them — audit trail gap fixed |
| 33 | `rule_audit.py` auto-flags harmful rules |
| 34 | Narration tie-break coverage attack: 0% precision → reverted |
            """
        )

elif page == "Architecture":
    hero("Architecture", "Constrained assignment, not classification. Order is load-bearing.")

    st.markdown(
        """
```
INGEST   integer paise · reject loudly, never drop
  │
L1  order ↔ payment   exact id → order ref → Hungarian (amount, date)
  │
L2  arithmetic        fee / GST / net · confidence exactly 1.0 · no AI
  │
L3  settlement ↔ bank
  │   R3.0  duplicate / UTR / near-duplicate clustering
  │   R3.1–3.2  identifier in narration + plausible withholding
  │   R3.3–3.4  Hungarian amount+date · margin confidence
  │   R3.6–3.7  MITM subset-sum · merges≤8 · splits≤5 · ambiguity refuse
  │
ABSTAIN  confidence < threshold → exception, not a guess
ARTIFACTS  run.json · matches.csv · exceptions.csv · close_report.html
```
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Where AI is used")
        st.markdown(
            """
- Narration similarity as **tie-break cost term** only
- Optional LLM re-rank of already-constrained candidates
- **Never** emits a ledger number
- Ablation: deleting the module changes no number to 4 decimals
            """
        )
    with c2:
        st.subheader("Where AI is refused")
        st.markdown(
            """
- All arithmetic (fees, GST, batch totals)
- All exact identifier joins
- Final assignment decision (optimiser + threshold)
- Any path that could invent an id
            """
        )

elif page == "Reproduce":
    hero("Reproduce every claim", "Stranger's machine. No secrets. Offline.")

    st.code(
        """pip install -r requirements.txt   # numpy, scipy
python -m recon.cli --orders 300 --seed 7
python -m pytest tests/ -q
python -m recon.benchrec_eval --data /path/to/benchrec
./run.sh                             # ~12 min full regeneration
streamlit run streamlit_app.py       # this console
""",
        language="bash",
    )

    st.markdown(
        """
| Command | What it proves |
|---------|----------------|
| `recon.cli` | Headline 100% / 0.000% FMR on seed 7 |
| `pytest` | 119 tests including metamorphic + live invariants |
| `mutate.py --fail-fast` | Adversarial residual + zero cash drift |
| `replay.py` | Bit-identical artifacts from event log |
| `benchrec_eval` | External ML head-to-head |
| `rule_audit` | Auto-flag domain-mismatched rules |
| `ai_contribution.py` | AI ablation |
        """
    )

    st.markdown(
        '<div class="ll-ok">Claims are only worth anything if a stranger\'s machine '
        "reproduces them. CI runs reconcile + tests + replay + adversarial on "
        "Python 3.11 and 3.12 with no secrets.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "LedgerLens · Razorpay AI Buildathon Track 04 · "
    "Read-only demo shell · Engine ownership stays in recon/"
)
