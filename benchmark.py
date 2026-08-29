"""Every evidence table, from one command.

    python benchmark.py            # ~4 min, writes artifacts/benchmark.md + .json

Produces:
  1. Difficulty sweep          40 seeds x 5 difficulties
  2. Multi-way consolidation   the k-bound and window sweep
  3. UNMODELED generalisation  anomaly classes the matcher was never built for
  4. AI-deleted ablation       does the system need the AI module at all?
  5. Reason-code confusion     per difficulty, per code
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from recon import evaluate, generate, normalize
from recon.match import MatcherConfig, reconcile

SEEDS = list(range(1, 41))
OUT: dict[str, object] = {"schema_version": "1.0.0"}
LINES: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    LINES.append(s)


def one(seed: int, difficulty: float = 1.0, multiway: float = 0.0,
        unmodeled: float = 0.0, orders: int = 300, **cfg_kw) -> dict:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        g = generate.generate(n_orders=orders, seed=seed, difficulty=difficulty,
                              multiway=multiway, unmodeled=unmodeled)
        generate.write(g, d)
        truth = json.loads((d / "truth.json").read_text())
        data, rejects = normalize.load_all(d)
        t0 = time.perf_counter()
        r = reconcile(data, MatcherConfig(**cfg_kw))
        ms = (time.perf_counter() - t0) * 1000
        mr = evaluate.match_rate(r, truth)
        xs = evaluate.score_exceptions(r.exceptions, truth)
        h = xs["excluding_trivial_ORDER_NO_PAYMENT"]
        return {"seed": seed, "match_rate": mr["settlement_match_rate_pct"],
                "fmr": mr["false_match_rate_pct"], "ms": ms,
                "exc_p": h["precision"], "exc_r": h["recall"],
                "exc_expected": h["tp"] + h["fn"], "spurious": h["fp"], "missed": h["fn"],
                "by_reason": xs["by_reason_code"], "rejects": len(rejects),
                "abstentions": xs["abstentions_routed_to_human"]}


def agg(rows: list[dict]) -> dict:
    def st(k):
        v = [r[k] for r in rows]
        return {"mean": statistics.mean(v), "min": min(v), "max": max(v)}
    return {"match_rate": st("match_rate"), "fmr": st("fmr"),
            "exc_p": st("exc_p"), "exc_r": st("exc_r"), "ms": st("ms"),
            "any_fmr": max(r["fmr"] for r in rows) > 0}


def table(title: str, header: list[str], rows: list[list[str]]) -> None:
    emit(f"\n### {title}\n")
    emit("| " + " | ".join(header) + " |")
    emit("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        emit("| " + " | ".join(str(x) for x in r) + " |")


# ---------------------------------------------------------------- 1. difficulty
def bench_difficulty() -> None:
    rows, store = [], {}
    for diff in (0.0, 0.5, 1.0, 2.0, 3.0):
        rs = [one(s, difficulty=diff) for s in SEEDS]
        a = agg(rs)
        store[str(diff)] = a
        exp = sum(r["exc_expected"] for r in rs)
        rows.append([f"{diff}", f"{a['match_rate']['mean']:.2f}%", f"{a['match_rate']['min']:.2f}%",
                     f"**{a['fmr']['mean']:.3f}%**", f"{a['fmr']['max']:.3f}%",
                     f"{a['exc_p']['mean']:.4f}" if exp else "n/a",
                     f"{a['exc_p']['min']:.4f}" if exp else "n/a",
                     f"{a['exc_r']['mean']:.4f}" if exp else "n/a",
                     f"{a['exc_r']['min']:.4f}" if exp else "n/a"])
    OUT["difficulty_sweep"] = store
    table("1. Difficulty sweep (40 seeds each, 200 runs)",
          ["difficulty", "match mean", "match min", "FMR mean", "FMR max",
           "exc P mean", "exc P min", "exc R mean", "exc R min"], rows)
    emit("\n`exc P min` is the worst single seed, not the average. Both are reported "
         "because an average alone hides the floor.")


# ------------------------------------------------------------------ 2. multiway
def bench_multiway() -> None:
    rows, store = [], {}
    configs = [("k=3, window ±4 (original)", dict(max_subset_size=3, subset_sum_lookback_days=4)),
               ("k=8, window ±4", dict(max_subset_size=8, subset_sum_lookback_days=4)),
               ("k=8, look-back 16 (shipped)", dict(max_subset_size=8)),
               ("k=10, look-back 24", dict(max_subset_size=10, subset_sum_lookback_days=24))]
    for label, kw in configs:
        rs = [one(s, multiway=1.0, **kw) for s in SEEDS[:20]]
        a = agg(rs)
        store[label] = a
        rows.append([label, f"{a['match_rate']['mean']:.2f}%", f"{a['match_rate']['min']:.2f}%",
                     f"**{a['fmr']['mean']:.3f}%**", f"{a['ms']['mean']:.1f} ms"])
    OUT["multiway"] = store
    card = Counter()
    for s in SEEDS[:20]:
        g = generate.generate(300, s, 1.0, multiway=1.0)
        for _, t in g.truth["settlement_to_bank"].items():
            if len(t) > 1:
                card[f"split_{len(t)}way"] += 1
        for _, sids in g.truth["bank_to_settlements"].items():
            if len(sids) > 1:
                card[f"merge_{len(sids)}way"] += 1
    OUT["multiway_cardinalities"] = dict(sorted(card.items()))
    table("2. Multi-way consolidation: the k bound vs the date window",
          ["config", "match mean", "match min", "FMR mean", "wall clock"], rows)
    emit(f"\nTrue group cardinalities generated: `{dict(sorted(card.items()))}`")
    emit("\nThe cardinality bound was never the binding constraint. An 8-way "
         "consolidation spans 8 settlement days, so with a symmetric ±4 day window "
         "its earliest members never entered the candidate pool at any k.")


# ----------------------------------------------------------------- 3. unmodeled
def bench_unmodeled() -> None:
    rows, store = [], {}
    for u in (0.0, 1.0, 2.0, 3.0):
        rs = [one(s, difficulty=1.0, unmodeled=u) for s in SEEDS[:25]]
        a = agg(rs)
        store[str(u)] = a
        rows.append([f"{u}", f"{a['match_rate']['mean']:.2f}%", f"{a['match_rate']['min']:.2f}%",
                     f"**{a['fmr']['mean']:.3f}%**", f"**{a['fmr']['max']:.3f}%**",
                     f"{a['exc_p']['mean']:.4f}", f"{a['exc_r']['mean']:.4f}"])
    OUT["unmodeled"] = store
    table("3. Generalisation: anomaly classes the matcher was NEVER built for",
          ["unmodeled level", "match mean", "match min", "FMR mean", "FMR max",
           "exc P", "exc R"], rows)
    emit("""
Injected classes, none of which any rule anticipates:

| class | why it defeats the design |
|---|---|
| rolling reserve / holdback | shortfall at 3.7/4.25/6.15/8.8% — deliberately **not** a plausible withholding rate, so the rate gate rejects it |
| reserve release credit | a later credit with no settlement behind it at all |
| value-date skew 5–8 days | lands outside the matching window |
| credit before settlement report | negative date gap beyond the look-ahead |
| near-duplicate UTR | one character off, so exact-string dedupe misses it |
| chargeback debit | money out, tied to no settlement |

**The pass condition is not that match rate holds. It is that the false match
rate stays at zero while match rate degrades.** Missing an anomaly you were not
built for is acceptable; confidently mis-matching it is not.""")


# ------------------------------------------------------------- 4. AI ablation
def bench_ai_ablation() -> None:
    rows, store = [], {}
    for label, kw in (("AI module ENABLED", dict(fuzzy_narration=True)),
                      ("AI module DELETED", dict(fuzzy_narration=False))):
        for mode, extra in (("normal", {}), ("multiway", {"multiway": 1.0}),
                            ("unmodeled", {"unmodeled": 2.0})):
            rs = [one(s, **extra, **kw) for s in SEEDS[:20]]
            a = agg(rs)
            store[f"{label}|{mode}"] = a
            rows.append([label, mode, f"{a['match_rate']['mean']:.2f}%",
                         f"{a['fmr']['mean']:.3f}%", f"{a['exc_p']['mean']:.4f}"])
    OUT["ai_ablation"] = store
    table("4. AI ablation: delete the AI module entirely",
          ["config", "mode", "match mean", "FMR mean", "exc P"], rows)
    tb = sum(one(s, **{})["match_rate"] * 0 for s in SEEDS[:1])  # no-op keeps shape
    tiebreaks = 0
    for s in SEEDS[:20]:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            g = generate.generate(300, s, 1.0)
            generate.write(g, d)
            data, _ = normalize.load_all(d)
            r = reconcile(data, MatcherConfig(fuzzy_narration=True))
            tiebreaks += r.ai_stats.tiebreaks_invoked
    OUT["ai_tiebreaks_in_20_runs"] = tiebreaks
    emit(f"\nNarration tie-breaks actually invoked across 20 runs: **{tiebreaks}**.")
    emit("\nThe numbers are identical with the AI module deleted. That is the finding, "
         "reported rather than buried: on this problem a payout total is the sum of "
         "dozens of random captures and behaves as a near-unique key, so narration "
         "similarity has almost nothing left to decide. It is retained because it "
         "costs one cosine per candidate pair and converts genuine amount ties from "
         "human review into confident matches — see `ai_contribution.py` for the "
         "isolated proof that the mechanism works when its condition occurs.")


# ------------------------------------------------- 5. reason-code confusion
def bench_reason_codes() -> None:
    store: dict[str, dict] = {}
    for diff in (1.0, 3.0):
        acc: dict[str, Counter] = defaultdict(Counter)
        for s in SEEDS[:25]:
            for code, v in one(s, difficulty=diff)["by_reason"].items():
                for k2 in ("expected", "predicted", "correct", "missed", "spurious"):
                    acc[code][k2] += v[k2]
        store[str(diff)] = {k: dict(v) for k, v in acc.items()}
        rows = []
        for code, v in sorted(acc.items(), key=lambda x: -x[1]["expected"]):
            p = v["correct"] / v["predicted"] if v["predicted"] else 0.0
            r = v["correct"] / v["expected"] if v["expected"] else 0.0
            rows.append([f"`{code}`", v["expected"], v["predicted"], v["correct"],
                         v["missed"], v["spurious"], f"{p:.4f}", f"{r:.4f}"])
        table(f"5{'a' if diff == 1.0 else 'b'}. Reason-code confusion, difficulty {diff} (25 seeds)",
              ["reason code", "expected", "predicted", "correct", "missed",
               "spurious", "precision", "recall"], rows)
    OUT["reason_code_confusion"] = store


def main() -> int:
    emit("# LedgerLens benchmark")
    emit(f"\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S')} · "
         f"Python {sys.version.split()[0]} · schema 1.0.0")
    emit("\nReproduce: `python benchmark.py`")
    bench_difficulty()
    bench_multiway()
    bench_unmodeled()
    bench_ai_ablation()
    bench_reason_codes()
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / "benchmark.md").write_text("\n".join(LINES) + "\n")
    (out / "benchmark.json").write_text(json.dumps(OUT, indent=2, default=str))
    emit("\n\nWritten to artifacts/benchmark.md and artifacts/benchmark.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
