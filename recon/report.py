"""Artifacts. Machine-readable first, human-readable second.

No build step, no npm, no CDN. The HTML report is a single self-contained file
written from the run's own JSON, so the dashboard cannot drift from the numbers
it claims to display — a real failure mode when a frontend keeps its own copy of
the metrics.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .money import fmt


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_exceptions_csv(path: Path, exceptions) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["severity", "cash_impact_inr", "entity_type", "entity_id", "reason_code",
                    "detail", "confidence", "suggested_action", "blocking_data",
                    "candidates_considered"])
        for e in exceptions:
            w.writerow([e.severity, f"{e.cash_impact_paise/100:.2f}", e.entity_type,
                        e.entity_id, e.reason.value, e.detail,
                        f"{e.confidence:.2f}", e.suggested_action,
                        json.dumps(e.blocking_data, default=str),
                        json.dumps(e.candidates_considered, default=str)])


def write_matches_csv(path: Path, matches) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "left_id", "right_ids", "rule", "confidence", "evidence", "rejected"])
        for m in matches:
            w.writerow([m.layer, m.left_id, "|".join(m.right_ids), m.rule,
                        f"{m.confidence:.4f}", json.dumps(m.evidence, default=str),
                        json.dumps(m.rejected, default=str)])


_CSS = """
:root{--bg:#0b0e14;--panel:#131822;--line:#1f2733;--txt:#e6edf3;--dim:#8b98a9;
--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;padding:28px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:.5px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1.4px;color:var(--dim);
margin:30px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--dim);font-size:12px;margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:13px}
.k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.9px}
.v{font-size:21px;margin-top:5px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--dim);font-weight:500;font-size:10.5px;
text-transform:uppercase;letter-spacing:.9px;padding:7px 9px;border-bottom:1px solid var(--line)}
td{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
tr:hover td{background:#161c27}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10.5px;border:1px solid}
.s-hi{color:var(--bad);border-color:#5c2323;background:#2a1414}
.s-md{color:var(--warn);border-color:#5c4a1e;background:#251f10}
.s-lo{color:var(--dim);border-color:var(--line);background:#161c27}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.dim{color:var(--dim)}
code{color:var(--acc);font-size:11px}
details{margin-top:4px}summary{cursor:pointer;color:var(--acc);font-size:11px}
pre{background:#0d1117;border:1px solid var(--line);border-radius:4px;padding:9px;
overflow-x:auto;font-size:11px;color:var(--dim);margin:5px 0 0}
"""


def _sev_class(s: int) -> str:
    return "s-hi" if s >= 80 else ("s-md" if s >= 50 else "s-lo")


def write_html(path: Path, payload: dict[str, Any], exceptions, matches) -> None:
    e = html.escape
    st, mr = payload["throughput"], payload["match_rate"]
    ms, xs = payload["match_scores"], payload["exception_scores"]
    l3 = ms["L3_settlement_bank"]

    cards = [
        ("Settlement match rate", f"{mr['settlement_match_rate_pct']}%", "ok"),
        ("False match rate", f"{mr['false_match_rate_pct']}%",
         "ok" if mr["false_match_rate_pct"] == 0 else "bad"),
        ("L3 precision", f"{l3['precision']:.4f}", "ok" if l3["precision"] > .98 else "warn"),
        ("L3 recall", f"{l3['recall']:.4f}", "ok" if l3["recall"] > .9 else "warn"),
        ("Records ingested", f"{st['records_ingested']:,}", ""),
        ("Records / second", f"{st['records_per_second']:,}", ""),
        ("Wall clock", f"{st['wall_clock_s']}s", ""),
        ("Open exceptions", f"{len(exceptions)}", "warn"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{e(k)}</div>'
        f'<div class="v {c}">{e(str(v))}</div></div>' for k, v, c in cards)

    tier_rows = "".join(
        f"<tr><td>{e(k)}</td><td>{v['truth_pairs']}</td><td>{v['tp']}</td>"
        f"<td class='bad'>{v['fp']}</td><td class='warn'>{v['fn']}</td>"
        f"<td>{v['precision']:.4f}</td><td>{v['recall']:.4f}</td></tr>"
        for k, v in ms["L3_by_difficulty_tier"].items())

    curve_rows = "".join(
        f"<tr><td>{r['threshold']:.2f}</td><td>{r['asserted']}</td>"
        f"<td>{r['coverage']*100:.2f}%</td><td>{r['precision']:.5f}</td>"
        f"<td class=\"{'ok' if r['false_match_rate']==0 else 'bad'}\">{r['false_match_rate']*100:.3f}%</td>"
        f"<td>{r['sent_to_human']}</td></tr>" for r in payload["abstention_curve"])

    rule_rows = "".join(
        f"<tr><td><code>{e(k)}</code></td><td>{v}</td></tr>"
        for k, v in sorted(st["by_rule"].items(), key=lambda x: -x[1]))

    reason_rows = "".join(
        f"<tr><td><code>{e(k)}</code></td><td>{v['expected']}</td><td>{v['predicted']}</td>"
        f"<td class='ok'>{v['correct']}</td><td class='warn'>{v['missed']}</td>"
        f"<td class='bad'>{v['spurious']}</td></tr>"
        for k, v in sorted(xs["by_reason_code"].items()))

    exc_rows = "".join(
        f"<tr><td><span class='tag {_sev_class(x.severity)}'>{x.severity}</span></td>"
        f"<td>{e(fmt(x.cash_impact_paise)) if x.cash_impact_paise else '<span class=dim>—</span>'}</td>"
        f"<td>{e(x.entity_type)}</td><td><code>{e(x.entity_id)}</code></td>"
        f"<td><code>{e(x.reason.value)}</code></td><td>{e(x.detail)}"
        f"<details><summary>evidence</summary><pre>{e(json.dumps(x.blocking_data, indent=2, default=str))}</pre>"
        + (f"<pre>candidates: {e(json.dumps(x.candidates_considered, indent=2, default=str))}</pre>"
           if x.candidates_considered else "")
        + f"</details></td><td>{x.confidence:.2f}</td><td class='dim'>{e(x.suggested_action)}</td></tr>"
        for x in exceptions[:120])

    abl = "".join(
        f"<tr><td>{e(r['config'])}</td><td class='dim'>{e(r['layers'])}</td>"
        f"<td>{r['settlement_match_rate_pct']:.2f}%</td>"
        f"<td class=\"{'ok' if r['false_match_rate_pct']==0 else 'bad'}\">{r['false_match_rate_pct']:.3f}%</td>"
        f"<td>{r['exceptions']}</td></tr>" for r in payload.get("ablation", []))

    ai = payload["ai"]
    doc = f"""<!doctype html><meta charset="utf-8"><title>LedgerLens close report</title>
<style>{_CSS}</style>
<h1>LedgerLens — finance close report</h1>
<div class="sub">run <code>{e(str(payload['run_id']))}</code> · seed {payload['seed']} ·
dataset {e(str(payload['dataset']))} · generated {e(str(payload['generated_at']))}</div>
<div class="grid">{card_html}</div>

<h2>Reconciled value</h2>
<div class="grid">
<div class="card"><div class="k">Gross captured</div><div class="v">{fmt(payload['value']['gross_captured'])}</div></div>
<div class="card"><div class="k">Payouts declared</div><div class="v">{fmt(payload['value']['payout_declared'])}</div></div>
<div class="card"><div class="k">Bank credits seen</div><div class="v">{fmt(payload['value']['bank_credits'])}</div></div>
<div class="card"><div class="k">Unreconciled exposure</div><div class="v bad">{fmt(payload['value']['unreconciled_exposure'])}</div></div>
</div>

<h2>Accuracy by difficulty tier <span class="dim">— one average would hide this</span></h2>
<table><tr><th>tier</th><th>truth pairs</th><th>tp</th><th>fp</th><th>fn</th><th>precision</th><th>recall</th></tr>{tier_rows}</table>

<h2>Abstention curve <span class="dim">— automation level vs. accepted risk</span></h2>
<table><tr><th>threshold</th><th>asserted</th><th>coverage</th><th>precision</th><th>false match rate</th><th>to human</th></tr>{curve_rows}</table>

<h2>Ablation <span class="dim">— what each layer actually contributes</span></h2>
<table><tr><th>config</th><th>layers enabled</th><th>match rate</th><th>false match rate</th><th>exceptions</th></tr>{abl}</table>

<h2>Matches by rule</h2>
<table><tr><th>rule</th><th>count</th></tr>{rule_rows}</table>

<h2>Exception detection by reason code</h2>
<table><tr><th>reason</th><th>expected</th><th>predicted</th><th>correct</th><th>missed</th><th>spurious</th></tr>{reason_rows}</table>

<h2>AI layer</h2>
<div class="card"><div class="k">backend</div><div class="v" style="font-size:15px">{e(ai['backend'])}</div>
<div class="dim" style="margin-top:8px">similarity scores computed: {ai['similarity_scores_computed']} ·
<b>tie-breaks actually decided by narration: {ai['tiebreaks_invoked']}</b> ·
llm calls: {ai['llm_calls']} · llm errors: {ai['llm_errors']}</div>
<div class="dim" style="margin-top:6px">A tie-break count of zero means the amount+date key was
already unique on this data. That is reported, not hidden.</div>
{"".join(f'<div class="dim" style="margin-top:6px">note: {e(n)}</div>' for n in ai['notes'])}</div>

<h2>Exception ledger <span class="dim">— {len(exceptions)} open, highest severity first</span></h2>
<table><tr><th>sev</th><th>cash at risk</th><th>type</th><th>entity</th><th>reason</th><th>detail</th><th>conf</th><th>suggested action</th></tr>{exc_rows}</table>
<div class="sub" style="margin-top:18px">Every row above is reproducible with the command in README.md.
Nothing on this page is hand-entered; it is rendered from run.json.</div>
"""
    path.write_text(doc)
