#!/usr/bin/env python3
"""Create one dependency-free HTML report from evaluation JSON values."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _number(value: float) -> str:
    return f"{value:.4f}%"


def write_report(
    summary: dict[str, Any], patch_stats: dict[str, Any], path: Path
) -> None:
    before = summary["aggregate"]["baseline"]
    after = summary["aggregate"]["final"]
    metric_label = str(summary.get("metric_label") or "CER")
    language_label = str(summary.get("language_label") or "中文")
    rate_key = "macro_wer" if metric_label == "WER" else "macro_cer"
    domain_rows = "".join(
        "<tr>"
        f"<td>{_escape(row['domain'])}</td>"
        f"<td>{_number(row['baseline']['rate'])}</td>"
        f"<td>{_number(row['final']['rate'])}</td>"
        f"<td>{row['final']['rate'] - row['baseline']['rate']:+.4f}</td>"
        f"<td>{_number(row['baseline']['hotword']['b_wer'])}</td>"
        f"<td>{_number(row['final']['hotword']['b_wer'])}</td>"
        f"<td>{row['final']['hotword']['b_wer'] - row['baseline']['hotword']['b_wer']:+.4f}</td>"
        "</tr>"
        for row in summary["domains"]
    )
    patch_rows = "".join(
        "<tr "
        f"data-outcome=\"{_escape(row['outcome'])}\" "
        f"data-domain=\"{_escape(row['domain'])}\">"
        f"<td>{_escape(row['domain'])}</td>"
        f"<td>{_escape(row['run_id'])}</td>"
        f"<td>{int(row['segment_id'])}</td>"
        f"<td class=old>{_escape(row['old_text'])}</td>"
        f"<td class=new>{_escape(row['new_text'])}</td>"
        f"<td><span class=\"tag {_escape(row['outcome'])}\">{_escape(row['outcome'])}</span></td>"
        f"<td>{int(row['delta']):+d}</td>"
        f"<td>{_escape(row.get('asr_text'))}</td>"
        f"<td>{_escape(row.get('reason'))}</td>"
        f"<td>{_escape(row.get('reference'))}</td>"
        "</tr>"
        for row in patch_stats["records"]
    )
    domains = sorted({row["domain"] for row in patch_stats["records"]})
    domain_options = "".join(
        f"<option value=\"{_escape(domain)}\">{_escape(domain)}</option>"
        for domain in domains
    )
    runtime = summary["runtime"]
    payload = json.dumps(
        {"summary": summary, "patches": patch_stats}, ensure_ascii=False
    ).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(language_label)} ASR Agent 评测</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#1d2633;--muted:#687386;--line:#dfe5ee;--blue:#315bea;--green:#16794b;--red:#b52d3a;--amber:#8b6508}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1480px;margin:auto;padding:28px}} h1{{margin:0 0 6px;font-size:27px}} h2{{margin:30px 0 12px;font-size:19px}}
.sub{{color:var(--muted);margin-bottom:22px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}} .label{{color:var(--muted);font-size:12px}} .value{{font-size:24px;font-weight:700;margin:4px 0}} .delta{{font-variant-numeric:tabular-nums}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}} table{{width:100%;border-collapse:collapse}}
th,td{{border-bottom:1px solid var(--line);padding:9px 11px;text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#eef2f8;white-space:nowrap}} tbody tr:last-child td{{border-bottom:0}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}} input,select{{border:1px solid var(--line);border-radius:7px;background:white;padding:8px 10px}}
input{{min-width:300px;flex:1}} .old{{color:var(--red)}} .new{{color:var(--green)}} .tag{{border-radius:99px;padding:2px 7px;background:#e9edf4}} .tag.improved{{color:var(--green);background:#e2f5ec}} .tag.worsened{{color:var(--red);background:#fde9eb}} .tag.neutral{{color:var(--amber);background:#f7f0d9}}
#patches td:nth-child(8),#patches td:nth-child(9),#patches td:nth-child(10){{min-width:260px;max-width:520px}}
@media(max-width:700px){{main{{padding:16px}} input{{min-width:100%}}}}
</style></head><body><main>
<h1>{_escape(language_label)} ASR Agent 评测</h1><div class=sub>{summary['recordings']} 条录音，{len(summary['domains'])} 个领域；所有数值来自跑完后的离线评测。</div>
<section class=cards>
<div class=card><div class=label>{_escape(metric_label)}（领域 macro）</div><div class=value>{_number(before[rate_key])} → {_number(after[rate_key])}</div><div class=delta>{after[rate_key]-before[rate_key]:+.4f} 个百分点</div></div>
<div class=card><div class=label>B-WER（领域 macro）</div><div class=value>{_number(before['macro_bwer'])} → {_number(after['macro_bwer'])}</div><div class=delta>{after['macro_bwer']-before['macro_bwer']:+.4f} 个百分点</div></div>
<div class=card><div class=label>接受的修改</div><div class=value>{patch_stats['accepted']}</div><div>改善 {patch_stats['improved']} · 变差 {patch_stats['worsened']} · 不变 {patch_stats['neutral']}</div></div>
<div class=card><div class=label>运行量</div><div class=value>{runtime['audio_hours']:.1f} 小时音频</div><div>LLM {runtime['llm_requests']} 次 · ASR {runtime['asr_requests']} 次</div></div>
</section>
<h2>各领域</h2><div class=table-wrap><table><thead><tr><th>领域</th><th>原 {_escape(metric_label)}</th><th>新 {_escape(metric_label)}</th><th>变化</th><th>原 B-WER</th><th>新 B-WER</th><th>变化</th></tr></thead><tbody>{domain_rows}</tbody></table></div>
<h2>逐项修改</h2><div class=filters><input id=search placeholder="搜索录音、原文、新文、ASR 或理由"><select id=outcome><option value="">全部结果</option><option value=improved>改善</option><option value=worsened>变差</option><option value=neutral>不变</option></select><select id=domain><option value="">全部领域</option>{domain_options}</select></div>
<div class=table-wrap><table id=patches><thead><tr><th>领域</th><th>录音</th><th>片段</th><th>原文</th><th>修改后</th><th>结果</th><th>距离变化</th><th>重听 ASR</th><th>理由</th><th>Reference</th></tr></thead><tbody>{patch_rows}</tbody></table></div>
<script type="application/json" id="report-data">{payload}</script>
<script>
const q=document.querySelector('#search'),o=document.querySelector('#outcome'),d=document.querySelector('#domain'),rows=[...document.querySelectorAll('#patches tbody tr')];
function filter(){{const text=q.value.trim().toLowerCase();for(const row of rows)row.hidden=!!((o.value&&row.dataset.outcome!==o.value)||(d.value&&row.dataset.domain!==d.value)||(text&&!row.textContent.toLowerCase().includes(text)));}}
q.addEventListener('input',filter);o.addEventListener('change',filter);d.addEventListener('change',filter);
</script></main></body></html>"""
    path.write_text(document, encoding="utf-8")
