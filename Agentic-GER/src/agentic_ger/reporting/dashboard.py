#!/usr/bin/env python3
"""Build a recording-centric case dashboard for one GigaSpeechBench run."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_ger.config import load_prompt_pack
from agentic_ger.trace_stats import read_trace_stats


from agentic_ger.utils import REPO_ROOT


ROOT = REPO_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "runs/cases"


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    run_root: Path


STYLE = r"""
:root{--bg:#f4f6f8;--paper:#fff;--ink:#17202a;--muted:#667085;--line:#dfe4ea;--navy:#15283d;--blue:#1769aa;--green:#08784f;--green-bg:#e4f4ec;--red:#b4233b;--red-bg:#fde9ed;--amber:#8a6200;--amber-bg:#fff3ce}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--blue);text-decoration:none}main{max-width:1580px;margin:auto;padding:28px}.hero{background:linear-gradient(130deg,#132235,#27435f);color:#fff;border-radius:16px;padding:27px 30px}.hero h1{margin:0 0 5px;font-size:29px}.hero .sub{color:#cbd8e5}.topnav{display:flex;gap:15px;margin-bottom:14px}.topnav a{color:#dcecff}.sub,.note{color:var(--muted)}h2{font-size:20px;margin:30px 0 12px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin-top:16px}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px}.hero .card{background:#ffffff10;border-color:#ffffff22}.label{font-size:12px;color:var(--muted)}.hero .label{color:#c8d5e2}.value{font-size:23px;font-weight:760;margin:4px 0}.good{color:var(--green);font-weight:650}.bad{color:var(--red);font-weight:650}.neutral{color:var(--amber)}.hero .good{color:#80e2b7}.hero .bad{color:#ff9daa}.table-wrap{overflow:auto;background:var(--paper);border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf1f5;color:#4c5968;position:sticky;top:0;z-index:1;white-space:nowrap}tbody tr:hover{background:#f8fafc}.num{text-align:right;white-space:nowrap}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 11px}.filters input,.filters select,.filters button,.button{border:1px solid #cfd7e0;border-radius:8px;background:#fff;color:var(--ink);padding:8px 11px;font:inherit}.filters input{min-width:300px;flex:1}.button{cursor:pointer}.button.active{background:var(--navy);color:#fff}.tag{display:inline-block;border-radius:99px;padding:2px 8px;background:#e9edf2}.tag.complete,.tag.improved,.tag.accept{background:var(--green-bg);color:var(--green)}.tag.failed,.tag.worsened{background:var(--red-bg);color:var(--red)}.tag.neutral,.tag.keep{background:var(--amber-bg);color:var(--amber)}.audio-bar{position:sticky;top:0;z-index:4;background:#f4f6f8ef;border:1px solid var(--line);border-radius:12px;padding:11px 14px;margin:16px 0}.audio-bar audio{width:100%;height:38px}.comparison th:first-child,.comparison td:first-child{position:sticky;left:0;background:inherit}.transcript mark{background:#ffe6a8}.reference{color:#634c19}.decision-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.before,.after,.asr,.reason,.error-box,pre{border-radius:8px;padding:10px 12px;white-space:pre-wrap;overflow-wrap:anywhere}.before{background:#fff0f2;color:#7e1f30}.after{background:#eaf7f0;color:#075f3d}.asr{background:#eef5fb}.reason{background:#f4f0ff}.error-box{background:#fff0f2;border:1px solid #f2bbc4;color:#7e1f30}details{background:#fff;border:1px solid var(--line);border-radius:9px;margin:8px 0;padding:9px 12px}summary{cursor:pointer;font-weight:650}.caveat{background:#fff8e6;border:1px solid #ebd28c;border-radius:11px;padding:13px 16px;margin:20px 0}@media(max-width:760px){main{padding:14px}.hero{padding:20px}.filters input{min-width:100%}.cards{grid-template-columns:1fr}}
"""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve(strict=False) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(target)


EXTRA_STYLE = r"""
.dataset-nav{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}.dataset-nav a{background:#ffffff16;border:1px solid #ffffff2b;border-radius:99px;padding:6px 12px;color:#fff}
.input-table{table-layout:fixed}.input-table th:first-child,.input-table td:first-child{width:125px}.input-table td{line-height:1.65}.input-table button,.context-play{border:0;background:#e5edf5;color:#164d74;border-radius:6px;padding:4px 7px;cursor:pointer;font:inherit}
.path-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:11px}.path-card{background:#fff;border:1px solid var(--line);border-radius:11px;padding:13px 15px;scroll-margin-top:70px}.path-card.worsened{border-left:5px solid var(--red)}.path-card.improved{border-left:5px solid var(--green)}.path-card.neutral{border-left:5px solid var(--amber)}
.path-text{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:8px}.schema-warning{background:#fff0f2;border:1px solid #f2bbc4;color:#8c2437;border-radius:7px;padding:8px 10px;margin:8px 0}.player-status{margin-left:8px}.section-note{margin:-5px 0 12px}.loading{padding:28px;text-align:center;color:var(--muted)}.transcript{min-width:1050px}.transcript th:first-child,.transcript td:first-child{width:125px}.transcript th:not(:first-child),.transcript td:not(:first-child){width:30%}
.gate-meta{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0}.gate-meta .tag{font-size:.78rem}.kept-path{margin-top:12px}.kept-path .path-card{border-left:5px solid #7890a4}
@media(max-width:760px){.path-grid{grid-template-columns:1fr}.path-text{grid-template-columns:1fr}}
"""


INDEX_JS = r"""
const query=document.querySelector('#query'),dataset=document.querySelector('#dataset'),domain=document.querySelector('#domain'),outcome=document.querySelector('#outcome');
const rows=[...document.querySelectorAll('#records tbody tr')];
function apply(){const text=query.value.trim().toLowerCase();for(const row of rows)row.hidden=!!(text&&!row.textContent.toLowerCase().includes(text))||!!(dataset.value&&row.dataset.dataset!==dataset.value)||!!(domain.value&&row.dataset.domain!==domain.value)||!!(outcome.value&&row.dataset.outcome!==outcome.value);document.querySelector('#shown').textContent=rows.filter(row=>!row.hidden).length;}
[query,dataset,domain,outcome].forEach(node=>node.addEventListener('input',apply));
for(const head of document.querySelectorAll('th[data-sort]'))head.addEventListener('click',()=>{const key=head.dataset.sort,number=head.dataset.number==='1',ascending=head.dataset.ascending!=='1';head.dataset.ascending=ascending?'1':'0';rows.sort((a,b)=>{const x=a.dataset[key]||'',y=b.dataset[key]||'';return(number?Number(x)-Number(y):x.localeCompare(y))*(ascending?1:-1);});const body=document.querySelector('#records tbody');for(const row of rows)body.appendChild(row);});
apply();
"""


CASE_JS = r"""
const page=document.body.dataset.case;
const audio=document.querySelector('#audio');
let baselineById=new Map();
let caseData=null;
function esc(value){return String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));}
function textOf(segment,final=false){const keys=final?['text_final','text_working','text_original','text']:['text_original','text_working','text_final','text'];for(const key of keys)if(segment?.[key])return String(segment[key]);return '';}
function clipUrl(data,segment){const start=Math.max(0,Number(segment.start||0)-.25),end=Number(segment.end||0)+.35;return `../../clip/audio/${data.dataset}/${data.audio_name}?start=${start.toFixed(3)}&end=${end.toFixed(3)}`;}
async function playSegment(data,segmentId){const segment=baselineById.get(Number(segmentId));if(!segment)return;audio.src=clipUrl(data,segment);audio.load();document.querySelector('#player-status').textContent=`#${segmentId} · ${Number(segment.start).toFixed(1)}–${Number(segment.end).toFixed(1)}s`;try{await audio.play();}catch(error){document.querySelector('#player-status').textContent+=` · 播放失败：${error.message}`;}}
document.addEventListener('click',event=>{const button=event.target.closest('[data-play]');if(button&&caseData)playSegment(caseData,button.dataset.play);});
function outcome(delta){return delta<0?'improved':delta>0?'worsened':'neutral';}
function renderInput(data,segments){document.querySelector('#input-body').innerHTML=segments.map(segment=>`<tr><td><button data-play="${Number(segment.id)}">▶ #${Number(segment.id)}</button><br><small>${Number(segment.start||0).toFixed(1)}–${Number(segment.end||0).toFixed(1)}s</small></td><td>${esc(textOf(segment))}</td></tr>`).join('');}
function gateMeta(row){const bits=[];if(row.evidence_source)bits.push(`<span class="tag neutral">证据：${esc(row.evidence_source)}</span>`);if(typeof row.baseline_spoken_form_valid==='boolean')bits.push(`<span class="tag ${row.baseline_spoken_form_valid?'keep':'accept'}">口语基线有效：${row.baseline_spoken_form_valid?'是':'否'}</span>`);return bits.length?`<div class="gate-meta">${bits.join('')}</div>`:'';}
function renderPath(data){const html=data.patches.map((patch,index)=>{const kind=outcome(patch.delta),warning=patch.schema_risk?`<div class="schema-warning"><strong>Schema 上限：</strong>Before ${patch.before.length} 字符，超过 edited_segment.maxLength=${patch.schema_limit}。</div>`:'';return `<article id="patch-${index}" class="path-card ${kind}"><div class="decision-head"><button class="context-play" data-play="${patch.segment_id}">▶ #${patch.segment_id}</button><span class="tag ${kind}">${kind}</span><strong class="${patch.delta<0?'good':patch.delta>0?'bad':'neutral'}">Reference ${esc(data.error_unit)} errors ${patch.delta>0?'+':''}${patch.delta}</strong><span class="note">第 ${index+1} 次接受修改</span></div>${warning}${gateMeta(patch)}<div class="path-text"><div><div class="label">Before</div><div class="before">${esc(patch.before)}</div></div><div><div class="label">After</div><div class="after">${esc(patch.after)}</div></div></div><div class="asr"><strong>Qwen3-ASR 重听：</strong>${esc(patch.asr_text||'（无）')}</div><div class="reason"><strong>Agent 理由：</strong>${esc(patch.reason||'（无）')}</div><div class="note"><strong>Reference：</strong>${esc(patch.reference)}</div></article>`;}).join('');document.querySelector('#path').innerHTML=html||'<div class="card note">该录音没有接受任何修改，final 与 baseline 相同。</div>';}
function renderKeeps(data){const rows=data.kept_decisions||[];document.querySelector('#kept-count').textContent=rows.length;document.querySelector('#kept-path').innerHTML=rows.map((row,index)=>`<article class="path-card"><div class="decision-head"><button class="context-play" data-play="${row.segment_id}">▶ #${row.segment_id}</button><span class="tag keep">keep</span><span class="note">候选：${esc(row.focus||'（未记录）')}</span></div>${gateMeta(row)}<div class="before">${esc(row.before)}</div><div class="asr"><strong>Qwen3-ASR 重听：</strong>${esc(row.asr_text||'（无）')}</div><div class="reason"><strong>保留理由：</strong>${esc(row.reason||'（无）')}</div></article>`).join('')||'<div class="card note">该录音没有被检查后保留的候选。</div>';}
function renderOutput(baseline,finalRecord,referenceRecord){const finals=new Map((finalRecord.segments||[]).map(row=>[Number(row.id),row])),refs=new Map((referenceRecord.segments||[]).map(row=>[Number(row.id),row]));const changed=new Set((baseline.segments||[]).filter(row=>textOf(row)!==textOf(finals.get(Number(row.id)),true)).map(row=>Number(row.id)));const context=new Set([...changed].flatMap(id=>[id-1,id,id+1]));document.querySelector('#output-body').innerHTML=(baseline.segments||[]).map(segment=>{const id=Number(segment.id),before=textOf(segment),after=textOf(finals.get(id),true),reference=textOf(refs.get(id),true),visible=context.has(id);return `<tr class="${changed.has(id)?'changed':'context'}" data-visible="${visible?1:0}"><td><button class="context-play" data-play="${id}">▶ #${id}</button><br><small>${Number(segment.start||0).toFixed(1)}–${Number(segment.end||0).toFixed(1)}s</small></td><td>${esc(before)}</td><td>${esc(after)}</td><td class="reference">${esc(reference)}</td></tr>`;}).join('');document.querySelector('#changed-count').textContent=changed.size;segmentMode(changed.size?'changed':'all');}
const segmentButtons=[...document.querySelectorAll('[data-segments]')];
function segmentMode(mode){for(const row of document.querySelectorAll('#output-body tr'))row.hidden=mode==='changed'&&row.dataset.visible!=='1';for(const button of segmentButtons)button.classList.toggle('active',button.dataset.segments===mode);}
for(const button of segmentButtons)button.addEventListener('click',()=>segmentMode(button.dataset.segments));
async function init(){const data=await fetch(page).then(response=>{if(!response.ok)throw new Error(`case data HTTP ${response.status}`);return response.json();});caseData=data;const baseline=await fetch(data.baseline_url).then(response=>{if(!response.ok)throw new Error(`baseline HTTP ${response.status}`);return response.json();});baselineById=new Map((baseline.segments||[]).map(row=>[Number(row.id),row]));renderInput(data,baseline.segments||[]);renderPath(data);renderKeeps(data);document.querySelector('#loading').hidden=true;const target=location.hash.match(/patch-(\d+)/);if(target)document.querySelector(location.hash)?.scrollIntoView();try{const [finalRecord,referenceRecord]=await Promise.all([fetch(data.final_url).then(response=>{if(!response.ok)throw new Error(`final HTTP ${response.status}`);return response.json();}),fetch(data.reference_url).then(response=>{if(!response.ok)throw new Error(`reference HTTP ${response.status}`);return response.json();})]);renderOutput(baseline,finalRecord,referenceRecord);}catch(error){document.querySelector('#output-body').innerHTML=`<tr><td colspan="4"><div class="error-box">纠正结果载入失败，但输入音频与纠正路径仍可使用：${esc(error)}</div></td></tr>`;}}
init().catch(error=>{document.querySelector('#loading').innerHTML=`<div class="error-box"><strong>页面数据载入失败</strong><pre>${esc(error.stack||error)}</pre></div>`;});
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def write_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(link, target)


def edit_max_length(config: dict[str, Any]) -> int | None:
    prompt_pack_path = config.get("prompt_pack_path")
    if prompt_pack_path:
        path = Path(prompt_pack_path)
        if not path.is_dir():
            return None
        schema = load_prompt_pack(path).check.schema
        value = schema.get("properties", {}).get("edited_segment", {}).get(
            "maxLength"
        )
        return int(value) if isinstance(value, int) else None
    # Runs created before prompt packs used the frozen 250-character schema.
    return 250


def collect_records(
    output_root: Path, datasets: tuple[Dataset, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        config = load_json(dataset.run_root / "run_config.json")
        base_config = config.get("source_run_config") or config
        schema_limit = edit_max_length(base_config)
        summary = load_json(dataset.run_root / "evaluation/summary.json")
        batch = load_json(dataset.run_root / "batch_summary.json")
        statuses = {
            str(row["run_id"]): row for row in batch.get("results") or []
        }
        summaries.append(
            {"dataset": dataset, "summary": summary, "config": config}
        )
        manifest = [
            json.loads(line)
            for line in (dataset.run_root / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        patch_rows = load_json(
            dataset.run_root / "evaluation/patch_outcomes.json"
        )["records"]
        patches_by_id: dict[str, list[dict[str, Any]]] = {}
        for patch in patch_rows:
            run_id = str(patch["run_id"])
            before = str(patch["old_text"])
            patches_by_id.setdefault(run_id, []).append(
                {
                    "segment_id": int(patch["segment_id"]),
                    "before": before,
                    "after": str(patch["new_text"]),
                    "reference": str(patch["reference"]),
                    "reason": str(patch.get("reason") or ""),
                    "asr_text": str(patch.get("asr_text") or ""),
                    "delta": int(patch["delta"]),
                    "outcome": str(patch["outcome"]),
                    "schema_limit": schema_limit,
                    "schema_risk": (
                        schema_limit is not None and len(before) > schema_limit
                    ),
                }
            )
        data_root = Path(base_config["data_root"])
        baseline_system = str(
            base_config.get("baseline_system") or "FunASR-Realtime"
        )
        for manifest_index, item in enumerate(manifest):
            run_id = str(item["run_id"])
            domain = run_id.split("#", 1)[0]
            status_row = statuses.get(run_id, {})
            status = str(status_row.get("runner_status") or "missing")
            patches = [dict(row) for row in patches_by_id.get(run_id, [])]
            filename = f"{safe_name(run_id)}.json"
            audio_name = f"{safe_name(run_id)}.wav"
            output = dataset.run_root / run_id / "workspace/output"
            decisions = read_trace_stats(output / "trace.jsonl").decisions
            accepted = [row for row in decisions if row["decision"] == "accept"]
            if len(accepted) == len(patches):
                for patch, decision in zip(patches, accepted, strict=True):
                    if patch["segment_id"] != decision["segment_id"]:
                        raise ValueError(
                            f"accepted decision order differs for {run_id}"
                        )
                    patch["evidence_source"] = decision["evidence_source"]
                    patch["baseline_spoken_form_valid"] = decision[
                        "baseline_spoken_form_valid"
                    ]
            kept_decisions = [
                row for row in decisions if row["decision"] == "keep"
            ]
            final_source = output / "final_transcript.json"
            if status != "complete" or not final_source.exists():
                final_source = Path(item["baseline"])
            reference_source = (
                data_root
                / domain
                / baseline_system
                / "ref"
                / f"{run_id}.json"
            )
            write_link(
                output_root / "audio" / dataset.key / audio_name,
                Path(item["audio"]),
            )
            write_link(
                output_root / "transcripts" / dataset.key / "baseline" / filename,
                Path(item["baseline"]),
            )
            write_link(
                output_root / "transcripts" / dataset.key / "final" / filename,
                final_source,
            )
            write_link(
                output_root / "transcripts" / dataset.key / "reference" / filename,
                reference_source,
            )
            case_data = {
                "dataset": dataset.key,
                "dataset_label": dataset.label,
                "error_unit": "characters" if domain.endswith("-CH") else "words",
                "run_id": run_id,
                "domain": domain,
                "status": status,
                "audio_name": audio_name,
                "baseline_url": f"../../transcripts/{dataset.key}/baseline/{filename}",
                "final_url": f"../../transcripts/{dataset.key}/final/{filename}",
                "reference_url": f"../../transcripts/{dataset.key}/reference/{filename}",
                "patches": patches,
                "kept_decisions": kept_decisions,
            }
            data_path = output_root / "data" / dataset.key / filename
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(
                json.dumps(case_data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            records.append(
                {
                    **case_data,
                    "dataset_object": dataset,
                    "manifest_index": manifest_index,
                    "net_delta": sum(int(row["delta"]) for row in patches),
                    "improved": sum(row["outcome"] == "improved" for row in patches),
                    "worsened": sum(row["outcome"] == "worsened" for row in patches),
                    "neutral": sum(row["outcome"] == "neutral" for row in patches),
                    "kept": len(kept_decisions),
                    "schema_risks": sum(row["schema_risk"] for row in patches),
                    "runner": status_row,
                }
            )
    return records, summaries


def summary_cards(summaries: list[dict[str, Any]]) -> str:
    cards = []
    for item in summaries:
        dataset, summary, config = (
            item["dataset"],
            item["summary"],
            item["config"],
        )
        before = summary["aggregate"]["baseline"]
        after = summary["aggregate"]["final"]
        metric_label = str(summary.get("metric_label") or "CER")
        rate_key = "macro_wer" if metric_label == "WER" else "macro_cer"
        relative = 100.0 * (
            float(after[rate_key]) - float(before[rate_key])
        ) / float(before[rate_key])
        bwer = (
            f"B-WER {float(before['macro_bwer']):.4f}% → "
            f"{float(after['macro_bwer']):.4f}%"
            if int(before["biased_reference_tokens"])
            else "无实体标签，B-WER 不可用"
        )
        prompt_pack_id = config.get("prompt_pack_id")
        cards.append(
            "<div class='card'>"
            f"<div class='label'>{esc(dataset.label)}</div>"
            f"<div class='value'>{metric_label} {float(before[rate_key]):.4f}% → "
            f"{float(after[rate_key]):.4f}%</div>"
            f"<div class='{'good' if relative <= 0 else 'bad'}'>"
            f"{metric_label} 相对变化 {relative:+.2f}% · errors "
            f"{int(after['errors']) - int(before['errors']):+d}</div>"
            f"<div>{esc(bwer)}</div>"
            f"<div class='note'>Prompt pack: "
            f"{esc(prompt_pack_id or 'legacy embedded')}</div></div>"
        )
    return "".join(cards)


def page_path(row: dict[str, Any]) -> str:
    return f"pages/{row['dataset']}/{safe_name(row['run_id'])}.html"


def render_index(
    records: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> str:
    summary = summaries[0]["summary"]
    language_label = str(summary.get("language_label") or "中文")
    domains = sorted({row["domain"] for row in records})
    domain_options = "".join(
        f"<option value='{esc(domain)}'>{esc(domain)}</option>" for domain in domains
    )
    dataset_options = "".join(
        f"<option value='{esc(item['dataset'].key)}'>"
        f"{esc(item['dataset'].label)}</option>"
        for item in summaries
    )
    rows = []
    for row in records:
        net_delta = int(row["net_delta"])
        record_outcome = (
            "failed"
            if row["status"] != "complete"
            else "improved"
            if net_delta < 0
            else "worsened"
            if net_delta > 0
            else "neutral"
        )
        rows.append(
            f"<tr data-dataset='{esc(row['dataset'])}' "
            f"data-domain='{esc(row['domain'])}' data-outcome='{record_outcome}' "
            f"data-run='{esc(row['run_id'])}' data-delta='{net_delta}'>"
            f"<td><a href='{esc(page_path(row))}'>{esc(row['run_id'])}</a></td>"
            f"<td>{esc(row['dataset'])}</td><td>{esc(row['domain'])}</td>"
            f"<td><span class='tag {record_outcome}'>{record_outcome}</span></td>"
            f"<td class='num'>{len(row['patches'])}</td>"
            f"<td class='num'>{int(row['kept'])}</td>"
            f"<td class='num good'>{int(row['improved'])}</td>"
            f"<td class='num bad'>{int(row['worsened'])}</td>"
            f"<td class='num'>{int(row['neutral'])}</td>"
            f"<td class='num {'good' if net_delta < 0 else 'bad' if net_delta > 0 else 'neutral'}'>{net_delta:+d}</td>"
            f"<td class='num {'bad' if row['schema_risks'] else ''}'>{int(row['schema_risks'])}</td></tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(language_label)} ASR Case Dashboard</title><link rel="stylesheet" href="assets/style.css"></head><body><main>
<section class="hero"><h1>ASR Agent · {esc(language_label)} Case Dashboard</h1><div class="sub">{len(records)} 条录音 · 每页保持“输入集 → 纠正路径 → 纠正结果”</div><div class="dataset-nav"><a href="#records-section">进入逐录音列表</a></div><div class="cards">{summary_cards(summaries)}</div></section>
<div class="caveat">每张卡片对应各自冻结的 run_config 和 prompt-pack 指纹；Reference 只在页面中用于离线观察。失败录音按 baseline 不变展示，不重跑。</div>
<h2 id="records-section">逐录音</h2><div class="filters"><input id="query" placeholder="搜索录音 ID 或领域"><select id="dataset"><option value="">全部数据集</option>{dataset_options}</select><select id="domain"><option value="">全部领域</option>{domain_options}</select><select id="outcome"><option value="">全部结果</option><option value="improved">改善</option><option value="worsened">变差</option><option value="neutral">不变</option><option value="failed">失败</option></select><span class="note">显示 <strong id="shown">{len(records)}</strong> 条；点击表头排序</span></div>
<div class="table-wrap"><table id="records"><thead><tr><th data-sort="run">录音</th><th>数据集</th><th>领域</th><th>状态</th><th>接受修改</th><th>检查后保留</th><th>改善</th><th>变差</th><th>不变</th><th data-sort="delta" data-number="1">离线 errors Δ</th><th>Schema 上限</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<script src="assets/index.js"></script></main></body></html>"""


def render_case(
    position: int,
    records: list[dict[str, Any]],
) -> str:
    row = records[position]
    dataset = row["dataset_object"]
    same_dataset = [
        item for item in records if item["dataset"] == row["dataset"]
    ]
    local_position = same_dataset.index(row)
    previous = (
        f"<a href='{safe_name(same_dataset[local_position - 1]['run_id'])}.html'>"
        "← 上一条</a>"
        if local_position
        else ""
    )
    next_link = (
        f"<a href='{safe_name(same_dataset[local_position + 1]['run_id'])}.html'>"
        "下一条 →</a>"
        if local_position + 1 < len(same_dataset)
        else ""
    )
    net_delta = int(row["net_delta"])
    failure = ""
    if row["status"] != "complete":
        error = row["runner"].get("error") or row["runner"].get("console_tail") or ""
        failure = (
            "<div class='error-box'><strong>运行失败，final 按 baseline 不变展示"
            f"</strong><pre>{esc(error)}</pre></div>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(row['run_id'])} · Case</title><link rel="stylesheet" href="../../assets/style.css"></head><body data-case="../../data/{esc(row['dataset'])}/{safe_name(row['run_id'])}.json"><main>
<section class="hero"><div class="topnav"><a href="../../index.html">← 总览</a>{previous}{next_link}</div><h1>{esc(row['run_id'])}</h1><div class="sub">{esc(dataset.label)} · {esc(row['status'])} · 接受 {len(row['patches'])} / 保留 {int(row['kept'])} · 离线 {esc(row['error_unit'])} errors {net_delta:+d}</div></section>
<div id="loading" class="card loading">正在载入 timestamped baseline…</div><div id="case-content">
<h2>1. 输入集：音频 + timestamped baseline</h2><p class="note section-note">点击任意 ▶，服务端即时截取该 timestamp 对应的小 WAV；不再通过 VS Code 转发整条 40–90 分钟 WAV。<a href="../../audio/{esc(row['dataset'])}/{esc(row['audio_name'])}">打开完整 WAV</a></p>
<div class="audio-bar"><audio id="audio" controls preload="none"></audio><span id="player-status" class="note player-status">尚未选择片段</span></div>
{failure}<div class="table-wrap"><table class="input-table"><thead><tr><th>时间</th><th>Baseline transcript</th></tr></thead><tbody id="input-body"><tr><td colspan="2" class="loading">载入 baseline…</td></tr></tbody></table></div>
<h2>2. 纠正路径</h2><p class="note section-note">按运行顺序展示每次被接受的修改：输入文本、Qwen3-ASR 重听证据、证据门判断、Agent 理由与本轮输出。Reference 只用于离线标注 outcome。</p><div id="path" class="path-grid"><div class="card loading">载入纠正路径…</div></div>
<details><summary>检查后保留的候选（<span id="kept-count">0</span>）</summary><p class="note">这些候选经过重听但没有写回，可直接观察口语保护门挡下的 ASR 跟随、语法规范化与表面格式修改。</p><div id="kept-path" class="path-grid kept-path"></div></details>
<h2>3. 纠正结果</h2><p class="note section-note">同一 segment ID 和 timestamp 下并排展示 baseline、最终输出与 Reference。差异片段数量：<strong id="changed-count">0</strong>。</p><div class="filters"><button class="button active" data-segments="changed">修改片段及上下文</button><button class="button" data-segments="all">全部片段</button></div><div class="table-wrap"><table class="transcript comparison"><thead><tr><th>时间</th><th>Baseline</th><th>Final</th><th>Reference</th></tr></thead><tbody id="output-body"><tr><td colspan="4" class="loading">载入 final/reference…</td></tr></tbody></table></div>
</div><script src="../../assets/case.js"></script></main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--language", choices=("CH", "EN"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.run_root.resolve() / "run_config.json")
    datasets = (
        Dataset(
            "run",
            "Vertical-Domain · "
            f"{config.get('baseline_system') or 'FunASR-Realtime'} · "
            f"{args.language}",
            args.run_root.resolve(),
        ),
    )
    output_root = args.output_root.resolve()
    (output_root / "assets").mkdir(parents=True, exist_ok=True)
    records, summaries = collect_records(output_root, datasets)
    (output_root / "assets/style.css").write_text(
        STYLE.strip() + "\n" + EXTRA_STYLE.strip() + "\n", encoding="utf-8"
    )
    (output_root / "assets/index.js").write_text(
        INDEX_JS.strip() + "\n", encoding="utf-8"
    )
    (output_root / "assets/case.js").write_text(
        CASE_JS.strip() + "\n", encoding="utf-8"
    )
    (output_root / "index.html").write_text(
        render_index(records, summaries), encoding="utf-8"
    )
    for position, row in enumerate(records):
        page = output_root / page_path(row)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_case(position, records), encoding="utf-8")
    stale_cases = output_root / "assets/cases.json"
    stale_cases.unlink(missing_ok=True)
    print(f"recordings: {len(records)}")
    print(f"dashboard: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
