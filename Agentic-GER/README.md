# GigaSpeechBench ASR Correction

A small-file pipeline for vertical-domain Chinese and English ASR correction.
Qwen3.8-27B reads the transcript, locates suspicious spans, requests acoustic
evidence from Qwen3-ASR-1.7B, and decides whether to keep or edit each segment.
References and entity labels are used **only in offline evaluation**.

**Release candidate, not yet approved for public distribution.** License and
third-party authorization checks remain in [distribution prerequisites](docs/reproduction.md#distribution-prerequisites).
This repository includes no dataset, model weights, private experiment results,
credentials, or imported Git history.

## Default reproduction: two experiments, not a matrix

| Language | Initial ASR | Auxiliary ASR | Correction model | Mode |
|---|---|---|---|---|
| Chinese (CH) | FunASR-Realtime | Qwen3-ASR-1.7B | Qwen3.8-27B | non-thinking |
| English (EN) | FunASR-Realtime | Qwen3-ASR-1.7B | Qwen3.8-27B | non-thinking |

The quick acceptance path runs **one complete recording per language**. It checks
the full processing path, not full-dataset quality. The full language commands
below process 524 Chinese / 387 English recordings, with 12 domains per language.
Neither path launches a multi-model matrix.

```text
HF metadata + published FunASR hypotheses + audio archives
  → prepare_data.py: validate IDs/times; separate hypothesis/reference files
  → src/agentic_ger/experiment.py: summary → scan → relisten → check → edit (bounded loops)
  → src/agentic_ger/evaluation/evaluate_zh.py / src/agentic_ger/evaluation/evaluate_en.py: reference-based offline evaluation
```

## 1. Prerequisites and uv environment

Run all commands from this repository root. Use Python 3.12 on Linux/macOS,
network access to Hugging Face, and `uv` ([installation](https://docs.astral.sh/uv/getting-started/installation/)).
The default prepared audio links require a filesystem supporting symlinks.

```bash
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-data.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
```

- `requirements-metrics.txt`: frozen metric adapters and normalization dependencies.
- `requirements.txt`: correction client/orchestration + metrics; no GPU runtime.
- `requirements-data.txt`: Hugging Face downloader, not needed once data is prepared.
- For metrics only: install `requirements-metrics.txt`, then install the local
  package with `uv pip install --python .venv/bin/python --no-deps -e .`.

The editable install exposes the `agentic_ger` modules and the optional
`agentic-ger` command. Keep this source checkout: `configs/`, `prompts/` and `runs/`
are repository resources, not bundled wheel data. Use the documented source
installation rather than copying a wheel to a machine without the checkout.

`requirements.lock` additionally freezes the full resolved environment with package
hashes. To reproduce that tested dependency resolution, use
`uv pip install --python .venv/bin/python --require-hashes -r requirements.lock`,
followed by the same `--no-deps -e .` package installation.
The smaller requirements files document dependency ownership and support metrics-only use.

vLLM, CUDA, PyTorch and model deployment are **external prerequisites**, not part
of this environment. You must already have compatible Qwen3.8-27B and
Qwen3-ASR-1.7B services. See [service contracts](docs/SERVICES.md).

## 2. Get and organize GigaSpeechBench

Official source: [speechcolab/GigaSpeechBench on Hugging Face](https://huggingface.co/datasets/speechcolab/GigaSpeechBench).
We use its [Vertical-Domain files](https://huggingface.co/datasets/speechcolab/GigaSpeechBench/tree/680d3057641b7507a1ef14974407c7b0a7964e64/Vertical-Domain),
pinned to revision `680d3057641b7507a1ef14974407c7b0a7964e64`.
Consult the source's access and usage terms before downloading or redistributing
any material. A public download is not itself a grant to redistribute it.

The raw snapshot is:

```text
gsb_data/hf/Vertical-Domain/
├── data/
│   ├── AGR-CH/{metadata.json,audio.tar.gz}
│   ├── AGR-EN/{metadata.json,audio.tar.gz}
│   └── ... 12 CH + 12 EN domains
└── results/FunASR-Realtime.json
```

For the quick path, download all metadata and FunASR hypotheses, but only the
AGR-CH and AGR-EN audio archives. No ASR baseline generation is needed. Archives
are downloaded per domain, not per recording; this can still take substantial
bandwidth and disk space. Full Vertical-Domain audio is roughly tens of GB;
allow space for both compressed and extracted audio.

Keep data outside the repository. The path below is an example sibling directory,
not a dependency on any pre-existing installation:

```bash
export GSB_DATA_DIR="$(pwd)/../gsb_data"
.venv/bin/python tools/prepare_data.py \
  --snapshot "$GSB_DATA_DIR/hf" \
  --output "$GSB_DATA_DIR/prepared/Vertical-Domain" \
  --download --audio-domains AGR-CH AGR-EN
```

The tool extracts WAV files safely, joins on recording ID + SID + exact decimal
time bounds (never list position), and checks complete 524/387 recording coverage.
It does not normalize or modify the published hypotheses. It produces:

```text
gsb_data/prepared/Vertical-Domain/
├── PREPARATION.<hash>.json       # source hashes, coverage, alignment and audio availability
├── AGR-CH/FunASR-Realtime/
│   ├── hyp/<recording-id>.json   # inference text + IDs/timestamps, no references/entities
│   ├── ref/<recording-id>.json   # offline evaluation only
│   └── audio/<recording-id>.wav  # relative link to extracted public audio
└── ...
```

Metadata-only preparation omits `--audio-domains`; it suffices for the baseline
metric checks, **not inference**. Partial-audio preparation is explicit in the
manifest. Selected recordings must have audio; unavailable recordings are never
silently omitted from a full run. Keep the snapshot and prepared trees together,
or regenerate links when moving them. A changed input is rejected rather than
overwriting a previous prepared file; use a new output directory for a new revision.

Verify the frozen baseline metric adapters (no model calls):

```bash
.venv/bin/python -m agentic_ger.evaluation.verify_zh --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain"
.venv/bin/python -m agentic_ger.evaluation.verify_en --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain"
```

Expected terminal messages: `PASS: 524/524` and `PASS: 387/387`. Exact integer
error-count checks must pass; rounded metric agreement alone is insufficient.

## 3. Check external services

Default correction endpoint: `http://127.0.0.1:8002/v1/chat/completions`, model ID
`qwen3.8-27b`. Default auxiliary ASR endpoint:
`http://127.0.0.1:7879/v1/chat/completions`, model ID `Qwen3-ASR-1.7B`.
Both must expose `/v1/models`; correction must additionally expose `/tokenize`
and support the specified structured output and sampling fields.

```bash
.venv/bin/python tools/check_services.py configs/zh.json
.venv/bin/python tools/check_services.py configs/en.json
```

For different endpoints, pass `--llm-url` / `--asr-url` to both service checks and
experiment commands. For different served model IDs, copy a config to an ignored
`configs/*.local.json` and edit `models`. Optional authentication uses environment
variables `CORRECTION_API_KEY` and `ASR_API_KEY`; never put credentials in URLs or
committed configs. No credentials are required for unauthenticated local services.

## 4. Quick end-to-end run: Chinese + English non-thinking

These are the **default acceptance commands**. Each processes one full recording
in the downloaded domain, then evaluates it and generates local HTML reports.
No prompt or loop budget is shortened for the smoke.

```bash
.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain" \
  --domain AGR-CH --limit 1 --no-thinking --run-root runs/quick_zh

.venv/bin/python -m agentic_ger.experiment configs/en.json \
  --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain" \
  --domain AGR-EN --limit 1 --no-thinking --run-root runs/quick_en
```

First add `--dry-run` to inspect commands without model calls. A recording can
still require many calls; expect minutes or longer depending on service load,
output length and number of suspects. Two completed commands alone are not a
pass: inspect `batch_summary.json` for `complete=1, failed=0`, the final transcript,
and `evaluation/summary.json`. A failed recording can have a valid baseline-scored
report, which does not demonstrate a successful correction run.

To test reporting independently after a terminal run:

```bash
.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain" \
  --run-root runs/quick_zh --report-only
```

## 5. Optional: full cohorts for the same two experiments

First download all audio; the downloader reuses already downloaded files:

```bash
.venv/bin/python tools/prepare_data.py \
  --snapshot "$GSB_DATA_DIR/hf" --output "$GSB_DATA_DIR/prepared/Vertical-Domain" \
  --download --all-audio

.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain" --no-thinking
.venv/bin/python -m agentic_ger.experiment configs/en.json \
  --data-root "$GSB_DATA_DIR/prepared/Vertical-Domain" --no-thinking
```

The default run root is revisioned from configuration, prompt and core-code hashes.
Use a fresh run root after any scientific change. Completed recordings are retained;
terminal failures are not automatically retried under the default `retry_rounds=0`.
Requests have bounded recovery (three LLM attempts with schema feedback/limited
JSON repair); this is distinct from restarting a failed recording.

Advanced experiments may use `--thinking` or `--baseline-system Whisper-Large-v3`
after preparing that published baseline with `--baselines FunASR-Realtime Whisper-Large-v3`.
A language × baseline × mode matrix is optional, manually configured work, **not
the default workflow or acceptance target**. No local/remote model-comparison
launcher or private Local correction results are included.

## Outputs and evaluation

| File | Purpose |
|---|---|
| `effective_experiment.json`, `run_config.json` | resolved settings and fingerprints |
| `manifest.jsonl`, `batch_summary.json` | selected cohort and real completion status |
| `repro/` | code, prompt, evaluator and configuration snapshots |
| per-recording trace + `final_transcript.json` | decisions, evidence and final hypothesis |
| `evaluation/summary.json`, `summary.md` | baseline/final metrics and fallback accounting |
| `evaluation/report.html`, `cases/index.html` | private, offline diagnostic reports |

Primary metric: Chinese **B-CER** (serialized as `b_wer` / `macro_bwer` for historical
adapter compatibility), English **B-WER**. Auxiliary metrics: Chinese CER and
English WER. Full cohorts use 12-domain equal-weight macro averages; micro totals
are retained. A quick subset is averaged only over its included domains and must
not be represented as the 12-domain result. Failed recordings contribute their
unchanged baseline, with true statuses retained. See [evaluation details](docs/EVALUATION.md).

Reports and traces contain transcripts, references, audio links and local paths.
Keep them private. Do not publicly expose the report server or commit `runs/`.
For local playback, run `.venv/bin/python -m agentic_ger.reporting.server --directory
runs/quick_zh/cases --bind 127.0.0.1 --port 12398` and open
`http://127.0.0.1:12398`. Use an SSH tunnel for remote machines, not a public bind.

## Code map

```text
src/agentic_ger/
├── experiment.py                   public experiment/report entry point
├── agent.py                        single-recording correction flow
├── config.py                       prompt/schema loading
├── runner.py                       internal orchestration and frozen snapshots
├── utils.py                        service authentication and checkout paths
├── trace_stats.py                  trace aggregation
├── evaluation/                     metrics_zh/en, evaluate_zh/en, verify_zh/en
└── reporting/                      dashboard, server, visualize
configs/{zh,en}.json                independent language configurations
prompts/{zh,en}/                    frozen language prompts and schemas
vendor/                            byte-identical upstream normalizers
tools/prepare_data.py               public HF → validated recording files
tools/check_services.py             model/context/tokenizer preflight
tools/check_release.py              tracked-file hygiene and provenance check
tests/                             offline unit and integration tests
docs/reproduction.md               clean-room reproduction procedure
docs/SERVICES.md                    external service contracts
docs/EVALUATION.md                  metrics and reproducibility boundaries
```

For independent-agent acceptance, follow [reproduction.md](docs/reproduction.md).
Python modules use `_zh` / `_en`; official dataset IDs retain their upstream
`CH` / `EN` suffixes so alignment and metric denominators remain unchanged.

## Dependency inventory

Observed from installed distribution metadata in the fresh Python 3.12 environment.
`requirements.lock` records exact versions and distribution hashes. This inventory
is not a substitute for reviewing license text, transitive notices or compatibility.
No third-party wheel is copied into the source repository.

| Distribution | Version | Declared license metadata |
|---|---|---|
| anyio | 4.15.1 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.1 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| filelock | 3.32.6 | MIT |
| fsspec | 2026.7.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface-hub | 1.31.0 | Apache-2.0 |
| idna | 3.19 | BSD-3-Clause |
| indic-numtowords | 1.1.0 | MIT |
| kaldialign | 0.12.0 | Apache-2.0 |
| more-itertools | 11.1.0 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| text2num | 3.1.0 | MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.7.0 | MIT |
| whisper-normalizer | 0.1.15 | MIT |
| zhconv | 1.4.3 | GPLv2+ — compatibility review required |

`pip` is installed as environment tooling by `uv venv --seed`, not an application
dependency. Model-serving distributions and weights are outside this environment;
their licensing remains the deployer's responsibility.
