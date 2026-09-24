# Agentic-GER

Agentic-GER is an LLM-based agent for terminology recovery in long-form
speech. It uses the full transcript as global context to propose suspicious
terms, re-transcribes the selected segments with an ASR model, and applies a
minimal edit only when the evidence is sufficient. Accepted edits and keep
decisions remain in working memory for later rounds. Segment boundaries are
not changed.

The paper evaluates four LLMs, two initial ASR systems, two languages, and
both thinking modes. This repository contains the implementation, prompts,
evaluation adapters, and two tracked default configurations:

| Language | Initial ASR | Correction LLM | Re-transcription ASR | Mode |
|---|---|---|---|---|
| Chinese | FunASR-Realtime | Qwen3.8-27B | Qwen3-ASR-1.7B | without thinking |
| English | FunASR-Realtime | Qwen3.8-27B | Qwen3-ASR-1.7B | without thinking |

Other LLMs require a copied local configuration with the served model ID and
endpoint. Other paper dimensions can be selected with `--thinking` and
`--baseline-system`; for the Whisper baseline, prepare both published systems
with `--baselines FunASR-Realtime Whisper-Large-v3`. The repository does not
include datasets, model weights, credentials, raw per-recording run results,
or a multi-model experiment launcher.

## Method

For each timestamped input transcript, the agent:

1. summarizes topic, domain, and recurring terminology;
2. proposes suspicious spans using global context and working memory;
3. re-transcribes the corresponding audio segments;
4. retains or minimally edits each span based on re-transcription and context;
5. repeats until no new candidate is found or a budget is reached.

The tracked budgets are 4 candidates per scan for Chinese and 8 for English,
with at most 32 rounds and 48 accepted edits per audio file. References and
entity annotations are used only by offline evaluation.

## Paper Results

These are aggregate results from the paper's full evaluation. The quick-start
commands below exercise the processing path and do not reproduce these numbers.
Error rates are percentages and lower is better; parentheses show relative error
reduction from the baseline. Full-cohort scores are 12-domain macro averages.

### Without Thinking

| System | Chinese B-CER (FunASR) | Chinese B-CER (Whisper) | English B-WER (FunASR) | English B-WER (Whisper) |
|---|---:|---:|---:|---:|
| Baseline | 10.91 | 35.07 | 12.94 | 14.67 |
| + Qwen3.8-27B | 10.19 (6.7) | 28.88 (17.7) | 12.35 (4.6) | 14.08 (4.0) |
| + Gemma4-31B | 9.88 (9.5) | 25.09 (28.5) | 12.05 (6.9) | 13.71 (6.5) |
| + Qwen3.8-Flash | 9.64 (11.7) | 26.61 (24.1) | 11.91 (8.0) | 13.72 (6.5) |
| + Qwen3.8-Max | **8.96** (17.9) | **22.16** (36.8) | **11.74** (9.3) | **13.64** (7.0) |

### With Thinking

| System | Chinese B-CER (FunASR) | Chinese B-CER (Whisper) | English B-WER (FunASR) | English B-WER (Whisper) |
|---|---:|---:|---:|---:|
| + Qwen3.8-27B | 9.03 | 25.89 | 11.96 | 13.87 |
| + Gemma4-31B | 9.32 | 23.20 | 11.79 | 13.60 |
| + Qwen3.8-Flash | 9.61 | 26.75 | 11.78 | 13.63 |
| + Qwen3.8-Max | 9.45 | 25.54 | 11.79 | 13.62 |

Relative to the without-thinking results, thinking changes the relative error
reduction as follows (percentage points; positive is improvement):

| System | FunASR Chinese | Whisper Chinese | FunASR English | Whisper English |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | +10.6 | +8.5 | +3.0 | +1.5 |
| Gemma4-31B | +5.1 | +5.4 | +2.0 | +0.8 |
| Qwen3.8-Flash | +0.3 | −0.4 | +0.9 | +0.6 |
| Qwen3.8-Max | −4.5 | −9.7 | −0.4 | +0.1 |

## Setup

Commands must run from this directory. Python 3.12, `uv`, and network access to
Hugging Face are required. The default prepared audio layout uses symlinks.

```bash
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-data.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
```

The application does not install vLLM, CUDA, or PyTorch; model services are
external prerequisites. See [service contracts](docs/SERVICES.md). If dependency
resolution fails on a supported platform, open an issue with the Python, uv, OS,
and CPU architecture versions.

The editable install uses this checkout's `configs/` and `prompts/`; keep the
checkout in place. Dependencies are declared in the requirements files, with
matching package extras for correction and data preparation. To record installed
package versions for a run:

```bash
mkdir -p runs
uv pip freeze --python .venv/bin/python > runs/environment.txt
```

## Data

The data source is
[speechcolab/GigaSpeechBench](https://huggingface.co/datasets/speechcolab/GigaSpeechBench),
pinned to revision
`680d3057641b7507a1ef14974407c7b0a7964e64`. The benchmark contains 524
Chinese and 387 English recordings across 12 domains per language. Download and
review its current access and usage terms; a public download does not grant
redistribution rights.

The quick path downloads all metadata and FunASR hypotheses, but only the
AGR-CH and AGR-EN audio archives. Full runs use all domains and require
`--all-audio`. Downloaded and prepared data stay under `data/`, which is
ignored by Git.

```bash
.venv/bin/python tools/prepare_data.py \
  --snapshot data/gigaspeechbench/hf \
  --output data/gigaspeechbench/prepared/Vertical-Domain \
  --download --audio-domains AGR-CH AGR-EN

.venv/bin/python -m agentic_ger.evaluation.verify_zh \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain
.venv/bin/python -m agentic_ger.evaluation.verify_en \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain
```

Both verifiers must print `PASS: 524/524` and `PASS: 387/387`, respectively.
They compare exact insertion, deletion, and substitution counts with the frozen
benchmark adapters; rounded metric agreement is not sufficient.

## Services

Default endpoints are:

| Role | Endpoint | Model ID |
|---|---|---|
| Correction LLM | `http://127.0.0.1:8002/v1/chat/completions` | `qwen3.8-27b` |
| Re-transcription ASR | `http://127.0.0.1:7879/v1/chat/completions` | `Qwen3-ASR-1.7B` |

```bash
.venv/bin/python tools/check_services.py configs/zh.json
.venv/bin/python tools/check_services.py configs/en.json
```

Override endpoints with `--llm-url` and `--asr-url`. Optional authentication uses
the `CORRECTION_API_KEY` and `ASR_API_KEY` environment variables; credentials
must not be committed.

## Run

The quick path processes one complete recording per language, then evaluates it
and generates local reports. It tests execution, not full-benchmark quality.

```bash
.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain \
  --domain AGR-CH --limit 1 --no-thinking --run-root runs/quick_zh

.venv/bin/python -m agentic_ger.experiment configs/en.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain \
  --domain AGR-EN --limit 1 --no-thinking --run-root runs/quick_en
```

Add `--dry-run` before an actual run to inspect the resolved commands. A
successful command requires `complete=1` and `failed=0` in
`batch_summary.json`; a failed recording can still receive a valid baseline
score, so completion status must be checked separately.

### Full evaluation without thinking

Download all audio, then run the two full FunASR-baseline cohorts (524 Chinese
and 387 English recordings). These commands do not launch the full paper matrix:

```bash
.venv/bin/python tools/prepare_data.py \
  --snapshot data/gigaspeechbench/hf \
  --output data/gigaspeechbench/prepared/Vertical-Domain \
  --download --all-audio

.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain --no-thinking
.venv/bin/python -m agentic_ger.experiment configs/en.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain --no-thinking
```

### Full evaluation with thinking

Use the same prepared data. Thinking enables `reasoning_effort=medium`, a
32,768-token output budget per call, and the thinking sampling profile in each
config. **It substantially increases generation cost and runtime.**

```bash
.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain --thinking
.venv/bin/python -m agentic_ger.experiment configs/en.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain --thinking
```

Without `--run-root`, each mode gets a separate path derived from its configuration,
prompt and code hashes. Completed recordings are skipped on resumption; failed
recordings are not retried by default. Use a fresh run root after scientific changes.

### Runtime and token usage

Historical Qwen3.8-27B runs with Qwen3-ASR on our H100 80 GB host provide a rough
cost reference for the full FunASR cohorts (132.9 Chinese / 136.3 English audio
hours). Serving concurrency and code/prompt revisions differed; these are not a
controlled speed comparison or a timing guarantee for the current release.

| Language | Mode | Batch time | Recording RTF | LLM input / output tokens (M) |
|---|---|---:|---:|---:|
| Chinese | Without thinking | 1.1 h | 0.078 | 36.35 / 0.76 |
| English | Without thinking | 2.1 h | 0.208 | 79.43 / 2.02 |
| Chinese | With thinking | 11.4 h | 1.213 | 69.08 / 21.89* |
| English | With thinking | 15.7 h | 1.939 | 84.79 / 26.70* |

The two languages took about **3.3 hours combined without thinking**, versus
**27 hours with thinking** in these runs. Budget several hours versus a day or
more on a comparable deployment, excluding downloads and model loading.

Recording RTF is `sum(recording processing seconds) / sum(audio seconds)`, including
LLM and re-transcription calls. Concurrent recordings overlap, so this is not
batch time divided by audio duration. Tokens are service-reported LLM usage,
excluding ASR tokens; output includes reasoning when counted by the server.

\* Thinking token totals are lower bounds: counters were saved for 520/524 Chinese
and 340/387 English recordings. Timing includes failed recordings but excludes
later retry batches; the English thinking run used an earlier prompt revision.
The without-thinking runs completed all recordings. These costs are separate
from the final paper scores above. For your own run, inspect `batch_summary.json`
and its per-recording counters; resumed-run timing is not full-pass timing.

### Rebuild reports

For a completed run, rebuild reports without inference:

```bash
.venv/bin/python -m agentic_ger.experiment configs/zh.json \
  --data-root data/gigaspeechbench/prepared/Vertical-Domain \
  --run-root runs/quick_zh --report-only
```

## Outputs

The command prints the run root. Open `evaluation/report.html` for aggregate
results; inspect `batch_summary.json` separately to verify completion.

```text
runs/<experiment>/<revision>/
├── batch_summary.json         # completion, failures, elapsed time and counters
├── evaluation/summary.json    # baseline/final metrics
├── evaluation/report.html    # aggregate report
└── cases/index.html          # per-recording inspection (local server required)
```

To inspect cases and play audio, serve the generated directory locally:

```bash
.venv/bin/python -m agentic_ger.reporting.server \
  --directory runs/quick_zh/cases --bind 127.0.0.1 --port 12398
```

Open `http://127.0.0.1:12398` on that host, or use an SSH tunnel. Do not expose the
server publicly; audio clips require its Python backend, not static hosting.

Each run root contains the resolved configuration, cohort manifest, completion
summary, reproducibility snapshots, per-recording traces, final transcripts,
and offline evaluation. Chinese terminology is measured with B-CER and English
with B-WER; overall CER and WER are also reported. Full-cohort scores are
12-domain macro averages. Failed recordings retain the baseline transcript and
their true completion status.

Traces and reports contain transcripts, references, audio links, and local
paths. Keep `runs/` private and do not commit it.

## Repository Layout

```text
src/agentic_ger/         agent, experiment orchestration, evaluation, reporting
configs/                 Chinese and English default configurations
prompts/                 language-specific prompts and JSON schemas
tools/                   data preparation, service checks, release checks
vendor/                  frozen GigaSpeechBench normalization adapters
tests/                   offline unit and integration tests
docs/                    service, evaluation, and clean-room procedures
```

Dataset IDs and language suffixes retain upstream `CH` and `EN` so alignment and
metric denominators remain unchanged.

For a clean-room procedure, see [reproduction](docs/reproduction.md). For metric
boundaries, see [evaluation](docs/EVALUATION.md). Third-party notices and
license concerns are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
