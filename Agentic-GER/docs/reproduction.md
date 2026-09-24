# Reproduction and verification

## Clean-room procedure

Use a fresh source checkout and a new agent/session with no development context.
Provide only this repository, README, permission to download the documented public
dataset, and access to the documented model services. Do not reuse a prepared
development dataset or installed Python environment. Package caches may be reused
if disclosed. Context isolation is not filesystem isolation: use a separate
container or machine for a stronger boundary.

1. Follow README to create a fresh uv environment and install the editable package.
2. Run `.venv/bin/python -m unittest discover -s tests -v`.
3. Download the pinned HF metadata, published FunASR hypotheses and AGR-CH/AGR-EN
   audio. Run preparation and inspect the alignment/coverage manifest.
4. Run `agentic_ger.evaluation.verify_zh` and `verify_en` as Python modules.
   Require exact baseline count agreement: 524/524 and 387/387.
5. Run service preflight and both default experiment commands with `--dry-run`.
6. Run the two README non-thinking smokes, one complete recording per language.
   Do not launch the full cohorts or matrix for default acceptance.
7. Require `complete=1, failed=0` for each batch. Inspect the final transcript,
   offline evaluation, traces, HTML reports and `--report-only` regeneration.
   Record which of summary/scan/re-transcription/check/edit actually occurred. A normal
   keep decision is valid; do not force edits or tune prompts against references.
8. Run `.venv/bin/python tools/check_release.py --tracked`. Check Git status,
   ignored generated files, source provenance and frozen vendor hashes.

Do not stop or reconfigure another user's model services. Coordinate service
availability with the host owner. Record failures honestly: baseline-scored
failed outputs are valid evaluation artifacts, not successful inference runs.
Mock-service tests check packaging and control flow, not model quality or real
service compatibility. Two real recording smokes do not establish full-cohort
quality reproduction.

## Local verification record

Keep commands, exit statuses, environment versions, public source revision/hashes,
recording IDs, real completion statuses, exercised stages, metrics and limitations
under ignored `runs/`. Record download size/time and manual interventions. Never
publish raw logs without reviewing transcripts, local paths and service metadata.
Use a new run revision after changing code or scientific settings.

## Distribution prerequisites

Technical reproduction does not grant publication or redistribution permission.
Before distributing a fork or release:

- Confirm code ownership and contributor authorization, and choose a project
  LICENSE. This candidate has no owner-approved main-project license yet.
- Resolve redistribution permission for the frozen normalizers. The upstream
  tree at `ca782bff09a424233cd3aa1aff11c346cd0f2ed7` inspected during preparation
  had no explicit LICENSE/NOTICE. Hash agreement is not legal clearance.
- Review the HF dataset's current access/use terms; data and weights are not
  included. Do not infer redistribution rights from a public download.
- Review dependency notices and compatibility, particularly `zhconv==1.4.3`
  (GPLv2+ metadata). Do not change frozen metrics just to simplify licensing.
- Review the complete tracked-file list for credentials, private endpoints,
  paths, data and model artifacts. The heuristic scanner is not a full audit.
- Inspect Git history, remotes, author identity and hooks before publishing.
  Source-only exports must not import development Git objects or credentials.

See [third-party notices](../THIRD_PARTY_NOTICES.md).
