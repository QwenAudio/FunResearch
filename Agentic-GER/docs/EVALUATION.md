# Evaluation and reproducibility boundaries

Frozen source: [SpeechColab/GigaSpeechBench](https://github.com/SpeechColab/GigaSpeechBench/tree/ca782bff09a424233cd3aa1aff11c346cd0f2ed7).
The two vendored normalizers must remain byte-identical to the hashes in vendor/README.md.
Baseline regression checks compare exact per-domain insertion, deletion,
substitution, correct/reference and biased-error counts, as well as cohort size.

The published HF FunASR hypotheses are the baseline: this project does not rerun
FunASR. Official reference segmentation/timestamps provide the shared evaluation
units. The preparation tool joins hypotheses by SID and time bounds, never by
assuming two lists have matching order. It retains empty baseline strings.

Only hypothesis text and timestamped audio are available to inference. Reference
text and entities live in `ref/` and are consumed only by evaluators/reporting.
The correction flow does not use the published Qwen3-ASR result cache: it calls
the configured Qwen3-ASR service on selected audio crops.

Chinese biased error is character-based (B-CER); English is word-based (B-WER).
The inherited JSON field names use `b_wer` for both languages. General CER/WER
is auxiliary. Macro averages weight each included domain equally. A full-language
report includes all 12 domains; a filtered smoke is not a full-language report.
Micro totals preserve corpus-wide counts and denominators.

A terminal failed recording is scored as the entire unchanged baseline, not as
its partially corrected output. Failure counts remain visible. Default recording
retry rounds are zero. Request-level JSON/schema recovery does not imply a
quality improvement and is not a post-hoc judge. Success means a valid completed
run, not that its error rate necessarily decreased.

Prompt packs and core inference behavior are inherited without scientific tuning.
This release packages the current default CH/EN configurations; it does not claim
to recreate every historical experiment or the private Local correction matrix.
Model weights, tokenizer/chat template and serving versions also affect results.
Record them externally when reproducing quality; seed 0 does not guarantee
bit-identical GPU inference across deployments.
