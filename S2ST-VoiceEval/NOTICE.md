# Third-party notices

The model combines Microsoft's WavLM-Large speaker encoder and ECAPA backend,
adapted with bilingual identity learning, language regularization and ranking
distillation by the authors of S2ST-VoiceEval.

## Model implementation

The ECAPA implementation in src/s2st_voiceeval/_model.py preserves the module
names and operations of the Microsoft UniSpeech speaker-verification code.
The compatibility layer maps its WavLM parameters to Transformers and preserves
the original pooling, normalization and layer-fusion behavior.

Upstream: https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification

The upstream notice distributed with the original implementation is retained
verbatim in src/s2st_voiceeval/UPSTREAM_LICENSE (CC BY-SA 3.0).
That notice applies to the derived implementation; the repository's Apache-2.0
license does not replace third-party terms.

## Weights

The exported weights contain the original frozen WavLM parameters and the
adapted parameters of the S2ST-VoiceEval evaluator. Their provenance and source
checkpoint hashes are recorded in config.json. Upstream terms remain applicable
to the incorporated pretrained weights; this export does not relicense them
under the repository's code license.

The original WavLM speaker checkpoint is the "WavLM large / Fix pre-train: No"
entry on the upstream page. No training audio, speaker-ID mapping, optimizer
state or language-classifier parameters are included in the inference export.
