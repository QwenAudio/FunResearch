---
license: cc-by-sa-3.0
language:
- zh
- en
tags:
- speech-to-speech
- speaker-similarity
- wavlm
- voice-preservation
---

# S2ST-VoiceEval

[GitHub](https://github.com/QwenAudio/FunResearch/tree/main/S2ST-VoiceEval)

Evaluate how well translated speech preserves the source speaker's voice,
without a target-language reference. Higher cosine scores indicate greater
speaker similarity.

This WavLM-Large-based model implements BIA+ALA+Rank from
**Cross-Lingual Voice-Preservation Evaluation in Speech-to-Speech Translation**.

## Usage

Install the companion S2ST-VoiceEval package, then run:

```bash
s2st-voiceeval score \
  --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav \
  --candidate translated.wav
```

## License

CC BY-SA 3.0. Based on Microsoft's
[WavLM speaker model](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification).
See [NOTICE.md](NOTICE.md) and [UPSTREAM_LICENSE](UPSTREAM_LICENSE) for upstream terms.
