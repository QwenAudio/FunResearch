# S2ST-VoiceEval

  Models and evaluation protocols for cross-lingual voice preservation in speech-to-speech translation.

  Part of [Fun Research](../README.md), maintained by the Qwen Audio Team at Alibaba Group.

  > 🚧 **Coming soon:** Model checkpoints and inference instructions are being prepared for release.

  ## Overview

  S2ST-VoiceEval evaluates how well a translated speech output preserves the source speaker's voice. Given a source-language recording
  and candidate outputs in another language, the evaluator compares speaker-embedding cosine similarities to rank the candidates by
  voice preservation.

  At inference, the adapted evaluator uses only the source recording and candidate outputs. It does not require a target-language
  recording of the same speaker.

  This project accompanies **Cross-Lingual Voice-Preservation Evaluation in Speech-to-Speech Translation**. A public paper link and
  citation will be added when available.

  ## Method

  The evaluator adapts a WavLM speaker model using:

  - **Bilingual identity adaptation (BIA):** Supervised contrastive learning and relationship preservation on real bilingual speech.
  - **Same-language ranking distillation:** An additional training stage that transfers a frozen teacher's target-reference preferences
  to a student using source-language references.
  - **Alternating language adaptation (ALA):** Language regularization applied during adaptation and studied through controlled
  comparisons.

  Evaluation measures agreement with fixed proxy and human preference labels under source- and target-reference conditions. Voice
  preservation is evaluated separately from translation correctness and naturalness.

  ## Release Roadmap

  - [ ] Adapted model checkpoints
  - [ ] Installation and inference instructions
  - [ ] Scoring examples
  - [ ] Evaluation documentation
  - [ ] Public paper link and citation

  Checkpoint download links, dependencies, supported audio formats, and runnable examples will be provided with the model release.
