# Reproduction notes

This repository contains the HotBias code, but does not bundle model weights, upstream ASR models, or original training/evaluation audio. The three separately prepared HotBias checkpoints are identified by SHA256 in [checkpoints.md](checkpoints.md); their shared 130-tensor core is identical.

The reported setup and audited training artifacts do not describe precisely the same experimental configuration. We record the differences rather than claiming that the current public scripts exactly regenerate reported results:

| Item | Reported setup | Audited checkpoint / released code |
| --- | --- | --- |
| Source training | 2,000 h pretraining, then 350.05 h hard examples | The source transfer package traces to a 2,000 h stage, a 2,000 h continuation, and 350.051 h of hard examples. |
| Target alignment data | 98.59 h for each target | Nano final checkpoint comes from a 100.00065 h selected manifest; FireRed selected about 10.002 h, of which 4,713 utterances (7.098 h) contributed training updates. |
| Acoustic gate | The same frozen Chinese Wav2Vec2-CTC model is described for all three ASR systems | Nano has an **optional** external character-CTC gate, not enabled by default and not end-to-end validated in the local PyTorch 2.3 environment. The released FireRed decoder has **no CTC gate**. |
| Bias formula | The displayed dynamic increment is uncapped | All three checkpoint configurations use `max_bias=10.0`; the released core clamps the increment to this value. |
| Candidate selection | The paper reports aggregate evaluation results | Released Nano inference keeps the first 40 distinct supplied candidates without consulting the reference transcript. This differs from historical evaluation cases that used reference-aware truncation above the cap. |

The Nano adapter initializer loads 142 compatible tensors from the source package but creates six dimension-dependent projection matrices afresh. The original pre-alignment Nano initialization is not among these three checkpoints, so retraining can follow the same method without reproducing the historical final weights bit for bit.

The source Qwen3-based ASR backbone is internal and absent. `source_transfer.pt` is sufficient to initialize target alignment, not to run source-side ASR or its original pretraining. The FireRed and Nano adapters also require their separately obtained upstream ASR models.

**Hugging Face links for the weights and public evaluation-data downloads have not yet been supplied.** Add approved downloads with licenses and checksums when available. Private speech data and internal selection manifests must not be uploaded without authorization.

For a new evaluation, use the same candidate list, text normalization, acoustic gate, model revision, and decoding parameters for each paired baseline/HotBias comparison. The current scripts should not be used to quote exact reported values without rerunning that evaluation.
