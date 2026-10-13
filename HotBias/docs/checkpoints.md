# Checkpoint identities

The first SHA256 column identifies the **three original input checkpoint files** for `scripts/prepare_release_weights.py`. The second identifies the model-only files prepared for release. The script writes their hashes to `SHA256SUMS` and metadata to `manifest.json` in the user-specified `--output-dir`, outside this repository.

| Prepared file | Original input checkpoint | Input SHA256 | Prepared SHA256 | Hugging Face link |
| --- | --- | --- | --- | --- |
| `source_transfer.pt` | `hotword_attention_fusion_hard_only_transfer_v1.pt` | `486b9a36e5d8f594a79343364b2ed04f542121664eda9f08f1b20b75d3a3e70a` | `f32a23145100421b58eb5583f4eda8420a8cca8778d31ddac916fe1b5683b582` | |
| `nano_aligned.pt` | `checkpoint_final.pt` from the Fun-ASR-Nano 100-hour alignment run | `9d5f9f5cf7cf4769fd9dca72604abab68178ae5b741fc90a890e653731be4a23` | `80101174363607bd348add455949cdb0fdc43a8f285513e18effcd65671213f9` | |
| `firered_aligned.pt` | `checkpoint_final.pt` from the FireRedASR2-AED Top-50 alignment run | `b73e7619e1765833b72e37c7e33bd4d218365a94d3a8c9d5c784785a19977aee` | `3b7483ed58eba521d57bf5554b9cd566f7c2304d6e32893ec6794598bf5812f8` | |

Download links for prepared model-only files are pending publication. To verify the original input files before preparation, compare their hashes with the table. To verify prepared outputs after preparation, run:

```bash
(cd /path/to/hotbias-checkpoints && sha256sum -c SHA256SUMS)
```

## Source transfer package

`source_transfer.pt` is the export of the final Qwen3 source HotBias run, after the 2,000-hour first stage, 2,000-hour continuation, and 350.051-hour hard-example stage. It is a packaging of that run's trained parameters, not a separate training result. Its payload has a `shared_core_state` with 130 tensors and a `source_adapter.state` with 18 source-model projection tensors. The corresponding ASR backbone is absent.

The shared core can initialize Nano, FireRed, or another target implementation. Target input projections must match the target model's frozen feature dimensions and be aligned before target inference. The original source projections are tied to the source model's 2,048-dimensional audio and text features; they are not a replacement for Nano or FireRed projections.

## Fun-ASR-Nano-2512 aligned checkpoint

`nano_aligned.pt` contains the full 148-tensor HotBias module state, configuration, and inference status. The original training checkpoint also held optimizer and scheduler states, which the preparation script removes. Neither file contains Fun-ASR-Nano base-model weights. The prepared checkpoint retains `ready_for_target_inference=true` and `initialization_only=false`.

The reported one-epoch alignment used a 100.00065-hour selection of 60,176 rows, with 941 optimizer steps. The Nano backbone and the 130-tensor shared core were frozen; only the six input-projection modules (18 tensors, 1,575,936 parameters) were trainable. The original selection file refers to data stored outside this repository and is not a redistributable dataset.

The recorded Nano base-model snapshot was the original toolkit-format `FunAudioLLM/Fun-ASR-Nano-2512` at revision `272c57b82523ada6fd87095e955f8e29100979ab`; the experiment's `QwenAudio/Fun-ASR` checkout was `53a56d80667320b44a7dd779f5bf8c024b6c30a8`.

Running `fun_asr_nano/initialize_adapter.py` with the source package recreates the 142 reused tensors, while six dimension-dependent Nano projection matrices start from fresh random values. The original Nano initialization file is not among these three release files. Consequently, this alignment script reproduces the method and training scope but cannot promise a bitwise recreation of the provided Nano final checkpoint, even with the same seed.

The default Nano inference script uses reference-independent candidate truncation and Top-50/Trie gating. An optional character-CTC gate is implemented, but its frozen Wav2Vec2-CTC model is separate from these three checkpoints and was not exercised end to end in the local PyTorch 2.3 environment. Candidate truncation remains reference-independent even with CTC enabled; cases with more than 40 supplied hotwords therefore differ from the historical evaluator and require reevaluation before direct comparison with the paper's main table.

## FireRedASR2-AED aligned checkpoint

`firered_aligned.pt` contains the full 148-tensor HotBias module state, configuration, and inference status. The original training checkpoint also held optimizer state, which the preparation script removes. Neither file contains FireRedASR2-AED base-model weights. The prepared checkpoint retains `alignment_completed=true` and `initialization_only=false`.

The reported one-epoch run used 5,929 selected utterances (10.002 hours). Of those, 4,713 utterances (7.098 hours) produced training updates, over 590 optimizer steps. The FireRed backbone and the 130-tensor shared core remained frozen; only its six target projections changed. This checkpoint is for the raw-base Top-50 plus token-Trie decoder with no CTC gate, no hotword prompt, and no external language model. The reported base-model revision is `2304afed56eacfee6256dee5937ed22ffa0b64ec` and the FireRedASR2S code revision is `4e7d9aaf4482a47cec1724807026b9b151926eb5`.

## Compatibility

These three files represent one source transfer package and two completed target adaptations. Do not mix their target projections. Load the corresponding ASR backbone and tokenizer separately, using each model's own integration code. A fresh target model or a changed feature interface requires another alignment run using `source_transfer.pt`.

Their stored fusion configuration uses `max_bias=10.0`, `max_gain=1.5`, and `logit_margin=0.3`. The released implementation clamps the dynamic per-candidate logit increment to that `max_bias`; uncapped equations introduced later are not represented by these checkpoints.
