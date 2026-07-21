# Checkpoints

Model weights live here and are **git-ignored** (only this README is tracked). Place or
symlink each weight below; the scripts reference them by these relative paths, so run from
the repo root.

Base models:

| path | role | used by |
|------|------|---------|
| `Qwen2.5-1.5B/` | base LLM | training, inference |
| `whisper-small/small.pt` | speech encoder | training, inference |
| `Fun-CosyVoice3-0.5B/` | codec / vocoder | inference |
| `llmlingua-2-xlm-roberta-large-meetingbank/` | prompt compressor | data construction |

Trained ECoM checkpoints are a directory containing `model.pt` (LLM + projector + group-decode
adapter; the frozen Whisper encoder loads separately), e.g. `comthink-0302/`, selected via
`ckpt_path` in `src/scripts/inference_com.sh`.

Register a weight without copying it:

```bash
ln -s /abs/path/to/model checkpoints/<name>
```
