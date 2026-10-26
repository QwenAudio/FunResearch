# Examples

This directory provides 30 audio–prompt pairs illustrating the three
training-data construction routes of FineAudio-Judge. Here, a prompt
means the text description or audio-generation instruction associated
with an audio sample.

The examples are grouped by data-license category, then by construction route.
`A1` contains five FineVideo-derived examples. `A2` contains the other 25
examples. These categories do not replace the construction-route labels.

| Category | Construction route | Pairs | English prompts | Chinese prompts | Browse |
|---|---|---:|---:|---:|---|
| A1 | Route 1: FineVideo real-world audio | 5 | 5 | 0 | [Pairs](audio_prompt_pairs/A1/route1/pairs.jsonl) · [Audio](audio_prompt_pairs/A1/route1/audio/) |
| A2 | Route 1: Vript real-world audio | 5 | 5 | 0 | [Pairs](audio_prompt_pairs/A2/route1/pairs.jsonl) · [Audio](audio_prompt_pairs/A2/route1/audio/) |
| A2 | Route 2: Deterministic composition | 10 | 5 | 5 | [Pairs](audio_prompt_pairs/A2/route2/pairs.jsonl) · [Audio](audio_prompt_pairs/A2/route2/audio/) |
| A2 | Route 3: Prompt-driven Audio-Oscar generation | 10 | 5 | 5 | [Pairs](audio_prompt_pairs/A2/route3/pairs.jsonl) · [Audio](audio_prompt_pairs/A2/route3/audio/) |

Each category/route directory contains a `pairs.jsonl` manifest and an `audio/`
directory with the corresponding WAV files.
The examples preserve the original prompts and audio files recorded
by the construction pipeline.

## Pair records

Each line in `pairs.jsonl` contains one audio–prompt pair:

| Field | Description |
|---|---|
| `sample_id` | Sample identifier, unique within the route. |
| `route` | `route1`, `route2`, or `route3`. |
| `language` | Prompt language: `en` or `zh`. This does not specify the language of every sound or spoken segment. |
| `prompt` | Text description or audio-generation instruction associated with the audio. |
| `audio` | WAV path relative to the directory containing this manifest. |
| `duration_seconds` | Duration measured from the released WAV file, in seconds. |

From the directory containing `pyproject.toml`, load an example with:

```python
import json
from pathlib import Path

manifest = Path("examples/audio_prompt_pairs/A1/route1/pairs.jsonl")
with manifest.open(encoding="utf-8") as handle:
    example = json.loads(next(handle))

audio_path = manifest.parent / example["audio"]
print(example["prompt"])
print(audio_path)
```

All audio is provided as WAV with its downloaded encoding preserved. Route 1
and Route 2 examples are 16 kHz mono PCM16; Route 3 examples are 44.1 kHz
stereo PCM24.

## Construction guide

To construct audio–prompt pairs from your own inputs, see the
[data-construction guide](../data_pipeline/README.md).

## Rubric decomposition example

`example_prompt.txt` contains an audio-generation instruction for a
train-carriage scene with multiple sound events and spoken exchanges.
It can be used as input to the rubric decomposition pipeline to produce
rubric questions.

## Sources

`A1/route1/` contains the five FineVideo-derived examples
(`route1_demo_01` through `route1_demo_05`). `A2/route1/` contains the five
Vript-derived examples (`route1_demo_06` through `route1_demo_10`).
`A2/route2/` contains the asset-based composition examples, and `A2/route3/`
contains the Audio-Oscar-generated examples.

## License / 数据许可声明

`audio_prompt_pairs/A1/` 内的数据筛选自开源的 FineVideo 数据集，使用时请参阅并遵守 FineVideo 的 [Terms of Use](https://huggingface.co/datasets/HuggingFaceFV/finevideo#terms-of-use-for-finevideo)。

The data in `audio_prompt_pairs/A1/` is derived from a filtered subset of the
FineVideo dataset; please refer to and comply with the
[Terms of Use of FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo#terms-of-use-for-finevideo)
when using it.

`audio_prompt_pairs/A2/` 内的数据采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可发布。该许可仅适用于本项目拥有或有权授予许可的部分。

The data in `audio_prompt_pairs/A2/` is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), to the extent
that this project holds or is authorized to license the relevant rights.

Third-party source materials retain their upstream licenses and terms of use.
Generated audio also remains subject to the applicable model or service terms.
See the [project license statement and disclaimer](../README.md#license) and
[third-party notices](../THIRD_PARTY_NOTICES.md) for further information.
