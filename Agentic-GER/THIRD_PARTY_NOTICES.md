# Third-party provenance (clearance pending)

## GigaSpeechBench normalizers

Upstream: https://github.com/SpeechColab/GigaSpeechBench

Commit: `ca782bff09a424233cd3aa1aff11c346cd0f2ed7`.

| Local file | Upstream path | SHA-256 |
|---|---|---|
| `vendor/gsb_chinese_normalizer.py` | `text_norm/CHN.py` | `926ea04cc2417b96c14345efb4a7a342794e162cad0e43901b0c00bf3b3ff88d` |
| `vendor/gsb_english_normalizer.py` | `text_norm/USA.py` | `08dfb3f04120e19615b947d6dee822023921a52b8510c477a0ea9cd5cff19056` |

Both bytes were checked against the upstream raw files during candidate preparation.
Existing source comments and attribution are preserved. The upstream tree inspected
did not contain an explicit LICENSE/NOTICE file. Obtain permission or identify the
applicable license before distributing these files. This notice is not a license.
The normalizers also reference Zhon/pinyin resources in their original comments;
any transitive attribution/permission requirements remain part of release review.

## Data and models

GigaSpeechBench data, published baseline hypotheses and model weights are not
included. Their access, usage and redistribution terms remain separate from this
project's eventual license. Model services are user-managed external dependencies.

## Python dependencies

Dependencies are installed from their distributions, not vendored here. Inspect
the installed package license metadata/notices for requests, kaldialign,
whisper-normalizer, zhconv, huggingface-hub and transitive dependencies as part of
the final release review. No blanket project license overrides their terms.

The verified environment's installed metadata declares **GPLv2+ for zhconv 1.4.3**.
This needs an explicit compatibility review before choosing the project's license.
It is not a conclusion that every part of this project must use that license.
See [metric dependencies](requirements-metrics.txt); do not change metric dependencies
merely to simplify packaging without establishing metric equivalence.
