# Vendored evaluator component

`gsb_chinese_normalizer.py` is the Chinese text normalizer used by the local
GigaSpeechBench checkout from
<https://github.com/AlexTYJ/Multilingual-ASR-Benchmark>.

The file was copied from `text_norm/CHN.py`.  Its SHA-256 before copying was:

```text
926ea04cc2417b96c14345efb4a7a342794e162cad0e43901b0c00bf3b3ff88d
```

Keeping this exact version in the repository prevents later changes to another
checkout from silently changing CER or B-WER.

`gsb_english_normalizer.py` is the official `text_norm/USA.py` from
GigaSpeechBench commit `ca782bff09a424233cd3aa1aff11c346cd0f2ed7`.
Its source SHA-256 is:

```text
08dfb3f04120e19615b947d6dee822023921a52b8510c477a0ea9cd5cff19056  USA.py
```
