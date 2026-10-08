# Release validation

Version 0.1.0 was checked with Python 3.12, PyTorch/torchaudio 2.7.0,
Transformers 4.53.1 and an NVIDIA H100.

- 14 interface tests passed, covering audio preprocessing, input errors,
  candidate ordering, reference reuse, pairwise accuracy and atomic batch output.
- All 710 exported model-state tensors exactly matched the original base model
  plus the paper's final adaptation checkpoint.
- On 10 real evaluation recordings covering Mandarin-to-English and
  English-to-Mandarin translation, exported-model embeddings exactly matched
  both the original inference engine and the saved paper embeddings.
- All 8 candidate cosine scores in that check matched exactly using CUDA BF16.

These checks validate export and inference parity on the selected recordings;
they do not represent a new full benchmark evaluation.

Source checkpoint SHA256:
`a1de689ba064d5cadb91d40d7355bde4dac76186aa1a4164f3ba17edd3431a76`

Exported safetensors SHA256:
`92a221b2fd3b272f7bb99a23baa2dd34c5c1f0cfaaecada3b5b282d72332676b`
