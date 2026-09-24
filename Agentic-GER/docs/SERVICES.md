# External service contract

This repository does not install or start model servers. Existing services must
implement the following API, not merely listen on the expected port.

## Correction (Qwen3.8-27B)

- `GET /v1/models`: `data` contains the configured model ID and `max_model_len`.
- `POST /tokenize`: accepts model, messages, chat_template_kwargs; returns integer
  `count` and `max_model_len`. This vLLM-style endpoint is required; a generic
  OpenAI-compatible chat endpoint alone is insufficient.
- `POST /v1/chat/completions`: accepts `response_format.type=json_schema`,
  strict JSON schemas, seed, max_tokens, temperature, top_p, top_k, min_p,
  presence_penalty, repetition_penalty and chat_template_kwargs.
- Non-thinking sends `enable_thinking=false`, `preserve_thinking=true`.
  Thinking optionally sends reasoning_effort both top-level and in chat-template
  kwargs. It is not exercised by default acceptance.
- Return valid JSON as `choices[0].message.content`, not only a reasoning field.
- Minimum advertised context is 65,536 for the provided configs; each actual call
  additionally checks prompt tokens + output budget against the service limit.
- Default LLM request deadline: 7,200 seconds; recording timeout: 21,600 seconds.
  These are upper bounds, not expected latencies. Three bounded LLM attempts.

## Re-transcription ASR (Qwen3-ASR-1.7B)

- `GET /v1/models`: configured model ID must exist.
- `POST /v1/chat/completions`: one user message with a content item
  `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}`.
- Requests contain a PCM WAV crop selected from the original recording, temperature
  0, seed 0 and max_tokens 512. Timeout is 240 seconds.
- Return text in `choices[0].message.content`; optional `<asr_text>` delimiters are
  handled. No reference, hotword list or reference-derived hint is sent to ASR.

## Configuration, credentials and checks

Use localhost defaults or trusted endpoints passed explicitly. Optional Bearer
authentication comes from `CORRECTION_API_KEY` / `ASR_API_KEY` in the environment;
the variables are not serialized into configuration. Never embed keys in URL
userinfo or query strings. A `.env` file is not auto-loaded.

Preflight validates model IDs, context and tokenizer response, not actual acoustic
quality or schema decoding. The two real recording smokes validate generation.
If a smoke finds no suspects, it may not exercise ASR; inspect the trace and report
this as untested rather than claiming acoustic-path verification.

The included configurations retain language-specific sampling: Chinese NT
temperature 0.7, English NT temperature 0.1; both top_p 0.8, top_k 20, min_p 0,
presence_penalty 1.5, repetition_penalty 1.0, seed 0. Stage token budgets are in
each `pack.json`. Default workers are CH 10 / EN 16 per correction endpoint;
one-recording acceptance only occupies one worker. No deployment changes are
performed by this repository. Do not change shared servers to make a smoke pass.
