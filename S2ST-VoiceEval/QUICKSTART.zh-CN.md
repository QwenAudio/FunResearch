# 快速使用

输入原说话人的源语音和翻译后的语音，输出说话人相似度。分数越高，模型认为音色越相似。无需目标语言参考录音。

## 安装

使用 Python 3.10–3.13，克隆代码并安装：

```bash
git clone https://github.com/QwenAudio/FunResearch.git
cd FunResearch/S2ST-VoiceEval
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

首次运行会从 [Hugging Face](https://huggingface.co/liangwenrui/S2ST-VoiceEval) 自动下载约 1.30 GB 的完整权重并缓存。也可以将 `--model` 指向包含 `config.json` 和 `model.safetensors` 的本地目录。

## 两段音频评分

```bash
s2st-voiceeval score --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav --candidate translated.wav
```

结果为 JSON，其中 score 是 [-1, 1] 内的余弦相似度，不是概率。默认有 GPU 就使用 GPU，没有则使用 CPU；可加 --device cpu 强制使用 CPU。

## 多个系统排序

```bash
s2st-voiceeval rank --model liangwenrui/S2ST-VoiceEval \
  --reference source.wav --candidate system_a.wav system_b.wav system_c.wav
```

输出按分数从高到低排列。输入支持 WAV、FLAC，自动转为单声道并重采样到 16 kHz。

## Python 调用

```python
from s2st_voiceeval import VoiceEvaluator

model = VoiceEvaluator('liangwenrui/S2ST-VoiceEval')
print(model.score('source.wav', 'translated.wav'))
print(model.rank('source.wav', ['system_a.wav', 'system_b.wav']))
```

连续评分时复用同一个 model，避免反复加载权重。批量 JSONL 输入、PairAcc 计算和完整参数见 [README.md](README.md)。
