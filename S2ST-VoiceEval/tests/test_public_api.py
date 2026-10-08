import json

import numpy as np
import pytest
import soundfile as sf
import torch
from torchaudio.functional import resample

from s2st_voiceeval.cli import batch, main
from s2st_voiceeval.evaluator import VoiceEvaluator, read_audio, resolve_model
from s2st_voiceeval.metrics import pairwise_accuracy


def test_stereo_resampling_preserves_amplitude(tmp_path):
    # Unequal channels catch selecting just one channel or normalizing amplitude.
    t = np.arange(8000, dtype=np.float32) / 8000
    left = 0.2 * np.sin(2 * np.pi * 250 * t)
    right = 0.6 * np.sin(2 * np.pi * 250 * t)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1), 8000, subtype="FLOAT")
    expected = resample(torch.from_numpy((left + right) / 2), 8000, 16000)
    actual = read_audio(path)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (16000,)


@pytest.mark.parametrize("case", ["empty", "short", "silent", "nonfinite"])
def test_invalid_reference_has_clear_error(tmp_path, case):
    wave = np.zeros(16000, dtype=np.float32)
    if case == "empty":
        wave = wave[:0]
    elif case == "short":
        wave = wave[:100]
    elif case == "nonfinite":
        wave[100] = np.nan
    path = tmp_path / f"{case}.wav"
    sf.write(path, wave, 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        read_audio(path)
    if case == "silent":
        assert read_audio(path, allow_silence=True).sum() == 0


def test_local_typo_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path / "missing")


def fake_evaluator():
    evaluator = object.__new__(VoiceEvaluator)
    evaluator.precision = "fp32"
    calls = []
    vectors = {"ref": [1, 0], "a": [0, 2], "b": [3, 0], "c": [6, 0]}
    def embed(path, *, reference):
        calls.append((str(path), reference))
        return np.array(vectors[str(path)], dtype=np.float32)
    evaluator._embed = embed
    return evaluator, calls


def test_score_order_and_stable_ranking():
    evaluator, calls = fake_evaluator()
    assert evaluator.score_many("ref", ["a", "b", "c"]) == [0, 1, 1]
    assert calls.count(("ref", True)) == 1
    ranking = evaluator.rank("ref", ["a", "b", "c"])
    assert [item["index"] for item in ranking] == [1, 2, 0]
    assert [item["rank"] for item in ranking] == [1, 2, 3]
    assert evaluator.score("ref", "b") == 1
    with pytest.raises(ValueError):
        evaluator.score_many("ref", [])
    with pytest.raises(TypeError):
        evaluator.rank("ref", "a.wav")


def test_pairacc_pools_examples_and_handles_ties():
    report = pairwise_accuracy([
        {"score_a": .8, "score_b": .3, "preferred": "a"},
        {"score_a": .2, "score_b": .7, "preferred": "a"},
        {"score_a": .5, "score_b": .5, "preferred": "b"},
    ])
    assert report == {"n": 3, "correct": 1.5, "ties": 1, "pairacc_percent": 50.0}


@pytest.mark.parametrize("rows", [
    [], [{"score_a": .1, "score_b": .2, "preferred": "tie"}],
    [{"score_a": float("nan"), "score_b": .2, "preferred": "a"}],
])
def test_invalid_metric_inputs(rows):
    with pytest.raises(ValueError):
        pairwise_accuracy(rows)


def test_pairacc_cli(tmp_path, capsys):
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"score_a": 1, "score_b": 0, "preferred": "a"}\n')
    assert main(["pairacc", "--input", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["pairacc_percent"] == 100


class RecordingEvaluator:
    precision = "fp32"

    def __init__(self):
        self.calls = []

    def rank(self, reference, candidates):
        self.calls.append((reference, candidates))
        return [{"rank": 1, "index": 0, "candidate": str(candidates[0]), "score": .5}]


def test_batch_relative_paths_and_atomic_output(tmp_path):
    folder = tmp_path / "manifests"
    folder.mkdir()
    manifest = folder / "input.jsonl"
    manifest.write_text('{"id":"one","reference":"source.wav","candidates":["a.wav"]}\n')
    output = tmp_path / "result.jsonl"
    evaluator = RecordingEvaluator()
    batch(evaluator, manifest, output)
    assert evaluator.calls == [(folder / "source.wav", [folder / "a.wav"])]
    result = json.loads(output.read_text())
    assert result["ranking"][0]["candidate"] == "a.wav"
    with pytest.raises(FileExistsError):
        batch(evaluator, manifest, output)


def test_batch_failure_leaves_no_partial_output(tmp_path):
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"reference":"ref.wav","candidates":["a.wav"]}\n{"reference":5}\n')
    output = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        batch(RecordingEvaluator(), manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))
