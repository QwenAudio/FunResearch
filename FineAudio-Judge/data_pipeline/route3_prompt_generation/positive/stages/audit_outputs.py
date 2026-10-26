from __future__ import annotations

import argparse
import hashlib
import json
import re
import wave
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def language_matches(text: str, language: str) -> bool:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if language == "zh":
        return cjk >= 20 and cjk / max(1, cjk + latin) >= 0.5
    if language == "en":
        return cjk == 0
    return False


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Route 3 Audio-Oscar outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-chinese", type=int)
    parser.add_argument("--expected-english", type=int)
    args = parser.parse_args()

    sample_roots = sorted(args.output_root.glob("samples/*"))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    seen_audio: set[str] = set()
    for sample_root in sample_roots:
        try:
            if not (sample_root / "COMPLETE").is_file():
                raise ValueError("missing COMPLETE marker")
            source = read_json(sample_root / "input.json")
            summary = read_json(sample_root / "run_summary.json")
            prompt = str(source.get("prompt", "")).strip()
            language = str(source.get("language", ""))
            if not prompt or not language_matches(prompt, language):
                raise ValueError("empty prompt or language validation failed")
            final_audio = Path(str(summary.get("final_audio", "")))
            if not final_audio.is_file() or final_audio.stat().st_size == 0:
                raise ValueError("missing final audio")
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            audio_hash = sha256_file(final_audio)
            if prompt_hash in seen_prompts:
                raise ValueError("duplicate prompt hash")
            if audio_hash in seen_audio:
                raise ValueError("duplicate audio hash")
            seen_prompts.add(prompt_hash)
            seen_audio.add(audio_hash)
            rows.append({
                "sample_id": sample_root.name,
                "route": "route3_prompt_generation",
                "language": language,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "audio_path": str(final_audio.resolve()),
                "audio_sha256": audio_hash,
                "audio_duration_seconds": round(wav_duration(final_audio), 6),
                "final_mix_remixed": bool(summary.get("final_mix_remixed")),
                "source": source.get("source"),
                "source_attempt_id": source.get("source_attempt_id"),
            })
        except Exception as exc:
            errors.append({"sample_id": sample_root.name, "error": str(exc)})

    languages = Counter(row["language"] for row in rows)
    checks = {
        "expected_total": args.expected_total is None or len(rows) == args.expected_total,
        "expected_chinese": args.expected_chinese is None or languages["zh"] == args.expected_chinese,
        "expected_english": args.expected_english is None or languages["en"] == args.expected_english,
        "no_invalid_samples": not errors,
        "unique_prompts": len(seen_prompts) == len(rows),
        "unique_audio": len(seen_audio) == len(rows),
    }
    write_jsonl(args.output_root / "pairs.jsonl", rows)
    write_jsonl(args.output_root / "validation_errors.jsonl", errors)
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "samples": len(rows),
        "languages": dict(sorted(languages.items())),
        "final_mix_remixed": sum(row["final_mix_remixed"] for row in rows),
        "invalid_samples": len(errors),
        "pairs": str((args.output_root / "pairs.jsonl").resolve()),
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
