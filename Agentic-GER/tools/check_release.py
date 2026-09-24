#!/usr/bin/env python3
"""Check tracked/candidate files without printing suspected secret values."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", ".cache", "__pycache__", "data", "runs", "models", "logs", "acceptance-work"}
PATTERNS = {
    "private_absolute_path": re.compile(r"/(?:cpfs_speech|mnt/workspace|root|home)/"),
    "private_network": re.compile(r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)"),
    "URL_credentials": re.compile(r"https?://[^\s/]+:[^\s/]+@"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "token": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "webhook": re.compile(r"https://(?:qyapi\.weixin\.qq\.com/cgi-bin/webhook/send\?key=|open\.feishu\.cn/open-apis/bot/v2/hook/)[A-Za-z0-9-]+"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    if args.tracked:
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
        files = [ROOT / name for name in output.split("\0") if name]
        if not files:
            raise SystemExit("no tracked files; stage reviewed candidates first")
    else:
        files = [p for p in ROOT.rglob("*") if p.is_file() and not any(part in IGNORED or part.startswith(".venv-") or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]
    findings = []
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            findings.append({"file": name, "kind": "tracked_symlink"})
            continue
        if any(part in IGNORED for part in path.relative_to(ROOT).parts) or path.suffix in {".wav", ".mp3", ".safetensors", ".pem", ".key"} or path.name.startswith(".env"):
            findings.append({"file": name, "kind": "excluded_artifact"})
        text = path.read_text(encoding="utf-8")
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({"file": name, "kind": kind, "line": text.count("\n", 0, match.start()) + 1})
        if path.suffix == ".py":
            ast.parse(text, filename=name)
        elif path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix == ".md":
            for target in re.findall(r"\]\(([^\s)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                relative = target.split("#", 1)[0]
                if not (path.parent / relative).exists():
                    findings.append({"file": name, "kind": "broken_local_link", "target": target})
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text())
    for entry in provenance["files"]:
        path = (ROOT / entry["file"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            findings.append({"file": entry["file"], "kind": "missing_provenance_source"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["candidate_sha256"]:
            findings.append({"file": entry["file"], "kind": "provenance_hash_mismatch"})
        if entry["byte_identical"] != (digest == entry["source_sha256"]):
            findings.append({"file": entry["file"], "kind": "provenance_identity_mismatch"})
    print(json.dumps({"files_checked": len(files), "findings": findings, "scope": "heuristic content/syntax scan, not a comprehensive secret or license audit"}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
