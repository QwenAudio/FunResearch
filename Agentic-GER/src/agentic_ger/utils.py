"""Optional service authentication; secrets remain in environment variables."""

import os
from pathlib import Path


# This application uses a source checkout installed with `uv pip install -e .`.
# Prompts, scientific configs and generated runs remain outside the Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent


def service_headers(role: str) -> dict[str, str]:
    if role not in {"correction", "asr"}:
        raise ValueError("unknown service role")
    key = os.environ.get("CORRECTION_API_KEY" if role == "correction" else "ASR_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}
