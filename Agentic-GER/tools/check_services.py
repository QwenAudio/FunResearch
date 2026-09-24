#!/usr/bin/env python3
"""Check the documented model, context and tokenizer service contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from agentic_ger.runner import require_asr_service, require_full_context_services
from agentic_ger.utils import service_headers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--llm-url", action="append")
    parser.add_argument("--asr-url")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    urls = args.llm_url or config["services"]["correction_urls"]
    asr = args.asr_url or config["services"]["retranscription_url"]
    model = config["models"]["correction"]
    require_full_context_services(urls, config["runtime"]["required_context"], model)
    require_asr_service(asr, config["models"]["retranscription"])
    for url in urls:
        response = requests.post(url.split("/v1/chat/completions", 1)[0] + "/tokenize",
                                 headers=service_headers("correction"),
                                 json={"model": model, "messages": [{"role": "user", "content": "Service check."}],
                                       "chat_template_kwargs": {"enable_thinking": config["inference"]["thinking"], "preserve_thinking": True}}, timeout=60)
        response.raise_for_status()
        info = response.json()
        if not isinstance(info.get("count"), int) or int(info.get("max_model_len", 0)) < config["runtime"]["required_context"]:
            raise SystemExit("tokenizer response must include count and sufficient max_model_len")
    print("PASS: model IDs, context capacity and /tokenize contract. Run a recording smoke to verify generation/audio.")


if __name__ == "__main__":
    main()
