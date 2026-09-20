"""Explicit, single-request API access using environment credentials."""
import json
import os
import time
from pathlib import Path
import urllib.request
import urllib.error

from .io import digest, project_root, read_json, write_json


def request_json(provider, model, payload, destination, *, enabled=False):
    """Persist the request before sending. Ambiguous calls require manual reconciliation."""
    if not enabled:
        raise RuntimeError("API execution is disabled")
    destination = Path(destination).resolve()
    if not destination.is_relative_to((project_root() / "outputs").resolve()):
        raise ValueError("API request records belong under ignored outputs/")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Request folder already exists; inspect it before deciding whether to retry"
        )
    key_name = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[provider]
    key = os.environ.get(key_name)
    if not key:
        raise RuntimeError("Set " + key_name + " in the environment")
    endpoints = {
        "gemini": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "openai": "https://api.openai.com/v1/responses",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }
    headers = {"Content-Type": "application/json"}
    headers.update(
        {"x-goog-api-key": key}
        if provider == "gemini"
        else {"Authorization": "Bearer " + key}
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "request.json", payload)
    write_json(
        destination / "pending.json",
        {"model": model, "provider": provider, "state": "response_not_yet_saved"},
    )
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            endpoints[provider],
            json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        value = json.loads(raw)
        # Raw responses may contain provider-internal metadata. Keep them under ignored outputs/.
        temporary = destination / "response.json.tmp"
        temporary.write_bytes(raw)
        temporary.replace(destination / "response.json")
        write_json(
            destination / "receipt.json",
            {
                "response_sha256": digest(raw),
                "elapsed_seconds": time.perf_counter() - started,
                "state": "response_saved",
                "cost_note": "Provider invoice is authoritative",
            },
        )
        (destination / "pending.json").unlink()
        return value
    except Exception as error:
        write_json(
            destination / "error.json",
            {
                "type": type(error).__name__,
                "http_status": getattr(error, "code", None),
                "state": "requires_review_before_retry",
            },
        )
        raise RuntimeError(
            "API request failed; inspect the saved receipt. No automatic retry was made."
        ) from None


def generation_payload(provider, model, prompt, opus_settings=None):
    if provider == "openai":
        return {
            "model": model,
            "input": prompt,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": 1024,
            "store": False,
        }
    if provider == "gemini":
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }
    if provider != "openrouter":
        raise ValueError("Unknown provider")
    if opus_settings is None:
        raise ValueError("Supply the frozen OpenRouter generation settings")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **opus_settings,
    }


def answer_text(provider, response):
    if provider == "openai":
        return "".join(
            part.get("text", "")
            for item in response.get("output", [])
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    if provider == "gemini":
        candidates = response.get("candidates", [])
        return "".join(
            part.get("text", "")
            for candidate in candidates
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        )
    return response["choices"][0]["message"].get("content") or ""
