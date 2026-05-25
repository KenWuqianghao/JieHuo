"""Shared Ollama client for local LLM generation and labeling."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def ollama_available() -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def call_ollama(
    system: str,
    user: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.7,
    json_mode: bool = True,
) -> str:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["message"]["content"]


def extract_json(content: str):
    """Parse JSON from model output, tolerating markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find first JSON array or object in the text
        for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        raise


def parse_query_list(parsed) -> list[str]:
    if isinstance(parsed, list):
        out: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                q = item.get("query") or item.get("text") or item.get("q")
                if isinstance(q, str) and q.strip():
                    out.append(q.strip())
        return out
    if isinstance(parsed, dict):
        for key in ("queries", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parse_query_list(parsed[key])
    return []
