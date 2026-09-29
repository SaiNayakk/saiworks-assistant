"""Model providers behind one small interface.

Uses only the standard library for HTTP so it installs on Termux without
compiling anything. Swap providers by implementing `Provider`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import deque
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import Settings


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    pass


class Overloaded(ProviderError):
    """The model is busy (503) or we're over quota (429). Another model may still answer."""


class SafetyBlocked(ProviderError):
    """The provider's own content filter refused the prompt or the response."""


SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_LOW_AND_ABOVE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


class Provider(Protocol):
    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        """task is 'document' or 'query'. Returned vectors are L2-normalised."""
        ...

    def generate_json(self, system: str, user: str, model: str | None = None) -> tuple[dict, Usage]:
        ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class _RateLimiter:
    """Sliding 60s window: calls go through immediately until `rpm` calls have
    been made in the last minute, then wait for the oldest to age out.
    (Evenly spacing calls instead added a 5s delay to every quick follow-up.)"""

    def __init__(self, rpm: int):
        self.rpm = max(rpm, 1)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return
                delay = 60 - (now - self._calls[0])
            time.sleep(delay)


class GeminiProvider:
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    _TASKS = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.s = settings
        self._chat_limit = _RateLimiter(settings.chat_rpm)
        self._embed_limit = _RateLimiter(settings.embed_rpm)

    def _post(self, url: str, body: dict, limiter: _RateLimiter, attempts: int = 3, timeout: float = 20) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(attempts):
            limiter.wait()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.s.gemini_api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                # A 503 means the model is overloaded right now: retrying it only burns time the
                # fallback model could use. Rate limits (429) and other 5xx are worth one more try.
                retryable = e.code == 429 or (e.code >= 500 and e.code != 503)
                if not retryable or attempt == attempts - 1:
                    cls = Overloaded if e.code in (429, 503) else ProviderError
                    raise cls(f"Gemini HTTP {e.code}: {detail}") from e
            except TimeoutError as e:
                # Slow reads raise TimeoutError directly. A model this slow is usually overloaded: hand over to the fallback.
                raise Overloaded(f"Gemini timed out: {e}") from e
            except (urllib.error.URLError, ConnectionError, json.JSONDecodeError) as e:
                if attempt == attempts - 1:
                    raise Overloaded(f"Gemini unreachable: {getattr(e, 'reason', e)}") from e
            time.sleep(0.5 * 2 ** attempt)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        model = f"models/{self.s.embed_model}"
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit
            batch = texts[i : i + 100]
            body = {
                "requests": [
                    {
                        "model": model,
                        "content": {"parts": [{"text": t}]},
                        "taskType": self._TASKS[task],
                        "outputDimensionality": self.s.embed_dims,
                    }
                    for t in batch
                ]
            }
            resp = self._post(f"{self.BASE}/{self.s.embed_model}:batchEmbedContents", body, self._embed_limit)
            # Truncated gemini-embedding-001 vectors are not unit length; normalise ourselves.
            out.extend(normalize(e["values"]) for e in resp["embeddings"])
        return out

    def generate_json(
        self, system: str, user: str, model: str | None = None, max_tokens: int = 400,
        images: list[tuple[str, str]] | None = None, timeout: float = 10,
    ) -> tuple[dict, Usage]:
        """images: [(mime_type, base64_data)], sent inline before the text.
        If the model is overloaded, one more try on the fallback model."""
        try:
            return self._generate(system, user, model or self.s.chat_model, max_tokens, images, timeout)
        except Overloaded:
            fallback = self.s.fallback_model
            if not fallback or fallback == (model or self.s.chat_model):
                raise
            return self._generate(system, user, fallback, max_tokens, images, timeout, attempts=1)

    def _generate(self, system, user, model, max_tokens, images, timeout, attempts: int = 2) -> tuple[dict, Usage]:
        parts = [{"inlineData": {"mimeType": m, "data": d}} for m, d in images or []] + [{"text": user}]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            # A short answer never needs more; caps cost if the model is coaxed into rambling.
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": max_tokens},
            "safetySettings": SAFETY_SETTINGS,
        }
        # Flash-Lite normally answers in 1-2s; a stalled call is cheaper to retry than to wait out.
        resp = self._post(f"{self.BASE}/{model}:generateContent", body, self._chat_limit, attempts=attempts, timeout=timeout)
        meta = resp.get("usageMetadata", {})
        usage = Usage(meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0))
        if resp.get("promptFeedback", {}).get("blockReason"):
            raise SafetyBlocked(resp["promptFeedback"]["blockReason"])
        finish = (resp.get("candidates") or [{}])[0].get("finishReason")
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
            raise SafetyBlocked(finish)
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected Gemini response: {json.dumps(resp)[:300]}") from e
        return parse_json(text), usage


def parse_json(text: str) -> dict:
    """Models occasionally wrap JSON in code fences even in JSON mode."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ProviderError(f"Model returned invalid JSON: {text[:200]}") from e


class FakeProvider:
    """Deterministic offline provider for tests: hashed bag-of-words embeddings
    and a pluggable responder instead of a real model."""

    def __init__(self, responder: Callable[[str, str], dict] | None = None, dims: int = 256):
        self.dims = dims
        self.responder = responder or (lambda system, user: {"answerable": False, "answer": "", "citations": []})
        self.calls = 0

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dims
            for tok in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
            vecs.append(normalize(v))
        return vecs

    def generate_json(
        self, system: str, user: str, model: str | None = None, max_tokens: int = 400,
        images: list[tuple[str, str]] | None = None, timeout: float = 10,
    ) -> tuple[dict, Usage]:
        self.calls += 1
        self.last_images = images
        return self.responder(system, user), Usage(len(user) // 4, 20)
