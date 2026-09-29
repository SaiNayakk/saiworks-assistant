"""Runtime configuration, read once from environment variables.

Everything has a default so the app starts on a fresh Termux install with
only GEMINI_API_KEY set.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader for local dev (no python-dotenv dependency).
    Real environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gemini-3.5-flash-lite"))
    # Used once when the chat model is overloaded (HTTP 503/429 after retries).
    fallback_model: str = field(default_factory=lambda: _env("FALLBACK_MODEL", "gemini-3.1-flash-lite"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gemini-3.5-flash-lite"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "gemini-embedding-001"))
    embed_dims: int = field(default_factory=lambda: int(_env("EMBED_DIMS", "768")))
    # Free tier is ~15 RPM for Flash-Lite; stay under it on the client side.
    chat_rpm: int = field(default_factory=lambda: int(_env("CHAT_RPM", "12")))
    embed_rpm: int = field(default_factory=lambda: int(_env("EMBED_RPM", "60")))

    db_path: Path = field(
        default_factory=lambda: Path(
            _env("DB_PATH", str(Path.home() / ".saiworks-assistant" / "assistant.db"))
        ).expanduser()
    )
    seed_dir: Path = field(
        default_factory=lambda: Path(_env("SEED_DIR", str(Path(__file__).resolve().parent.parent / "seed")))
    )

    # Retrieval knobs — tuned via evals, not by feel.
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(_env("CANDIDATE_K", "20")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    # Below this cosine similarity *and* with no keyword hit, we abstain without calling the LLM.
    min_similarity: float = field(default_factory=lambda: float(_env("MIN_SIMILARITY", "0.45")))

    # Notional USD prices per 1M tokens, used only to report what a paid tier would cost.
    price_in_per_m: float = field(default_factory=lambda: float(_env("PRICE_IN_PER_M", "0.10")))
    price_out_per_m: float = field(default_factory=lambda: float(_env("PRICE_OUT_PER_M", "0.40")))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8300")))
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    # "waitlistdrop=<token>,invoicesnap=<token>" — one token per calling app (>= 24 chars each).
    service_tokens: str = field(default_factory=lambda: _env("SERVICE_TOKENS", ""))
    # Public chat is exposed through a tunnel and spends our free quota — cap it per IP.
    chat_per_ip_per_min: int = field(default_factory=lambda: int(_env("CHAT_PER_IP_PER_MIN", "6")))
    max_question_chars: int = 300
    # Hard cap on model calls per day across all tenants, kept under the free tier's
    # daily request limit so abuse can't exhaust the quota or run up a bill.
    daily_llm_budget: int = field(default_factory=lambda: int(_env("DAILY_LLM_BUDGET", "800")))
    # IPs that trip the guardrails this many times in STRIKE_WINDOW_S get locked out.
    max_strikes: int = field(default_factory=lambda: int(_env("MAX_STRIKES", "3")))
    strike_window_s: int = 600
    lockout_s: int = field(default_factory=lambda: int(_env("LOCKOUT_S", "1800")))


settings = Settings()
