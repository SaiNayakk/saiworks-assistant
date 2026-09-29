"""HTTP API + web chat. Flask + waitress: both pure Python, so they install on Termux.

Public surface is deliberately tiny: the chat page, /api/tenants and /api/chat.
Internals (retrieval scores, token counts, stats, reindex) need ADMIN_TOKEN.

    python -m assistant.app
"""

from __future__ import annotations

import hmac
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, request, send_from_directory

from .answer import answer_question
from .config import settings
from .db import connect
from .ingest import ingest_all
from .retrieval import VectorIndex
from .service import build_blueprint

STATIC = Path(__file__).parent / "static"
TENANT_ID = re.compile(r"^[a-z0-9-]{1,40}$")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


class AbuseTracker:
    """Per-IP request rate limit plus strikes: IPs that repeatedly trip the
    guardrails get locked out, so probing for a bypass is slow and noisy."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.requests: dict[str, deque] = defaultdict(deque)
        self.strikes: dict[str, deque] = defaultdict(deque)
        self.locked_until: dict[str, float] = {}
        self.lock = threading.Lock()

    def check(self, ip: str) -> str | None:
        now = time.monotonic()
        with self.lock:
            if self.locked_until.get(ip, 0) > now:
                return "locked"
            q = self.requests[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.cfg.chat_per_ip_per_min:
                return "rate"
            q.append(now)
            return None

    def strike(self, ip: str) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.strikes[ip]
            while q and now - q[0] > self.cfg.strike_window_s:
                q.popleft()
            q.append(now)
            if len(q) >= self.cfg.max_strikes:
                self.locked_until[ip] = now + self.cfg.lockout_s
                q.clear()


def create_app(provider=None, cfg=settings) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024  # a question is tiny; reject big bodies before parsing
    if provider is None:
        from .llm import GeminiProvider

        provider = GeminiProvider(cfg)

    local = threading.local()  # one SQLite connection per server thread
    index = VectorIndex()
    abuse = AbuseTracker(cfg)

    def db():
        if not hasattr(local, "conn"):
            local.conn = connect(cfg.db_path)
        return local.conn

    def client_ip() -> str:
        # We bind to localhost, so only cloudflared can reach us and its header is trustworthy.
        return request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"

    def is_admin() -> bool:
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return bool(cfg.admin_token) and bool(token) and hmac.compare_digest(token, cfg.admin_token)

    # Server-to-server API for the other SaiWorks apps (per-app tokens, namespaced tenants).
    app.register_blueprint(build_blueprint(SimpleNamespace(db=db, provider=provider, index=index, cfg=cfg, abuse=abuse)))

    @app.after_request
    def headers(resp):
        resp.headers.update(SECURITY_HEADERS)
        return resp

    @app.errorhandler(Exception)
    def on_error(e):
        # Never leak stack traces or internals to the public.
        code = getattr(e, "code", 500)
        code = code if isinstance(code, int) and 400 <= code < 600 else 500
        if code == 500:
            app.logger.exception("unhandled error")
        messages = {404: "not found", 405: "method not allowed", 413: "request too large"}
        return jsonify(error=messages.get(code, "something went wrong")), code

    @app.get("/")
    def home():
        return send_from_directory(STATIC, "index.html")

    @app.get("/app.js")
    def script():
        return send_from_directory(STATIC, "app.js")

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.get("/api/tenants")
    def tenants():
        rows = db().execute("SELECT id, name FROM tenants ORDER BY name").fetchall()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/chat")
    def chat():
        if not request.is_json:
            return jsonify(error="expected JSON"), 415
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="expected a JSON object"), 400
        tenant_id, question = body.get("tenant"), body.get("question")
        if not isinstance(tenant_id, str) or not TENANT_ID.match(tenant_id):
            return jsonify(error="invalid tenant"), 400
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="question is required"), 400
        if len(question) > cfg.max_question_chars:
            return jsonify(error=f"Please keep questions under {cfg.max_question_chars} characters."), 400

        ip = client_ip()
        admin = is_admin()
        if not admin:
            limit = abuse.check(ip)
            if limit:
                return jsonify(error="Too many requests. Please try again later."), 429
        try:
            result = answer_question(db(), index, provider, cfg, tenant_id, question)
        except KeyError:
            return jsonify(error="unknown business"), 404
        if result.refused and not admin:
            abuse.strike(ip)

        payload = {"answer": result.answer, "answerable": result.answerable, "citations": result.citations}
        if admin:  # retrieval internals are for the owner, not the public
            payload["debug"] = {
                "refused": result.refused,
                "abstain_reason": result.abstain_reason,
                "retrieved": result.retrieved,
                "usage": result.usage,
                "timings_ms": result.timings_ms,
            }
        return jsonify(payload)

    @app.get("/api/stats")
    def stats():
        """Owner view: volume, refusals, abstain reasons, latency percentiles, token spend."""
        if not is_admin():
            return jsonify(error="unauthorized"), 401
        tenant_id = request.args.get("tenant")
        where, args = ("WHERE tenant_id = ?", (tenant_id,)) if tenant_id else ("", ())
        rows = db().execute(
            f"SELECT answerable, abstain_reason, total_ms, input_tokens, output_tokens FROM queries {where}", args
        ).fetchall()
        if not rows:
            return jsonify(queries=0)
        lat = sorted(r["total_ms"] for r in rows)
        reasons: dict[str, int] = defaultdict(int)
        for r in rows:
            if r["abstain_reason"]:
                reasons[r["abstain_reason"]] += 1
        return jsonify(
            queries=len(rows),
            answered_rate=round(sum(r["answerable"] for r in rows) / len(rows), 3),
            abstain_reasons=reasons,
            latency_ms={"p50": percentile(lat, 50), "p95": percentile(lat, 95)},
            tokens={"input": sum(r["input_tokens"] for r in rows), "output": sum(r["output_tokens"] for r in rows)},
        )

    @app.post("/api/admin/reindex")
    def reindex():
        if not is_admin():
            return jsonify(error="unauthorized"), 401
        return jsonify(ingest_all(db(), provider, cfg.seed_dir))

    return app


def percentile(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1)))
    return round(sorted_vals[i], 1)


if __name__ == "__main__":
    from waitress import serve

    if not settings.admin_token or len(settings.admin_token) < 16:
        raise SystemExit("ADMIN_TOKEN must be set to a random string of at least 16 characters.")
    print(f"SaiWorks assistant on http://{settings.host}:{settings.port}")
    serve(create_app(), host=settings.host, port=settings.port, threads=4)
