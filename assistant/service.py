"""Server-to-server API for SaiWorks apps (WaitlistDrop, InvoiceSnap, ...).

Browsers never call this directly: each app's backend does, with its own
service token. Tenants are namespaced per app ("waitlistdrop--glow-salon"),
so one app can never read or overwrite another app's knowledge base.

    SERVICE_TOKENS="waitlistdrop=<long random>,invoicesnap=<long random>"
"""

from __future__ import annotations

import base64
import binascii
import hmac
import re
from dataclasses import asdict

from flask import Blueprint, current_app, g, jsonify, request

from . import guardrails
from .answer import answer_question
from .brief import write_brief
from .compose import LANGUAGES, TONES, compose_reminder
from .extract import IMAGE_TYPES, MAX_TEXT, extract_invoice
from .ingest import clone_tenant, delete_document, delete_tenant, ingest_document, upsert_tenant
from .retrieval import Hit

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}$")
DOC_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LIVE_FACT_BASE_ID = 900_000  # far above real chunk ids, so citations can't collide
BODY_LIMIT = 64 * 1024        # documents are <= 20k chars; JSON escaping can double that
IMAGE_BODY_LIMIT = 6 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def parse_tokens(raw: str) -> dict[str, str]:
    """'app=token,app2=token2' -> {token: app}. Tokens shorter than 24 chars are rejected."""
    out = {}
    for part in filter(None, (p.strip() for p in raw.split(","))):
        app, _, token = part.partition("=")
        if app and len(token) >= 24:
            out[token] = app
    return out


def build_blueprint(ctx) -> Blueprint:
    """ctx provides: db(), provider, index, cfg, abuse (AbuseTracker)."""
    bp = Blueprint("service", __name__, url_prefix="/v1")
    tokens = parse_tokens(ctx.cfg.service_tokens)

    @bp.before_request
    def authenticate():
        header = request.headers.get("Authorization", "")
        presented = header.removeprefix("Bearer ").strip()
        app = next((a for t, a in tokens.items() if hmac.compare_digest(t, presented)), None) if presented else None
        if app is None:
            return jsonify(error="unauthorized"), 401
        g.app = app
        # The public app keeps a 4 KB cap; authenticated apps may send documents and photos.
        request.max_content_length = IMAGE_BODY_LIMIT if request.endpoint == "service.extract" else BODY_LIMIT

    def tenant_id(slug: str) -> str | None:
        return f"{g.app}--{slug}" if SLUG.match(slug) else None

    def resolve_template(name: str) -> str | None:
        """A clone source: a shared seed template (no '--'), or one of this app's own tenants."""
        if SLUG.match(name) and ctx.db().execute("SELECT 1 FROM tenants WHERE id = ?", (name,)).fetchone():
            return name
        own = tenant_id(name)
        if own and ctx.db().execute("SELECT 1 FROM tenants WHERE id = ?", (own,)).fetchone():
            return own
        return None

    def body() -> dict:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    @bp.put("/tenants/<slug>")
    def put_tenant(slug):
        tid, name = tenant_id(slug), str(body().get("name", "")).strip()[:80]
        if not tid or not name:
            return jsonify(error="valid slug and name required"), 400
        upsert_tenant(ctx.db(), tid, name)
        return jsonify(tenant=slug)

    @bp.delete("/tenants/<slug>")
    def remove_tenant(slug):
        tid = tenant_id(slug)
        if not tid:
            return jsonify(error="invalid slug"), 400
        delete_tenant(ctx.db(), tid)
        return jsonify(deleted=slug)

    @bp.post("/tenants/<slug>/clone")
    def clone(slug):
        data = body()
        tid, src = tenant_id(slug), resolve_template(str(data.get("from", "")))
        name = str(data.get("name", "")).strip()[:80]
        rename = data.get("rename") if isinstance(data.get("rename"), dict) else {}
        rename = {str(k): str(v) for k, v in rename.items() if DOC_KEY.match(str(k)) and DOC_KEY.match(str(v))}
        if not tid or not src or not name:
            return jsonify(error="valid slug, name and an existing 'from' template required"), 400
        return jsonify(tenant=slug, chunks=clone_tenant(ctx.db(), src, tid, name, rename))

    @bp.put("/tenants/<slug>/documents/<key>")
    def put_document(slug, key):
        data = body()
        tid = tenant_id(slug)
        title = str(data.get("title", "")).strip()[:80]
        content = str(data.get("content", ""))
        if not tid or not DOC_KEY.match(key) or not title or not content.strip() or len(content) > 20_000:
            return jsonify(error="valid slug, key, title and content (<= 20k chars) required"), 400
        if not ctx.db().execute("SELECT 1 FROM tenants WHERE id = ?", (tid,)).fetchone():
            return jsonify(error="unknown tenant; PUT /v1/tenants/<slug> first"), 404
        chunks = ingest_document(ctx.db(), ctx.provider, tid, key, title, content)
        return jsonify(document=key, chunks=chunks, unchanged=chunks is None)

    @bp.delete("/tenants/<slug>/documents/<key>")
    def remove_document(slug, key):
        tid = tenant_id(slug)
        if not tid or not DOC_KEY.match(key):
            return jsonify(error="invalid slug or key"), 400
        return jsonify(deleted=delete_document(ctx.db(), tid, key))

    @bp.get("/tenants/<slug>/unanswered")
    def unanswered(slug):
        """Real questions the assistant couldn't answer from the knowledge base — the owner's to-do list.
        Refusals (off-topic, attacks) are excluded: those aren't gaps to fill."""
        tid = tenant_id(slug)
        if not tid:
            return jsonify(error="invalid slug"), 400
        rows = ctx.db().execute(
            """SELECT question, MAX(created_at) AS last_asked, COUNT(*) AS times FROM queries
               WHERE tenant_id = ? AND answerable = 0
                 AND abstain_reason IN ('model_abstained', 'no_relevant_context', 'no_valid_citation')
                 AND created_at >= datetime('now', '-30 days')
               GROUP BY lower(question) ORDER BY times DESC, last_asked DESC LIMIT 20""",
            (tid,),
        ).fetchall()
        return jsonify(questions=[dict(r) for r in rows])

    @bp.post("/chat")
    def chat():
        data = body()
        tid = tenant_id(str(data.get("tenant", "")))
        question = data.get("question")
        if not tid or not isinstance(question, str) or not question.strip():
            return jsonify(error="tenant and question required"), 400
        if len(question) > ctx.cfg.max_question_chars:
            return jsonify(error=f"Please keep questions under {ctx.cfg.max_question_chars} characters."), 400

        # The app forwards the end user's IP so abuse limits still apply per person.
        end_user = f"{g.app}:{str(data.get('end_user') or 'unknown')[:64]}"
        if ctx.abuse.check(end_user):
            return jsonify(error="Too many requests. Please try again later."), 429

        live = []
        for i, f in enumerate((data.get("live_facts") or [])[:12]):
            if isinstance(f, dict) and str(f.get("text", "")).strip():
                live.append(Hit(
                    chunk_id=LIVE_FACT_BASE_ID + i, source="live",
                    heading=str(f.get("title", "Live status"))[:80], text=str(f["text"])[:600],
                ))
        try:
            result = answer_question(ctx.db(), ctx.index, ctx.provider, ctx.cfg, tid, question, live_facts=live)
        except KeyError:
            return jsonify(error="unknown tenant"), 404
        if result.refused:
            ctx.abuse.strike(end_user)
        out = asdict(result)
        out["retrieved"] = [{k: h[k] for k in ("chunk_id", "source", "heading", "score")} for h in out["retrieved"]]
        return jsonify(out)

    @bp.post("/brief")
    def brief():
        data = body()
        tid = tenant_id(str(data.get("tenant", "")))
        facts = data.get("facts")
        if not tid or not isinstance(facts, list) or not facts:
            return jsonify(error="tenant and facts required"), 400
        clean = []
        for f in facts[:40]:
            if isinstance(f, dict) and str(f.get("id", "")).isdigit() and str(f.get("text", "")).strip():
                clean.append({"id": int(f["id"]), "text": guardrails.sanitize(str(f["text"]))[:300]})
        row = ctx.db().execute("SELECT name FROM tenants WHERE id = ?", (tid,)).fetchone()
        if not row:
            return jsonify(error="unknown tenant"), 404
        from .answer import llm_calls_today

        if llm_calls_today(ctx.db()) >= ctx.cfg.daily_llm_budget:
            return jsonify(error="daily AI budget reached"), 503
        result = write_brief(ctx.provider, row["name"], clean)
        with ctx.db():
            ctx.db().execute(
                """INSERT INTO queries(tenant_id, question, answer, answerable, abstain_reason, retrieved_ids, cited_ids,
                   input_tokens, output_tokens, retrieve_ms, generate_ms, total_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (tid, "[owner brief]", "\n".join(b["text"] for b in result.bullets), int(bool(result.bullets)),
                 None if result.bullets else "brief_empty", "[]", "[]",
                 result.usage["input_tokens"], result.usage["output_tokens"], 0, result.ms, result.ms),
            )
        return jsonify(asdict(result))

    def over_budget() -> bool:
        from .answer import llm_calls_today

        return llm_calls_today(ctx.db()) >= ctx.cfg.daily_llm_budget

    def log_call(tid: str, label: str, answer: str, ok: bool, reason: str | None, usage: dict, ms: float):
        with ctx.db():
            ctx.db().execute(
                """INSERT INTO queries(tenant_id, question, answer, answerable, abstain_reason, retrieved_ids, cited_ids,
                   input_tokens, output_tokens, retrieve_ms, generate_ms, total_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (tid, label, answer[:2000], int(ok), reason, "[]", "[]",
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0), 0, ms, ms),
            )

    def tenant_name(tid: str) -> str | None:
        row = ctx.db().execute("SELECT name FROM tenants WHERE id = ?", (tid,)).fetchone()
        return row["name"] if row else None

    @bp.post("/extract")
    def extract():
        """Invoice draft from a message and/or a photo. Numbers are verified against the source."""
        data = body()
        tid = tenant_id(str(data.get("tenant", "")))
        text = data.get("text") if isinstance(data.get("text"), str) else ""
        img = data.get("image") if isinstance(data.get("image"), dict) else None
        if not tid or (not text.strip() and not img):
            return jsonify(error="tenant and text or image required"), 400
        if len(text) > MAX_TEXT:
            return jsonify(error=f"Please keep the message under {MAX_TEXT} characters."), 400
        image = None
        if img:
            mime = str(img.get("mime", ""))
            try:
                raw = base64.b64decode(str(img.get("data", "")), validate=True)
            except (binascii.Error, ValueError):
                return jsonify(error="image must be base64"), 400
            if mime not in IMAGE_TYPES or not raw or len(raw) > MAX_IMAGE_BYTES:
                return jsonify(error="image must be a JPEG, PNG or WebP under 4 MB"), 400
            image = (mime, base64.b64encode(raw).decode())
        name = tenant_name(tid)
        if not name:
            return jsonify(error="unknown tenant"), 404
        end_user = f"{g.app}:{str(data.get('end_user') or 'unknown')[:64]}"
        if ctx.abuse.check(end_user):
            return jsonify(error="Too many requests. Please try again later."), 429
        if over_budget():
            return jsonify(error="daily AI budget reached"), 503
        result = extract_invoice(ctx.provider, name, text, image)
        if result.refused:
            ctx.abuse.strike(end_user)
        log_call(tid, "[extract]" + (" +photo" if image else ""), text[:500], bool(result.items), None if result.items else "nothing_extracted",
                 result.usage, result.ms)
        return jsonify(asdict(result))

    @bp.post("/compose")
    def compose():
        """A short reminder message in the tone the app chose, checked before it's returned."""
        data = body()
        tid = tenant_id(str(data.get("tenant", "")))
        facts = data.get("facts")
        tone, language = str(data.get("tone", "")), str(data.get("language", "en"))
        if not tid or not isinstance(facts, list) or not facts or tone not in TONES or language not in LANGUAGES:
            return jsonify(error="tenant, facts, tone and language required"), 400
        name = tenant_name(tid)
        if not name:
            return jsonify(error="unknown tenant"), 404
        if over_budget():
            return jsonify(error="daily AI budget reached"), 503
        result = compose_reminder(ctx.provider, name, [str(f) for f in facts], tone, language)
        log_call(tid, f"[compose:{tone}:{language}]", result.message or "", result.message is not None, result.reason,
                 result.usage, result.ms)
        return jsonify(asdict(result))

    return bp
