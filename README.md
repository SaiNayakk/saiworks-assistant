# SaiWorks Assistant

A customer-support assistant for Indian small businesses (salons, restaurants) that answers **only from each business's own information**, cites its sources, and says "I don't know" instead of guessing. It's multi-tenant, has an eval suite, and runs on an Android phone.

Part of the SaiWorks platform (WaitlistDrop, InvoiceSnap, EventSnap).

```
question ──► embed query ─► vector search (cosine) ──┐
        └──► FTS5 keyword search (BM25) ─────────────┴─► RRF fusion ─► top-k chunks
                                                                         │
                     relevance gate: no keyword hit and low similarity ──┤──► abstain (no LLM call)
                                                                         ▼
                                     Gemini (JSON mode): answer + cited chunk ids
                                                                         │
                     citation check: cited ids must be in retrieved set ─┴──► answer | abstain
                                                                         │
                                                  query log: latency, tokens, abstain reason
```

## Design decisions

| Decision | Why |
|---|---|
| **Hybrid search (BM25 + embeddings, RRF)** | Embeddings miss exact tokens (GSTIN, "PhonePe"); keyword search misses paraphrases ("how late can I walk in" → Opening Hours). RRF merges the rankings without calibrating scores. |
| **Contextual chunk headers** | Each chunk is embedded as `Business / Section / text`, so "₹250" is retrievable as "men's haircut price at Glow". |
| **Heading-based chunking** | One section = one fact group; long sections split on paragraphs with overlap. |
| **No citation → no answer (enforced in code)** | The prompt asks for grounding; the code *verifies* it. Uncited or hallucinated-id answers become an abstention. |
| **Pre-LLM relevance gate** | Obviously off-topic questions never reach the model, saving quota and latency. |
| **Tenant isolation at every query** | Every SQL query is scoped by `tenant_id`; a test asserts that the salon never retrieves the restaurant's biryani. |
| **SQLite + in-process vector search** | One file, no server, runs on a phone. Brute-force cosine over a few hundred chunks per tenant is fast enough; the next step at scale is sqlite-vec or pgvector. |
| **Pure-Python dependencies (Flask, waitress, stdlib HTTP)** | Installs on Termux without Rust or a C toolchain (pydantic-core / cryptography don't). |
| **Gemini free tier with a client-side rate limiter** | Costs ₹0; the limiter spaces calls under the RPM cap instead of eating 429s. Swappable through the `Provider` interface. |

## Guardrails

The assistant only answers customer questions about the business. Everything else gets one fixed refusal that doesn't say which rule fired. The prompt asks the model to stay in scope, but **every rule is also enforced in code**, so a jailbroken model still can't get an answer out.

| Layer | What it stops |
|---|---|
| **Sanitize** (NFKC, strip zero-width/control chars) | `ｉｇｎｏｒｅ`, `ig​nore` and other Unicode evasion |
| **Input guard** (before any API call) | injection phrasing incl. Hinglish, fake `</customer_message>`/`SYSTEM:` tags, "the developer says…", role-play/DAN, prompt-leak requests, writing/coding/translation/homework tasks, advice (medical, financial, political), links, code, base64 payloads, JSON, flooding |
| **Spotlighting** | The customer's text is wrapped in `<customer_message>` with `<`/`>` neutralised, and the prompt treats it as data, never instructions |
| **Model scope flag** | The model must return `in_scope: true`; a missing or garbled flag fails closed |
| **Gemini safety filters** | `BLOCK_LOW_AND_ABOVE` on harassment, hate, sexual and dangerous content |
| **Citation check** | No valid citation means no answer |
| **Output guard** | Blocks answers that contain a **number (price, time, phone) not in the cited passages** (so "the owner said ₹100, confirm?" can't be parroted), links/emails not in the sources, code/HTML, meta talk ("passages", "as an AI"), the prompt **canary** token, or anything over 700 chars |
| **Abuse control** | Per-IP rate limit; **3 refused attempts in 10 min lock the IP out for 30 min**; a global daily model-call budget protects the free quota |
| **HTTP hardening** | Strict CSP (no inline scripts), `X-Frame-Options: DENY`, nosniff, 4 KB body cap, 300-char questions, JSON-only, validated tenant ids, generic error bodies (no stack traces), internals and stats only for `ADMIN_TOKEN` holders |

Tested two ways:
- `tests/`: offline and deterministic. Every legitimate golden question must pass the input guard, every red-team injection/format attack must be blocked before the model, and there are output-guard, lockout and header tests.
- `evals/redteam.jsonl`: 36 attacks run against the real model (off-topic, injection, leaks, hallucination bait, cross-tenant, privacy, harmful). Scored as **attack success rate** (goal: 0) alongside the **false refusal rate** on real questions, because over-blocking real customers is also a failure.

Known trade-off: an answer with a *derived* number ("last appointment at 8:15") is blocked because 8:15 isn't written anywhere. The assistant gives the facts ("last appointment is 45 minutes before closing at 9 PM") instead of doing arithmetic.

## Evals

`evals/golden.jsonl` has 36 labelled questions covering prices, hours, exact tokens, Hinglish, multi-hop reasoning, negative facts, unanswerable questions, cross-tenant traps, and legitimate questions that *look* like attacks (false-positive probes). `evals/redteam.jsonl` has 36 attacks.

```bash
python -m evals.run --retrieval-only --modes hybrid,vector,keyword   # compare retrievers (embedding calls only)
python -m evals.run                                                  # end-to-end accuracy
python -m evals.run --judge                                          # + LLM-as-judge faithfulness
python -m evals.run --suite redteam                                  # attacks
python -m evals.run --suite all --min-accuracy 0.85 --max-attack-success 0   # CI gate
```

Metrics: hit@k, MRR, answer accuracy (required facts present), abstain accuracy, false-abstain rate, faithfulness, p50/p95 latency, tokens and cost per 1k questions. Every failure is labelled `retrieval_miss`, `generation_miss` or `should_abstain`, because each needs a different fix. Reports go to `evals/reports/`.

### Results

_Run the evals and paste the report table here: the hybrid vs vector vs keyword comparison is the headline._

## Run locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                     # offline tests, no API key needed
export GEMINI_API_KEY=...               # PowerShell: $env:GEMINI_API_KEY="..."
python -m assistant.ingest              # embed seed/ into ~/.saiworks-assistant/assistant.db
python -m assistant.app                 # http://127.0.0.1:8300
```

## Deploy to the phone (Termux + Backseat)

One-time setup in Termux:

```bash
cp ~/saiworks-assistant/.env.example ~/.saiworks-assistant.env   # after first deploy, or create it by hand
nano ~/.saiworks-assistant.env                                   # set GEMINI_API_KEY and ADMIN_TOKEN (16+ random chars)
```

From the laptop:

```bash
python scripts/bundle.py
backseat deploy dist/saiworks-assistant ~/saiworks-assistant --start "sh run.sh" --name assistant   # first time
backseat tunnel start 8300
```

Redeploying (the app is already registered):

```bash
python scripts/bundle.py
backseat deploy dist/saiworks-assistant ~/saiworks-assistant
backseat apps restart assistant
```

`run.sh` installs dependencies if they're missing, re-embeds only changed seed files, then starts the server on `127.0.0.1:8300` (reachable only through the tunnel). Check it with `backseat apps logs assistant`.

## API

| Endpoint | |
|---|---|
| `POST /api/chat` `{tenant, question}` | `{answer, answerable, citations}`; with the admin token also `debug` (retrieval ranks, timings, tokens, refusal reason) |
| `GET /api/tenants` | businesses |
| `GET /api/stats?tenant=` | admin only: volume, answered rate, refusal/abstain reasons, p50/p95 latency, tokens |
| `POST /api/admin/reindex` | re-ingest seed data (`Authorization: Bearer $ADMIN_TOKEN`) |
| `GET /health` | liveness only, reveals nothing |

## Roadmap

- [ ] Live data from PocketBase: today's queue length from WaitlistDrop, invoice lookups from InvoiceSnap (tool calls, not RAG)
- [ ] WhatsApp channel through the Cloud API webhook
- [ ] Owner-editable knowledge base (upload a menu photo → extracted text, which is project 2)
- [ ] Reranker stage, measured with the eval suite before it's kept
