"""Local dev server: embed any changed seed files, then serve.

Reads GEMINI_API_KEY / ADMIN_TOKEN from .env in the project root.

    python scripts/dev.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waitress import serve  # noqa: E402

from assistant.app import create_app  # noqa: E402
from assistant.config import settings  # noqa: E402
from assistant.db import connect  # noqa: E402
from assistant.ingest import ingest_all  # noqa: E402
from assistant.llm import GeminiProvider  # noqa: E402

missing = [name for name, value in (("GEMINI_API_KEY", settings.gemini_api_key), ("ADMIN_TOKEN", settings.admin_token)) if not value]
if missing:
    sys.exit(f"Missing {', '.join(missing)} — fill them in {Path(__file__).resolve().parent.parent / '.env'}")
if len(settings.admin_token) < 16:
    sys.exit("ADMIN_TOKEN must be at least 16 characters.")

provider = GeminiProvider(settings)
for s in ingest_all(connect(settings.db_path), provider, settings.seed_dir):
    print(f"{s['tenant']}: {s['updated']} updated ({s['chunks']} chunks), {s['skipped']} unchanged", flush=True)

print(f"SaiWorks assistant on http://{settings.host}:{settings.port}", flush=True)
serve(create_app(provider), host=settings.host, port=settings.port, threads=4)
