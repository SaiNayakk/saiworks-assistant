#!/bin/sh
# Entry point on the phone. Backseat runs this as a managed app:
#   backseat deploy dist/saiworks-assistant ~/saiworks-assistant --start "sh run.sh" --name assistant
set -e
cd "$(dirname "$0")"

ENV_FILE="$HOME/.saiworks-assistant.env"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
if [ -z "$GEMINI_API_KEY" ]; then
  echo "GEMINI_API_KEY is not set. Create $ENV_FILE (see .env.example)." >&2
  exit 1
fi

# Pure-Python deps only — nothing here needs Rust or a C compiler on Termux.
python -c "import flask, waitress" 2>/dev/null || pip install -r requirements.txt

python -m assistant.ingest   # re-embeds only files whose content changed
exec python -m assistant.app
