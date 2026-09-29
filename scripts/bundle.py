"""Copy only what the phone needs into dist/saiworks-assistant.

Backseat uploads a folder as-is, so this keeps .venv, tests, eval reports and
local databases off the phone.

    python scripts/bundle.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "saiworks-assistant"
INCLUDE = ["assistant", "seed", "requirements.txt", "run.sh", ".env.example"]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for name in INCLUDE:
    src = ROOT / name
    if src.is_dir():
        shutil.copytree(src, OUT / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, OUT / name)
# Termux's sh chokes on CRLF line endings from a Windows checkout.
run_sh = OUT / "run.sh"
run_sh.write_bytes(run_sh.read_bytes().replace(b"\r\n", b"\n"))

files = sum(1 for p in OUT.rglob("*") if p.is_file())
print(f"Bundled {files} files into {OUT}")
