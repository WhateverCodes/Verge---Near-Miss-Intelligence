"""Load KEY=VALUE lines from the project's .env file into os.environ (without
overriding variables that are already set). Dependency-free on purpose; only
used for server-side settings such as the SMTP configuration."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
