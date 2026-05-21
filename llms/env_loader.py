"""Load API keys from .env file without python-dotenv dependency."""

import os
from pathlib import Path


def load_env(env_path: str = None):
    """Parse a .env file and set environment variables.

    Simple parser that handles KEY = "VALUE" and KEY = VALUE formats.
    """
    if env_path is None:
        env_path = Path.cwd() / ".env"
    env_path = Path(env_path)

    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ[key] = value
