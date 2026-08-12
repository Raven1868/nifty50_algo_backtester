"""
Secrets loading — keeps credentials OUT of config.yaml (which is version
controlled) and out of git history entirely.

This project doesn't call any broker/paid API today, so nothing here is
required for current functionality. It exists as the drop-in point for
when/if this project is extended toward paper trading or live execution
(see README's "Using this for actual trading" guidance) — broker API keys,
if ever added, must be loaded through this module, never hard-coded or
placed in config.yaml.

Usage:
    from secrets_config import get_secret
    api_key = get_secret("BROKER_API_KEY")  # raises clearly if unset
"""
from __future__ import annotations
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env in the project root, if present — .env is git-ignored
except ImportError:
    pass  # python-dotenv is optional; env vars can still be set directly


def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Read a secret from the environment (populated from `.env` if present).
    Never reads from config.yaml — config.yaml is version-controlled and
    must never contain credentials.

    Args:
        name: environment variable name, e.g. "BROKER_API_KEY".
        default: value to return if unset and not required.
        required: if True, raises RuntimeError instead of returning None/default.
    """
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Required secret '{name}' is not set. Add it to a `.env` file "
            f"(see .env.example) or export it as an environment variable. "
            f"Never put credentials directly in config.yaml or source code."
        )
    return value
