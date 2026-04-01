from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

from memprimitive.utils import _runtime


ROOT = Path(__file__).resolve().parent.parent
MEMPRIMITIVE_ENV_PATH = ROOT / "memprimitive" / ".env"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(MEMPRIMITIVE_ENV_PATH, override=False)


def _has_real_runtime_llm() -> bool:
    return all(
        bool(os.environ.get(key, "").strip())
        for key in ("MEMPRIMITIVE_API_KEY", "MEMPRIMITIVE_BASE_URL", "MEMPRIMITIVE_MODEL")
    )


@pytest.fixture
def require_real_runtime() -> None:
    if not _has_real_runtime_llm():
        pytest.skip(
            "Integration tests require MEMPRIMITIVE_API_KEY, "
            "MEMPRIMITIVE_BASE_URL, and MEMPRIMITIVE_MODEL."
        )
    _runtime._DEFAULT_RUNTIME = None
