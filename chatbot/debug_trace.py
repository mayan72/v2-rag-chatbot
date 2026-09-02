"""
RAG debugger.

Writes each pipeline step as JSON to:
  chatbot/logs/rag_debug.log

The terminal stays quiet unless RAG_DEBUG_VERBOSE=1.

Disable file traces with: RAG_DEBUG=0
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from config import LOG_DIR

ENABLED = os.getenv("RAG_DEBUG", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}

VERBOSE = os.getenv("RAG_DEBUG_VERBOSE", "0").strip().lower() in {
    "1",
    "true",
    "on",
    "yes",
}

DEBUG_LOG = LOG_DIR / "rag_debug.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def dbg(step: str, **data: Any) -> None:
    if not ENABLED:
        return

    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        **data,
    }
    line = json.dumps(payload, default=str, ensure_ascii=False)
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    if VERBOSE:
        print("\n========== RAG_DEBUG |", step, "==========")
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        print("====================================\n", flush=True)
