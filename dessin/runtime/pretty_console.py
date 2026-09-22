"""
TTY-aware colored console lines with category emojis for DeSSIN (CLI, nodes, tests).

Disable: ``NO_COLOR=1`` or ``DESSIN_NO_FANCY=1`` (also honors ``DESSIN_E2E_NO_FANCY``).
Force color when not a TTY: ``FORCE_COLOR=1``.
"""

from __future__ import annotations

import contextvars
import os
import sys
from contextlib import contextmanager
from typing import Any, Generator, Optional, TextIO

# When several DessinNode instances run in one process, bind the label around work
# (mining loop, gossip handling) so ``[dessin]`` / ``[node]`` lines show which instance emitted them.
_LOG_NODE_LABEL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "dessin_log_node_label", default=None
)


def resolve_log_node_label_from_config(config: Any) -> Optional[str]:
    """Derive a short console label from ``DessinConfig.network`` (or return ``None``)."""
    net = getattr(config, "network", None)
    if net is None:
        return None
    explicit = getattr(net, "log_node_label", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    port = getattr(net, "port", None)
    if port is not None:
        return f"p{port}"
    return None


@contextmanager
def bind_pretty_log_node(label: Optional[str]) -> Generator[None, None, None]:
    """Temporarily set the log-node suffix for the current context (supports nesting)."""
    if not label:
        yield
        return
    token = _LOG_NODE_LABEL.set(label)
    try:
        yield
    finally:
        _LOG_NODE_LABEL.reset(token)


def _tag_with_bound_node(base_tag: str) -> str:
    label = _LOG_NODE_LABEL.get()
    if not label:
        return base_tag
    if "|" in base_tag.rstrip("]"):
        return base_tag
    if base_tag.endswith("]") and base_tag.startswith("["):
        return f"{base_tag[:-1]}|{label}]"
    return f"{base_tag}|{label}"

# (emoji, ANSI SGR for message body)
PRETTY_KIND: dict[str, tuple[str, str]] = {
    "info": ("📋", "36"),
    "setup": ("🛰️", "94"),
    "mesh": ("🔗", "35"),
    "block": ("🧱", "32;1"),
    "soak": ("🌊", "33"),
    "wait": ("⏳", "34"),
    "ok": ("✅", "32"),
    "bad": ("❌", "31"),
    "warn": ("⚠️", "33"),
    "baseline": ("📍", "36"),
    "training": ("🧠", "35;1"),
    "gossip": ("📡", "34"),
    "detail": ("🔹", "90"),
    "clock": ("🕐", "96"),
    "time_slow": ("🐢", "33;1"),
    "time_fast": ("⚡", "92"),
    "time_keep": ("⏱️", "36"),
}


def format_interval_sec(sec: float) -> str:
    """Human-readable duration for logs (block slots, mining cadence)."""
    s = float(sec)
    if s >= 3600.0:
        return f"{s / 3600.0:.2f}h ({s:.0f}s)"
    if s >= 60.0:
        return f"{s / 60.0:.1f}min ({s:.0f}s)"
    return f"{s:.0f}s"


def interval_change_kind(old_sec: float, new_sec: float) -> str:
    """``pretty_print`` kind for a slot change (longer / shorter / ~unchanged)."""
    if new_sec > old_sec + 0.5:
        return "time_slow"
    if new_sec < old_sec - 0.5:
        return "time_fast"
    return "time_keep"


def adjustment_flag_kind(flag: str) -> str:
    """Map ``increase`` / ``decrease`` / ``maintain`` to colored timing kinds."""
    a = (flag or "").strip().lower()
    if a == "increase":
        return "time_slow"
    if a == "decrease":
        return "time_fast"
    return "time_keep"


def pretty_style_enabled() -> bool:
    if os.environ.get("DESSIN_NO_FANCY", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("DESSIN_E2E_NO_FANCY", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip().lower() in ("1", "true", "yes"):
        return True
    return sys.stdout.isatty()


def sgr(code: str, text: str) -> str:
    if not pretty_style_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def pretty_print(
    message: str,
    *,
    kind: str = "info",
    tag: str = "[dessin]",
    suffix: str = "",
    file: TextIO | None = None,
    flush: bool = True,
) -> None:
    """
    Print one line: ``tag`` + category emoji + ``message`` + optional ``suffix`` (often dimmed).

    ``kind`` selects emoji and body color from ``PRETTY_KIND``; unknown kinds fall back to info.
    When ``bind_pretty_log_node`` is active, the tag gains ``|label`` (e.g. ``[dessin|node_0]``).
    """
    out = file or sys.stdout
    emoji, code = PRETTY_KIND.get(kind, PRETTY_KIND["info"])
    tag = _tag_with_bound_node(tag)
    if pretty_style_enabled():
        tag_part = sgr("1;90", tag)
        body = sgr(code, message)
        suf = sgr("2", suffix) if suffix else ""
        print(f"{tag_part} {emoji} {body}{suf}", file=out, flush=flush)
    else:
        print(f"{tag} {emoji} {message}{suffix}", file=out, flush=flush)


def pretty_format(message: str, *, kind: str = "info", tag: str = "[dessin]", suffix: str = "") -> str:
    """Return the same string ``pretty_print`` would emit (no newline), for logging adapters."""
    tag = _tag_with_bound_node(tag)
    emoji, code = PRETTY_KIND.get(kind, PRETTY_KIND["info"])
    if pretty_style_enabled():
        tag_part = sgr("1;90", tag)
        body = sgr(code, message)
        suf = sgr("2", suffix) if suffix else ""
        return f"{tag_part} {emoji} {body}{suf}"
    return f"{tag} {emoji} {message}{suffix}"
