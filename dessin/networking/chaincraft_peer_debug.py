"""
Optional Chaincraft UDP diagnostics: invalid-message strikes (3 → ban) and ban events.

Enable strike/ban console lines with ``DESSIN_DEBUG_PEER_BANS=1`` (or ``true`` / ``yes``).

Payload preview is included in those lines when context is available.

Write full payload dumps (plus metadata JSON) when ``DESSIN_DEBUG_PEER_BANS`` or
``DESSIN_DUMP_INVALID_STRIKE_PAYLOAD=1`` is set. Override directory with
``DESSIN_INVALID_STRIKE_DUMP_DIR``; otherwise ``model_cache/debug_udp_strikes`` is used,
falling back to the system temp dir under ``dessin_udp_strikes``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chaincraft.node import ChaincraftNode

from ..runtime.pretty_console import pretty_print


def peer_ban_debug_enabled() -> bool:
    return os.environ.get("DESSIN_DEBUG_PEER_BANS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def strike_payload_dump_enabled() -> bool:
    return peer_ban_debug_enabled() or os.environ.get(
        "DESSIN_DUMP_INVALID_STRIKE_PAYLOAD", ""
    ).strip().lower() in ("1", "true", "yes")


_MAX_DUMP_BYTES = 512 * 1024


def _invalid_strike_dump_dir() -> Path:
    env = os.environ.get("DESSIN_INVALID_STRIKE_DUMP_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    mc = Path("model_cache/debug_udp_strikes")
    try:
        mc.mkdir(parents=True, exist_ok=True)
        return mc
    except OSError:
        p = Path(tempfile.gettempdir()) / "dessin_udp_strikes"
        p.mkdir(parents=True, exist_ok=True)
        return p


def _guess_rejection_stage(message: str) -> str:
    try:
        json.loads(message)
    except json.JSONDecodeError as e:
        return f"json_decode_error: {e}"
    return "parsed_json_but_rejected_by_chaincraft_or_shared_object"


def _write_invalid_strike_dump(
    *,
    peer: Tuple[str, int],
    local_port: int,
    display_strike: int,
    ctx: Dict[str, Any],
) -> Optional[str]:
    dump_dir = _invalid_strike_dump_dir()
    stem = (
        f"dessin_invalid_udp_{int(time.time() * 1000)}_"
        f"{peer[0]}_{peer[1]}_strike{display_strike}"
    )
    base = dump_dir / stem
    msg = ctx.get("message", "")
    meta: Dict[str, Any] = {
        "timestamp_unix": ctx.get("ts"),
        "peer_ip": peer[0],
        "peer_port": peer[1],
        "local_udp_port": local_port,
        "strike_index": display_strike,
        "message_hash": ctx.get("message_hash"),
        "message_length": len(msg),
        "rejection_guess": _guess_rejection_stage(msg),
        "text_file": str(base.with_suffix(".txt")),
        "binary_truncated": False,
    }
    raw = msg.encode("utf-8", errors="replace")
    if len(raw) > _MAX_DUMP_BYTES:
        raw = raw[:_MAX_DUMP_BYTES]
        meta["binary_truncated"] = True
        meta["binary_max_bytes"] = _MAX_DUMP_BYTES

    txt_path = base.with_suffix(".txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# peer={peer[0]}:{peer[1]} local_port={local_port} ")
            f.write(f"strike={display_strike} len={len(msg)}\n")
            f.write(f"# rejection_guess={meta['rejection_guess']}\n---\n")
            f.write(msg)
        with open(base.with_suffix(".bin"), "wb") as f:
            f.write(raw)
        meta_path = base.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return str(txt_path)
    except OSError as e:
        pretty_print(f"Could not write invalid-strike dump to {base}: {e}", kind="warn")
        return None


class LoggingChaincraftNode(ChaincraftNode):
    """Same as ChaincraftNode; logs strikes/bans when ``DESSIN_DEBUG_PEER_BANS`` is set."""

    def handle_message(
        self, message: str, message_hash: str, addr: Tuple[str, int]
    ) -> None:
        addr_norm = (addr[0], addr[1])
        self._invalid_strike_context = {
            "message": message,
            "message_hash": message_hash,
            "addr": addr_norm,
            "ts": time.time(),
        }
        try:
            super().handle_message(message, message_hash, addr_norm)
        finally:
            self._invalid_strike_context = None

    def handle_invalid_message(self, addr: Tuple[str, int]) -> None:
        peer = (addr[0], addr[1])
        ctx = getattr(self, "_invalid_strike_context", None)
        already_banned = peer in self.banned_peers
        prev_strikes = self.invalid_message_counts.get(peer, 0)
        super().handle_invalid_message(addr)
        if already_banned:
            return

        after_count = self.invalid_message_counts.get(peer, 0)
        hit_third_ban = prev_strikes == 2 and peer in self.banned_peers and after_count == 0
        strike_logged = after_count > prev_strikes or hit_third_ban
        if not strike_logged:
            return

        display_strike = prev_strikes + 1
        msg_snip = ""
        dump_path: Optional[str] = None
        if ctx and (ctx["addr"][0], ctx["addr"][1]) == peer:
            msg = ctx.get("message", "")
            if len(msg) > 240:
                msg_snip = f" len={len(msg)} preview={repr(msg[:240])}…"
            else:
                msg_snip = f" len={len(msg)} body={repr(msg)}"

            if strike_payload_dump_enabled():
                dump_path = _write_invalid_strike_dump(
                    peer=peer,
                    local_port=self.port,
                    display_strike=display_strike,
                    ctx=ctx,
                )

        if peer_ban_debug_enabled():
            if peer in self.banned_peers and not hit_third_ban:
                return
            pretty_print(
                f"Chaincraft invalid-message strike {display_strike}/3 from "
                f"{peer[0]}:{peer[1]} (this node UDP port {self.port}){msg_snip}",
                kind="warn",
            )
            if dump_path:
                pretty_print(f"Invalid strike payload saved: {dump_path}", kind="warn")
        elif strike_payload_dump_enabled() and dump_path:
            pretty_print(f"Invalid strike payload saved: {dump_path}", kind="warn")

    def ban_peer(self, peer: Tuple[str, int]) -> None:
        super().ban_peer(peer)
        if not peer_ban_debug_enabled():
            return
        exp = self.banned_peers.get(peer)
        exp_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp)) if exp else "?"
        pretty_print(
            f"Chaincraft BAN {peer[0]}:{peer[1]} — peer removed from list; "
            f"banned_until≈{exp_s} (48h rule); "
            f"this_node_port={self.port} active_peers_now={len(self.peers)}",
            kind="bad",
        )
