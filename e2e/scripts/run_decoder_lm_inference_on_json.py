#!/usr/bin/env python3
"""Greedy char-LM inference from a PoGO / e2e torrent JSON checkpoint (micro_gpt_char)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to micro_gpt_char_*.json (TorrentModelTrainer artifact)",
    )
    p.add_argument(
        "--prompt",
        default="From fairest creatures",
        help="Seed text (Shakespeare char vocab)",
    )
    p.add_argument("--max-new", type=int, default=96, dest="max_new", help="Greedy tokens to append")
    p.add_argument("--device", default=os.environ.get("DESSIN_MICRO_GPT_DEVICE", "cpu"))
    args = p.parse_args()

    path = args.json_path.expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    try:
        with path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        wh = (blob.get("model_weights") or {}).get("weights_hex", "")
        exp = (blob.get("model_weights") or {}).get("checksum", "")
        if wh and exp and hashlib.sha256(wh.encode()).hexdigest() != exp:
            print("ERROR: weights_hex checksum mismatch vs JSON", file=sys.stderr)
            return 3
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read JSON: {e}", file=sys.stderr)
        return 4

    try:
        from dessin.llm.decoder_lm_inference import greedy_generate_char_lm, load_decoder_lm_from_torrent_json
    except ImportError as e:
        print(f"ERROR: import failed (need torch): {e}", file=sys.stderr)
        return 5

    dev = args.device
    model, tok, spec = load_decoder_lm_from_torrent_json(path, device=dev)
    text = greedy_generate_char_lm(
        model,
        tok,
        args.prompt,
        max_new_tokens=int(args.max_new),
        device=dev,
        spec=spec,
    )
    print(f"file={path}")
    print(f"prompt={args.prompt!r}")
    print(f"greedy_len={len(text)}")
    print(text[:800] + ("…" if len(text) > 800 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
