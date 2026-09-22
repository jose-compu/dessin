"""Greedy char-LM generation for :class:`~dessin.llm.decoder_only_gpt.DecoderOnlyGPT` checkpoints (torrent JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from .decoder_lm_training import BUNDLED_SHAKESPEARE_CHAR_CORPUS
from .decoder_only_gpt import CharTokenizer, DecoderOnlyGPT, load_params_from_hex
from ..models.llm_model_spec import (
    DecoderOnlyTransformerSpec,
    TokenizerProfile,
    parse_architecture_dict,
)


def format_prompt_for_char_lm_profile(
    prompt: str,
    spec: DecoderOnlyTransformerSpec,
) -> str:
    """
    Map user prompt through the spec's tokenizer profile before char encoding.

    ``shakespeare_per_chat`` is a stub: framing matches ``flat_corpus`` until
    in-vocab role/turn delimiters are defined and training data aligns.
    """
    if spec.tokenizer_profile == TokenizerProfile.SHAKESPEARE_PER_CHAT:
        return prompt
    return prompt


def load_decoder_lm_from_torrent_json(
    json_path: str | Path,
    *,
    device: str = "cpu",
) -> Tuple[DecoderOnlyGPT, CharTokenizer, DecoderOnlyTransformerSpec]:
    """Load weights from a ``TorrentModelTrainer`` JSON artifact (same corpus as training)."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    arch = data.get("model_architecture") or {}
    spec = parse_architecture_dict(arch)
    tok = CharTokenizer(BUNDLED_SHAKESPEARE_CHAR_CORPUS)
    if tok.vocab_size != spec.vocab_size:
        raise ValueError(
            f"Vocab mismatch: tokenizer has {tok.vocab_size} chars, checkpoint expects {spec.vocab_size}"
        )
    dev = torch.device(device)
    model = DecoderOnlyGPT(
        vocab_size=spec.vocab_size,
        block_size=spec.block_size,
        n_layer=spec.n_layer,
        n_head=spec.n_head,
        n_embd=spec.n_embd,
    ).to(dev)
    weights_hex = (data.get("model_weights") or {}).get("weights_hex", "")
    if not weights_hex:
        raise ValueError("Missing model_weights.weights_hex")
    load_params_from_hex(model, weights_hex)
    model.eval()
    return model, tok, spec


@torch.no_grad()
def greedy_generate_char_lm(
    model: DecoderOnlyGPT,
    tok: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 96,
    device: Optional[str] = None,
    spec: Optional[DecoderOnlyTransformerSpec] = None,
) -> str:
    """Greedy continuation on the bundled Shakespeare char vocabulary (CPU-friendly)."""
    dev_s = device or "cpu"
    dev = torch.device(dev_s)
    model.to(dev)
    if spec is not None:
        prompt = format_prompt_for_char_lm_profile(prompt, spec)
    ids = [tok.stoi[c] for c in prompt if c in tok.stoi]
    if not ids:
        ids = [next(iter(tok.stoi.values()))]
    idx = torch.tensor([ids], dtype=torch.long, device=dev)
    bs = int(model.block_size)
    for _ in range(int(max_new_tokens)):
        idx_cond = idx if idx.size(1) <= bs else idx[:, -bs:]
        logits, _ = model(idx_cond)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id), dim=1)
    out_ids = idx[0, len(ids) :].tolist()
    continuation = "".join(tok.itos[int(i)] for i in out_ids)
    return prompt + continuation
