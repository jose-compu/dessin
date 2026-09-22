"""Shared char-LM training loop for decoder-only GPT specs (micro-GPT + GPT-2 nano presets)."""

from __future__ import annotations

import os
from typing import Optional

import torch

from .decoder_only_gpt import CharTokenizer, DecoderOnlyGPT, eval_lm_loss, flatten_params_float64, load_params_from_hex
from ..models.llm_model_spec import (
    DecoderOnlyTransformerSpec,
    LLMVariant,
    TokenizerKind,
    TokenizerProfile,
    read_int_env,
    resolve_char_lm_tokenizer_options,
    variant_default_hyperparams,
)
from ..runtime.pretty_console import pretty_print
from .simple_trainer import TrainingResult

# Bundled corpus (public-domain style) for deterministic PoGO / E2E / inference.
BUNDLED_SHAKESPEARE_CHAR_CORPUS = """From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
""" * 80


# Backwards-compatible alias
_CORPUS_SHAKESPEARE = BUNDLED_SHAKESPEARE_CHAR_CORPUS


def _resolve_hparams(
    variant: LLMVariant,
    env_prefix: str,
    *,
    n_layer: Optional[int],
    n_head: Optional[int],
    n_embd: Optional[int],
    block_size: Optional[int],
) -> tuple[int, int, int, int]:
    d = variant_default_hyperparams(variant)
    nl = n_layer if n_layer is not None else read_int_env(env_prefix, "N_LAYER", d["n_layer"])
    nh = n_head if n_head is not None else read_int_env(env_prefix, "N_HEAD", d["n_head"])
    ne = n_embd if n_embd is not None else read_int_env(env_prefix, "N_EMBD", d["n_embd"])
    bs = block_size if block_size is not None else read_int_env(env_prefix, "BLOCK_SIZE", d["block_size"])
    return nl, nh, ne, bs


def _resolve_device(env_prefix: str, device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    dev = os.environ.get(f"{env_prefix}DEVICE")
    if dev:
        return torch.device(dev)
    fallback = os.environ.get("DESSIN_LLM_DEVICE")
    if fallback:
        return torch.device(fallback)
    return torch.device("cpu")


def train_decoder_only_char_lm(
    variant: LLMVariant,
    *,
    data_seed: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    env_prefix: str,
    device: Optional[str] = None,
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
    block_size: Optional[int] = None,
    checkpoint_weights_hex: Optional[str] = None,
    tokenizer_kind: Optional[TokenizerKind] = None,
    tokenizer_profile: Optional[TokenizerProfile] = None,
) -> TrainingResult:
    """
    Train a character-level LM on the bundled Shakespeare excerpt.

    Hyperparameters default per ``variant`` and can be overridden by kwargs or
    ``{env_prefix}N_LAYER``, ``N_HEAD``, ``N_EMBD``, ``BLOCK_SIZE``, ``DEVICE``.
    
    If checkpoint_weights_hex is provided, loads those weights before training
    to continue from a previous checkpoint.

    Tokenization for the uploaded artifact is selected via ``tokenizer_kind`` /
    ``tokenizer_profile`` or env (see :func:`~dessin.models.llm_model_spec.resolve_char_lm_tokenizer_options`).
    Only ``TokenizerKind.CHARACTER`` is implemented in this trainer; other kinds raise.
    """
    device_t = _resolve_device(env_prefix, device)
    tk, tprof = resolve_char_lm_tokenizer_options(
        env_prefix,
        tokenizer_kind=tokenizer_kind,
        tokenizer_profile=tokenizer_profile,
    )
    if tk != TokenizerKind.CHARACTER:
        raise ValueError(
            f"train_decoder_only_char_lm supports only TokenizerKind.CHARACTER; got {tk!r}. "
            "Use a BPE-capable trainer when one is wired, or set DESSIN_LLM_TOKENIZER_KIND=character."
        )
    nl, nh, ne, bs = _resolve_hparams(
        variant, env_prefix, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size
    )

    torch.manual_seed(int(data_seed) % (2**31 - 1))

    tok = CharTokenizer(BUNDLED_SHAKESPEARE_CHAR_CORPUS)
    model = DecoderOnlyGPT(
        vocab_size=tok.vocab_size,
        block_size=bs,
        n_layer=nl,
        n_head=nh,
        n_embd=ne,
    ).to(device_t)
    
    # Load checkpoint weights if provided (continuous training across blocks)
    if checkpoint_weights_hex:
        try:
            load_params_from_hex(model, checkpoint_weights_hex)
            pretty_print("Loaded checkpoint weights into model", kind="ok")
        except Exception as e:
            pretty_print(f"Failed to load checkpoint weights: {e}. Starting from random init.", kind="warn")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    eval_seed = (int(data_seed) * 7919 + 1) % (2**31 - 1)
    loss_before = eval_lm_loss(model, tok, device_t, batch_size, bs, eval_seed)

    model.train()
    n = len(tok.data) - bs - 1
    if n <= 0:
        raise RuntimeError("Corpus too short for block_size")

    for step in range(training_steps):
        g = torch.Generator()
        g.manual_seed(int(data_seed) + step * 10007)
        ix = torch.randint(0, n, (batch_size,), generator=g)
        x = torch.stack([tok.data[i : i + bs] for i in ix]).to(device_t)
        y = torch.stack([tok.data[i + 1 : i + bs + 1] for i in ix]).to(device_t)
        _, loss = model(x, y)
        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    loss_after = eval_lm_loss(model, tok, device_t, batch_size, bs, eval_seed)

    weights_hex, pogo_shapes, nbytes = flatten_params_float64(model)

    spec = DecoderOnlyTransformerSpec(
        variant=variant,
        tokenizer_kind=tk,
        n_layer=nl,
        n_head=nh,
        n_embd=ne,
        block_size=bs,
        vocab_size=tok.vocab_size,
        tokenizer_profile=tprof,
        corpus_id="shakespeare_sonnet1_excerpt_x80",
    )
    arch = spec.to_architecture_dict(pogo_layer_shapes=[list(s) for s in pogo_shapes])

    tdata = {
        "kind": "character_lm",
        "text_hash_sha256": tok.text_hash(),
        "data_seed": int(data_seed),
        "samples_bound": int(n),
        "tokenizer_profile": tprof.value,
    }

    return TrainingResult(
        loss_before=loss_before,
        loss_after=loss_after,
        model_weights_hex=weights_hex,
        model_size_bytes=nbytes,
        training_steps=training_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        training_data_hash=tok.text_hash(),
        pogo_layer_shapes=pogo_shapes,
        model_architecture_json=arch,
        training_data_json=tdata,
    )
