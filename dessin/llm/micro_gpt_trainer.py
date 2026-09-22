#!/usr/bin/env python3
"""
Tiny character-level GPT preset (CPU-friendly defaults).

Used when ``DESSIN_MICRO_GPT_TRAINING=1`` for real LM loss during PoGO mining / E2E.
Architecture matches GPT-2 / nanoGPT decoder stack; see :mod:`dessin.models.llm_model_spec`.
"""

from __future__ import annotations

from typing import Optional

from .decoder_only_gpt import CharTokenizer, DecoderOnlyGPT, TinyCharGPT
from .decoder_lm_training import train_decoder_only_char_lm
from ..models.llm_model_spec import LLMVariant, TokenizerKind, TokenizerProfile

# Re-export shared modules for callers that imported symbols from here.
__all__ = [
    "CharTokenizer",
    "DecoderOnlyGPT",
    "TinyCharGPT",
    "train_micro_gpt",
]


def train_micro_gpt(
    *,
    data_seed: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    device: Optional[str] = None,
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
    block_size: Optional[int] = None,
    checkpoint_weights_hex: Optional[str] = None,
    tokenizer_kind: Optional[TokenizerKind] = None,
    tokenizer_profile: Optional[TokenizerProfile] = None,
):
    """Train micro-GPT char LM; hyperparameters from kwargs or ``DESSIN_MICRO_GPT_*`` env.
    
    If checkpoint_weights_hex is provided, loads those weights before training
    to continue from a previous checkpoint.
    """
    return train_decoder_only_char_lm(
        LLMVariant.MICRO_GPT_CHAR,
        data_seed=data_seed,
        training_steps=training_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        env_prefix="DESSIN_MICRO_GPT_",
        device=device,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        checkpoint_weights_hex=checkpoint_weights_hex,
        tokenizer_kind=tokenizer_kind,
        tokenizer_profile=tokenizer_profile,
    )
