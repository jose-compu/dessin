#!/usr/bin/env python3
"""
GPT-2-class decoder (nanoGPT-compatible) with larger default widths than :mod:`dessin.llm.micro_gpt_trainer`.

Select via ``DESSIN_LLM_VARIANT=gpt2_nano`` with ``DESSIN_MICRO_GPT_TRAINING=1``, or call
:func:`train_gpt2_nano` directly. Same Shakespeare char corpus until BPE tooling lands.
"""

from __future__ import annotations

from typing import Optional

from .decoder_lm_training import train_decoder_only_char_lm
from ..models.llm_model_spec import LLMVariant, TokenizerKind, TokenizerProfile

__all__ = ["train_gpt2_nano"]


def train_gpt2_nano(
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
    """Train with ``gpt2_nano`` preset; overrides via kwargs or ``DESSIN_GPT2_*`` env.
    
    If checkpoint_weights_hex is provided, loads those weights before training
    to continue from a previous checkpoint.
    """
    return train_decoder_only_char_lm(
        LLMVariant.GPT2_NANO,
        data_seed=data_seed,
        training_steps=training_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        env_prefix="DESSIN_GPT2_",
        device=device,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        checkpoint_weights_hex=checkpoint_weights_hex,
        tokenizer_kind=tokenizer_kind,
        tokenizer_profile=tokenizer_profile,
    )
