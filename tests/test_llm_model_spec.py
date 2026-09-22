"""Serialization of standardized LLM architecture specs (no torch)."""

from __future__ import annotations

import pytest

from dessin.models.llm_model_spec import (
    ARCHITECTURE_SCHEMA_VERSION,
    DecoderOnlyTransformerSpec,
    LLMVariant,
    NeuralArchitectureFamily,
    TokenizerKind,
    TokenizerProfile,
    parse_architecture_dict,
    resolve_char_lm_tokenizer_options,
)
from dessin.models.model_tokenization_ops import change_model_tokenization_stub


def test_decoder_only_spec_roundtrip():
    spec = DecoderOnlyTransformerSpec(
        variant=LLMVariant.GPT2_NANO,
        tokenizer_kind=TokenizerKind.CHARACTER,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=67,
    )
    d = spec.to_architecture_dict(pogo_layer_shapes=[[10], [20]])
    assert d["schema_version"] == ARCHITECTURE_SCHEMA_VERSION
    assert d["family"] == NeuralArchitectureFamily.DECODER_ONLY_TRANSFORMER.value
    assert d["variant"] == LLMVariant.GPT2_NANO.value
    assert d["hyperparameters"]["vocab_size"] == 67
    assert d["tokenizer"]["kind"] == TokenizerKind.CHARACTER.value
    assert d["tokenizer"]["profile"] == TokenizerProfile.FLAT_CORPUS.value
    back = parse_architecture_dict(d)
    assert back.variant == LLMVariant.GPT2_NANO
    assert back.vocab_size == 67
    assert back.tokenizer_profile == TokenizerProfile.FLAT_CORPUS


def test_decoder_spec_tokenizer_profile_roundtrip():
    spec = DecoderOnlyTransformerSpec(
        variant=LLMVariant.MICRO_GPT_CHAR,
        tokenizer_kind=TokenizerKind.CHARACTER,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        vocab_size=65,
        tokenizer_profile=TokenizerProfile.SHAKESPEARE_PER_CHAT,
    )
    d = spec.to_architecture_dict(pogo_layer_shapes=[[1]])
    assert d["tokenizer"]["profile"] == TokenizerProfile.SHAKESPEARE_PER_CHAT.value
    back = parse_architecture_dict(d)
    assert back.tokenizer_profile == TokenizerProfile.SHAKESPEARE_PER_CHAT


def test_parse_unknown_tokenizer_profile_defaults_flat():
    d = {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "family": NeuralArchitectureFamily.DECODER_ONLY_TRANSFORMER.value,
        "variant": LLMVariant.MICRO_GPT_CHAR.value,
        "tokenizer": {"kind": "character", "profile": "not_a_real_profile_yet"},
        "hyperparameters": {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 32,
            "vocab_size": 65,
        },
        "corpus": "x",
        "pogo_layer_shapes": [[1]],
    }
    spec = parse_architecture_dict(d)
    assert spec.tokenizer_profile == TokenizerProfile.FLAT_CORPUS


def test_change_model_tokenization_stub_not_applied():
    r = change_model_tokenization_stub(
        "m1",
        tokenizer_kind=TokenizerKind.CHARACTER,
        tokenizer_profile=TokenizerProfile.SHAKESPEARE_PER_CHAT,
    )
    assert r.applied is False
    assert "Not implemented" in r.message


def test_resolve_char_lm_tokenizer_options_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("DESSIN_LLM_TOKENIZER_PROFILE", "flat_corpus")
    tk, tp = resolve_char_lm_tokenizer_options(
        "DESSIN_MICRO_GPT_",
        tokenizer_kind=TokenizerKind.CHARACTER,
        tokenizer_profile=TokenizerProfile.SHAKESPEARE_PER_CHAT,
    )
    assert tp == TokenizerProfile.SHAKESPEARE_PER_CHAT
    assert tk == TokenizerKind.CHARACTER


def test_train_decoder_char_lm_rejects_bpe_tokenizer_kind():
    pytest.importorskip("torch")
    from dessin.llm.decoder_lm_training import train_decoder_only_char_lm

    with pytest.raises(ValueError, match=r"supports only TokenizerKind\.CHARACTER"):
        train_decoder_only_char_lm(
            LLMVariant.MICRO_GPT_CHAR,
            data_seed=0,
            training_steps=1,
            batch_size=1,
            learning_rate=0.001,
            env_prefix="DESSIN_MICRO_GPT_",
            tokenizer_kind=TokenizerKind.BPE_GPT2,
        )


def test_parse_legacy_micro_gpt_payload():
    legacy = {
        "type": "micro_gpt_char",
        "vocab_size": 65,
        "block_size": 32,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "corpus": "shakespeare_sonnet1_excerpt_x80",
        "pogo_layer_shapes": [[1]],
    }
    spec = parse_architecture_dict(legacy)
    assert spec.variant == LLMVariant.MICRO_GPT_CHAR
    assert spec.n_embd == 64
