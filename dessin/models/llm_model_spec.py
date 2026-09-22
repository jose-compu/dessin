"""
Standard JSON-serializable specs for LLM-style models in DeSSIN (PoGO blocks, torrent metadata).

Use :class:`DecoderOnlyTransformerSpec` for GPT-2 / nanoGPT-style decoder stacks. Larger families
(e.g. GPT-3-class) can extend the same schema with extra ``hyperparameters`` keys without changing
``family`` — implement trainers separately when added.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

ARCHITECTURE_SCHEMA_VERSION = "1.0"


class NeuralArchitectureFamily(str, Enum):
    """High-level topology (maps cleanly to implementation modules)."""

    DECODER_ONLY_TRANSFORMER = "decoder_only_transformer"


class LLMVariant(str, Enum):
    """Concrete recipe / preset within a family (training defaults + naming)."""

    MICRO_GPT_CHAR = "micro_gpt_char"
    GPT2_NANO = "gpt2_nano"


class TokenizerKind(str, Enum):
    CHARACTER = "character"
    BPE_GPT2 = "bpe_gpt2"


class TokenizerProfile(str, Enum):
    """
    How text is framed for training/inference on top of :class:`TokenizerKind`.

    ``flat_corpus`` matches today's char-LM loop (contiguous Shakespeare excerpt).
    ``shakespeare_per_chat`` is reserved for turn-based / role-delimited framing on the
    same character vocabulary (nanochat-style); training still uses the flat excerpt
    until a per-chat corpus path exists—value is stored on the uploaded architecture
    so nodes agree on intended inference formatting.
    """

    FLAT_CORPUS = "flat_corpus"
    SHAKESPEARE_PER_CHAT = "shakespeare_per_chat"


_DEFAULT_IMPLEMENTATION = "dessin.llm.decoder_only_gpt.DecoderOnlyGPT"


@dataclass(frozen=True)
class DecoderOnlyTransformerSpec:
    """
    GPT-2-class decoder (pre-norm, causal self-attention, GeLU MLP) — same stack as nanoGPT.

    ``variant`` distinguishes presets (tiny CPU micro-GPT vs. larger ``gpt2_nano`` defaults).
    """

    variant: LLMVariant
    tokenizer_kind: TokenizerKind
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    tokenizer_profile: TokenizerProfile = TokenizerProfile.FLAT_CORPUS
    corpus_id: str = "shakespeare_sonnet1_excerpt_x80"
    activation: str = "gelu"
    norm: str = "layernorm"
    mlp_ratio: float = 4.0

    def to_architecture_dict(
        self,
        *,
        pogo_layer_shapes: List[List[int]],
        implementation: str = _DEFAULT_IMPLEMENTATION,
    ) -> Dict[str, Any]:
        return {
            "schema_version": ARCHITECTURE_SCHEMA_VERSION,
            "family": NeuralArchitectureFamily.DECODER_ONLY_TRANSFORMER.value,
            "variant": self.variant.value,
            "type": self.variant.value,
            "implementation": implementation,
            "tokenizer": {
                "kind": self.tokenizer_kind.value,
                "profile": self.tokenizer_profile.value,
            },
            "hyperparameters": {
                "n_layer": self.n_layer,
                "n_head": self.n_head,
                "n_embd": self.n_embd,
                "block_size": self.block_size,
                "vocab_size": self.vocab_size,
                "activation": self.activation,
                "norm": self.norm,
                "mlp_ratio": self.mlp_ratio,
            },
            "corpus": self.corpus_id,
            "pogo_layer_shapes": list(pogo_layer_shapes),
        }


def parse_architecture_dict(data: Dict[str, Any]) -> DecoderOnlyTransformerSpec:
    """Rebuild a spec from ``model_architecture_json`` (schema v1 or legacy ``type`` rows)."""
    if data.get("schema_version") == ARCHITECTURE_SCHEMA_VERSION:
        hp = data.get("hyperparameters") or {}
        tok = data.get("tokenizer") or {}
        prof_raw = (tok.get("profile") or "flat_corpus").strip().lower()
        try:
            tprof = TokenizerProfile(prof_raw)
        except ValueError:
            tprof = TokenizerProfile.FLAT_CORPUS
        return DecoderOnlyTransformerSpec(
            variant=LLMVariant(data["variant"]),
            tokenizer_kind=TokenizerKind(tok.get("kind", "character")),
            n_layer=int(hp["n_layer"]),
            n_head=int(hp["n_head"]),
            n_embd=int(hp["n_embd"]),
            block_size=int(hp["block_size"]),
            vocab_size=int(hp["vocab_size"]),
            tokenizer_profile=tprof,
            corpus_id=str(data.get("corpus", "unknown")),
            activation=str(hp.get("activation", "gelu")),
            norm=str(hp.get("norm", "layernorm")),
            mlp_ratio=float(hp.get("mlp_ratio", 4.0)),
        )

    legacy_type = data.get("type") or data.get("variant")
    if legacy_type == LLMVariant.MICRO_GPT_CHAR.value:
        return DecoderOnlyTransformerSpec(
            variant=LLMVariant.MICRO_GPT_CHAR,
            tokenizer_kind=TokenizerKind.CHARACTER,
            n_layer=int(data["n_layer"]),
            n_head=int(data["n_head"]),
            n_embd=int(data["n_embd"]),
            block_size=int(data["block_size"]),
            vocab_size=int(data["vocab_size"]),
            tokenizer_profile=TokenizerProfile.FLAT_CORPUS,
            corpus_id=str(data.get("corpus", "unknown")),
        )
    if legacy_type == LLMVariant.GPT2_NANO.value:
        return DecoderOnlyTransformerSpec(
            variant=LLMVariant.GPT2_NANO,
            tokenizer_kind=TokenizerKind.CHARACTER,
            n_layer=int(data["n_layer"]),
            n_head=int(data["n_head"]),
            n_embd=int(data["n_embd"]),
            block_size=int(data["block_size"]),
            vocab_size=int(data["vocab_size"]),
            tokenizer_profile=TokenizerProfile.FLAT_CORPUS,
            corpus_id=str(data.get("corpus", "unknown")),
        )

    raise ValueError(f"Unsupported architecture payload: keys={sorted(data.keys())}")


def variant_default_hyperparams(variant: LLMVariant) -> Dict[str, int]:
    """nanoGPT-scale defaults: micro-GPT for CI; gpt2_nano is a larger same-arch preset."""
    if variant == LLMVariant.MICRO_GPT_CHAR:
        return {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 32}
    if variant == LLMVariant.GPT2_NANO:
        return {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 128}
    raise ValueError(variant)


def read_int_env(env_prefix: str, key: str, default: int) -> int:
    """Read ``{prefix}{KEY}`` e.g. DESSIN_MICRO_GPT_N_LAYER."""
    return int(os.environ.get(f"{env_prefix}{key}", str(default)))


def resolve_char_lm_tokenizer_options(
    env_prefix: str,
    *,
    tokenizer_kind: Optional[TokenizerKind] = None,
    tokenizer_profile: Optional[TokenizerProfile] = None,
) -> Tuple[TokenizerKind, TokenizerProfile]:
    """
    Tokenization choice for an LM training / torrent upload.

    Explicit kwargs win. Otherwise reads ``DESSIN_LLM_TOKENIZER_KIND`` and
    ``DESSIN_LLM_TOKENIZER_PROFILE``, then ``{env_prefix}TOKENIZER_KIND`` /
    ``{env_prefix}TOKENIZER_PROFILE`` (e.g. ``DESSIN_MICRO_GPT_TOKENIZER_PROFILE``).
    """
    if tokenizer_kind is not None:
        tk = tokenizer_kind
    else:
        raw = (
            os.environ.get("DESSIN_LLM_TOKENIZER_KIND")
            or os.environ.get(f"{env_prefix}TOKENIZER_KIND")
            or ""
        ).strip().lower()
        if raw:
            try:
                tk = TokenizerKind(raw)
            except ValueError:
                tk = TokenizerKind.CHARACTER
        else:
            tk = TokenizerKind.CHARACTER

    if tokenizer_profile is not None:
        tp = tokenizer_profile
    else:
        rawp = (
            os.environ.get("DESSIN_LLM_TOKENIZER_PROFILE")
            or os.environ.get(f"{env_prefix}TOKENIZER_PROFILE")
            or ""
        ).strip().lower()
        if rawp:
            try:
                tp = TokenizerProfile(rawp)
            except ValueError:
                tp = TokenizerProfile.FLAT_CORPUS
        else:
            tp = TokenizerProfile.FLAT_CORPUS

    return tk, tp
