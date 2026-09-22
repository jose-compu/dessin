"""Future chain/registry operations for model tokenization (stubs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_model_spec import TokenizerKind, TokenizerProfile


@dataclass(frozen=True)
class ChangeTokenizationResult:
    """Outcome of a (future) on-chain or registry tokenization update."""

    applied: bool
    message: str


def change_model_tokenization_stub(
    model_id: str,
    *,
    tokenizer_kind: Optional[TokenizerKind] = None,
    tokenizer_profile: Optional[TokenizerProfile] = None,
) -> ChangeTokenizationResult:
    """
    Reserved for a consensus transaction that rewires ``model_architecture_json``
    (and dependent manifests) after upload. Upload-time choice is persisted today;
    mutating it in-flight is not implemented.
    """
    _ = (model_id, tokenizer_kind, tokenizer_profile)
    return ChangeTokenizationResult(
        applied=False,
        message=(
            "Not implemented: changing tokenization after upload requires a dedicated "
            "transaction, manifest bump, and miner agreement on the new tokenizer block "
            "(see dessin.models.llm_model_spec / torrent model_architecture)."
        ),
    )
