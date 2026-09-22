"""
Model Submission & Training Data Transactions
=============================================

Extends the core transaction layer with three new transaction types that
implement the end-to-end model-registration and data-upload workflow:

RegisterModelTransaction
    Posts a :class:`~dessin.models.model_format_registry.ModelSubmissionSpec`
    on-chain.  The ``submission_spec_json`` field carries the full JSON
    manifest; ``content_hash`` lets verifiers confirm integrity without
    downloading the weights.  ``required_deposit`` is the DESSIN amount locked
    in escrow to cover the first training cycle (estimated by
    :class:`~dessin.economics.compute_fee_schedule.ComputeFeeSchedule`).

UploadTrainingDataTransaction
    Registers a training-data file (dataset / JSONL / text corpus) on-chain
    together with a *storage lease commitment*.  The ``storage_payment`` field
    locks DESSIN for ``storage_blocks`` blocks; nodes will pin the data until
    the lease expires.

RenewStorageLeaseTransaction
    Extends an existing storage lease by ``additional_blocks`` more blocks.
    Can be submitted by anyone (not just the owner) so communities can
    collectively fund datasets they depend on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..consensus.transactions import BaseTransaction


# ---------------------------------------------------------------------------
# RegisterModelTransaction
# ---------------------------------------------------------------------------

@dataclass
class RegisterModelTransaction(BaseTransaction):
    """
    Register a new model (all formats: GGUF, MLX, SafeTensors, …) on DeSSIN.

    Fields
    ------
    model_name:
        Human-readable name; must be unique per owner address.
    family:
        :class:`~dessin.models.model_format_registry.ModelFamily` value
        (plain string for chain portability).
    format:
        :class:`~dessin.models.model_format_registry.ModelFormat` value.
    size_mb:
        Weight file size in megabytes (used for fee heuristics).
    parameter_count_millions:
        Model parameter count in millions (0 if unknown).
    content_hash:
        SHA-256 of the :class:`~dessin.models.model_format_registry.ModelSubmissionSpec`
        canonical JSON (excluding mutable ``ipfs_hash`` / ``torrent_magnet``).
    submission_spec_json:
        Full serialised :class:`~dessin.models.model_format_registry.ModelSubmissionSpec`
        as a JSON string.  Nodes validate the hash before accepting.
    required_deposit:
        DESSIN locked for the first training cycle (covers base_fee × first
        pipeline stage).
    ipfs_hash:
        Optional; populated once weights are pinned on IPFS.
    torrent_magnet:
        Optional; populated once a torrent swarm exists.
    storage_blocks:
        How many blocks to rent weight storage for (0 = indefinite, not
        recommended).
    storage_payment:
        DESSIN paid for weight storage rental.
    license:
        SPDX license identifier.
    tags:
        Free-form discovery tags.
    """

    model_name: str
    family: str                         # ModelFamily.value
    format: str                         # ModelFormat.value
    size_mb: float
    parameter_count_millions: float
    content_hash: str
    submission_spec_json: str           # JSON of ModelSubmissionSpec
    required_deposit: float
    storage_blocks: int
    storage_payment: float
    ipfs_hash: str = ""
    torrent_magnet: str = ""
    license: str = "apache-2.0"
    tags: Optional[List[str]] = None

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "register_model",
            "sender": self.sender,
            "model_name": self.model_name,
            "family": self.family,
            "format": self.format,
            "size_mb": self.size_mb,
            "parameter_count_millions": self.parameter_count_millions,
            "content_hash": self.content_hash,
            "required_deposit": self.required_deposit,
            "storage_blocks": self.storage_blocks,
            "storage_payment": self.storage_payment,
            "ipfs_hash": self.ipfs_hash,
            "torrent_magnet": self.torrent_magnet,
            "license": self.license,
            "tags": self.tags or [],
            "fee": self.fee,
            "timestamp": self.timestamp,
        }

    def parsed_spec(self):
        """
        Deserialise ``submission_spec_json`` into a
        :class:`~dessin.models.model_format_registry.ModelSubmissionSpec`.
        """
        from .model_format_registry import ModelSubmissionSpec
        return ModelSubmissionSpec.from_dict(json.loads(self.submission_spec_json))

    def verify_content_hash(self) -> bool:
        """Return ``True`` if ``content_hash`` matches ``submission_spec_json``."""
        from .model_format_registry import ModelSubmissionSpec
        spec = ModelSubmissionSpec.from_dict(json.loads(self.submission_spec_json))
        return spec.content_hash() == self.content_hash


# ---------------------------------------------------------------------------
# UploadTrainingDataTransaction
# ---------------------------------------------------------------------------

@dataclass
class UploadTrainingDataTransaction(BaseTransaction):
    """
    Upload a training dataset and lock a storage lease.

    When this transaction is included in a block, the protocol:

    1. Creates a :class:`~dessin.economics.storage_lease.StorageLease` for
       ``file_id`` paid through ``current_block + storage_blocks``.
    2. Debits ``storage_payment`` from the sender.
    3. Associates the dataset with the model (via ``model_id``).

    If ``storage_payment`` runs out (expiry) and no renewal arrives, nodes
    start the grace-period countdown and eventually delete the data.

    Fields
    ------
    file_id:
        Unique dataset identifier (SHA-256 of file content recommended).
    model_id:
        The model this data is associated with (may be empty for shared
        public datasets).
    file_name:
        Original filename for display.
    file_hash:
        SHA-256 of the raw file bytes.
    size_mb:
        Dataset size in megabytes (drives storage_payment calculation).
    dataset_type:
        Semantic label: ``"pretrain"``, ``"sft"``, ``"dpo"``, ``"rlhf"``,
        ``"eval"``, or ``"custom"``.
    storage_blocks:
        Number of blocks to rent storage for.
    storage_payment:
        DESSIN escrowed for the storage lease (must be ≥ estimated cost).
    torrent_hash / magnet_link:
        Optional content-addressed distribution handles.
    description / tags:
        Discovery metadata.
    """

    file_id: str
    model_id: str
    file_name: str
    file_hash: str
    size_mb: float
    dataset_type: str               # "pretrain", "sft", "dpo", "rlhf", "eval", "custom"
    storage_blocks: int
    storage_payment: float          # DESSIN escrowed for storage lease
    torrent_hash: str = ""
    magnet_link: str = ""
    description: str = ""
    tags: Optional[List[str]] = None

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "upload_training_data",
            "sender": self.sender,
            "file_id": self.file_id,
            "model_id": self.model_id,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "size_mb": self.size_mb,
            "dataset_type": self.dataset_type,
            "storage_blocks": self.storage_blocks,
            "storage_payment": self.storage_payment,
            "torrent_hash": self.torrent_hash,
            "magnet_link": self.magnet_link,
            "description": self.description,
            "tags": self.tags or [],
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# RenewStorageLeaseTransaction
# ---------------------------------------------------------------------------

@dataclass
class RenewStorageLeaseTransaction(BaseTransaction):
    """
    Extend an existing storage lease.

    Submitted by *anyone* — the owner, a community donor, or a DAO treasury.
    The ``payment`` must cover ``additional_blocks × per-MB rate × size_mb``.

    If the lease has already reached ``DELETION_ELIGIBLE`` state, this
    transaction is rejected (data may already be gone).  Renewals are
    accepted during the ``WARNING`` and ``EXPIRED`` (grace) windows.

    Fields
    ------
    file_id:
        Lease to renew.
    additional_blocks:
        How many more blocks to extend the lease by.
    payment:
        DESSIN paid; must be ≥ ``estimate_storage_renewal_fee(size_mb, additional_blocks)``.
    """

    file_id: str
    additional_blocks: int
    payment: float

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "renew_storage_lease",
            "sender": self.sender,
            "file_id": self.file_id,
            "additional_blocks": self.additional_blocks,
            "payment": self.payment,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }
