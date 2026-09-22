"""
Transaction types for the DeSSIN blockchain network.
"""

import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from chaincraft.shared_message import SharedMessage


@dataclass
class BaseTransaction(ABC):
    """Base class for all DeSSIN transactions"""
    
    sender: str  # sender address
    fee: float  # transaction fee in DESSIN tokens
    timestamp: float  # transaction creation time
    public_key: str  # sender's public key for verification
    signature: str  # transaction signature
    tx_id: str  # transaction hash
    
    @abstractmethod
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get the core transaction data for signing"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseTransaction":
        """Create transaction from dictionary"""
        return cls(**data)


@dataclass
class TransferTransaction(BaseTransaction):
    """Standard token transfer transaction"""
    
    recipient: str  # recipient address
    amount: float  # amount to transfer
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "transfer",
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class UnstakeRequestTransaction(BaseTransaction):
    """Request exit from protocol stake; economic layer applies cooldown before liquid credit."""

    amount: float

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "unstake_request",
            "sender": self.sender,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


@dataclass
class ModelQueryTransaction(BaseTransaction):
    """Transaction for querying a model and paying per token"""
    
    model_id: str  # ID of the model to query
    query_data: str  # input query/prompt
    max_tokens: int  # maximum tokens to generate
    token_price: float  # price per token in DESSIN
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "model_query",
            "sender": self.sender,
            "model_id": self.model_id,
            "query_data": self.query_data,
            "max_tokens": self.max_tokens,
            "token_price": self.token_price,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class ConditionalTransferTransaction(BaseTransaction):
    """Transaction that transfers tokens based on model output conditions"""
    
    recipient: str  # recipient address if condition is met
    amount: float  # amount to transfer
    model_id: str  # model to query for condition
    condition_query: str  # query to determine condition
    condition_expected: str  # expected output pattern for condition to be true
    fallback_recipient: Optional[str] = None  # recipient if condition fails
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "conditional_transfer",
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "model_id": self.model_id,
            "condition_query": self.condition_query,
            "condition_expected": self.condition_expected,
            "fallback_recipient": self.fallback_recipient,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class ModelUploadTransaction(BaseTransaction):
    """Transaction for uploading a new model to the network"""
    
    model_name: str  # human-readable model name
    model_hash: str  # hash of the model data
    model_size_gb: float  # size of the model in GB
    storage_blocks: int  # number of blocks to rent storage
    storage_payment: float  # total payment for storage rental
    ipfs_hash: Optional[str] = None  # IPFS hash once uploaded
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "model_upload",
            "sender": self.sender,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "model_size_gb": self.model_size_gb,
            "storage_blocks": self.storage_blocks,
            "storage_payment": self.storage_payment,
            "ipfs_hash": self.ipfs_hash,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class ModelForkTransaction(BaseTransaction):
    """Transaction for forking an existing model"""
    
    parent_model_id: str  # ID of the model being forked
    fork_name: str  # name for the forked model
    storage_blocks: int  # blocks to rent storage for fork
    storage_payment: float  # payment for fork storage
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "model_fork",
            "sender": self.sender,
            "parent_model_id": self.parent_model_id,
            "fork_name": self.fork_name,
            "storage_blocks": self.storage_blocks,
            "storage_payment": self.storage_payment,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class ModelTrainingDataRefreshTransaction(BaseTransaction):
    """Owner-only: register new training data, unfreeze training, pay fee + optional escrow top-up."""

    model_id: str
    training_data_manifest_hash: str
    training_escrow_topup: float = 0.0
    training_data_uri: Optional[str] = None
    dataset_name: Optional[str] = None

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "model_training_data_refresh",
            "sender": self.sender,
            "model_id": self.model_id,
            "training_data_manifest_hash": self.training_data_manifest_hash,
            "training_escrow_topup": self.training_escrow_topup,
            "training_data_uri": self.training_data_uri,
            "dataset_name": self.dataset_name,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


@dataclass
class AttestationTransaction(BaseTransaction):
    """Transaction for verifier attestations (positive/negative)"""
    
    block_hash: str  # hash of the block being attested
    attestation_type: str  # "positive" or "negative"
    verification_data: Dict[str, Any]  # details of verification performed
    evidence: Optional[str] = None  # evidence for negative attestations (e.g., Merkle proof mismatch)
    merkle_proof_verified: bool = False  # whether Merkle proofs were verified
    quantized_model_verified: bool = False  # whether quantized model was verified
    data_availability_verified: bool = False  # whether data was available for verification
    verification_timestamp: float = 0.0  # when verification was performed
    
    def get_transaction_data(self) -> Dict[str, Any]:
        """Get core data for signing"""
        return {
            "type": "attestation", 
            "sender": self.sender,
            "block_hash": self.block_hash,
            "attestation_type": self.attestation_type,
            "verification_data": self.verification_data,
            "evidence": self.evidence,
            "merkle_proof_verified": self.merkle_proof_verified,
            "quantized_model_verified": self.quantized_model_verified,
            "data_availability_verified": self.data_availability_verified,
            "verification_timestamp": self.verification_timestamp,
            "fee": self.fee,
            "timestamp": self.timestamp
        }
    
    def is_positive(self) -> bool:
        """Check if this is a positive attestation"""
        return self.attestation_type == "positive"
    
    def is_negative(self) -> bool:
        """Check if this is a negative attestation"""
        return self.attestation_type == "negative"
    
    def has_evidence(self) -> bool:
        """Check if this attestation has supporting evidence"""
        return self.evidence is not None and len(self.evidence) > 0


@dataclass
class PostTrainingTaskTransaction(BaseTransaction):
    """
    Submit a model-training task to the fee-market mempool.

    The ``deposit`` (= ``base_fee + tip``) is escrowed immediately; the
    block leader that includes this task earns the tip while the base fee is
    burned.  If the task expires or is cancelled the deposit is refunded minus
    a small expiry fee.
    """

    model_id: str           # catalog model to train
    deposit: float          # base_fee + tip (escrowed)
    base_fee: float         # minimum gas the owner will pay
    tip: float              # extra incentive for the miner leader
    cooldown_blocks: int    # min gap before this model can be re-queued
    max_slots: int          # compute slots requested (1 for standard tasks)
    expiry_block: int       # cancel if not scheduled by this height

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "post_training_task",
            "sender": self.sender,
            "model_id": self.model_id,
            "deposit": self.deposit,
            "base_fee": self.base_fee,
            "tip": self.tip,
            "cooldown_blocks": self.cooldown_blocks,
            "max_slots": self.max_slots,
            "expiry_block": self.expiry_block,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


@dataclass
class CancelTrainingTaskTransaction(BaseTransaction):
    """
    Owner-initiated cancellation of a pending training task.

    The deposit is refunded minus a small cancellation fee so the queue is
    not spammed with tasks the owner never intends to honour.
    """

    task_id: str   # SHA-256 task identifier from PostTrainingTaskTransaction

    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "cancel_training_task",
            "sender": self.sender,
            "task_id": self.task_id,
            "fee": self.fee,
            "timestamp": self.timestamp,
        }


class TransactionFactory:
    """Factory for creating and validating transactions"""
    
    TRANSACTION_TYPES = {
        "transfer": TransferTransaction,
        "unstake_request": UnstakeRequestTransaction,
        "model_query": ModelQueryTransaction,
        "conditional_transfer": ConditionalTransferTransaction,
        "model_upload": ModelUploadTransaction,
        "model_fork": ModelForkTransaction,
        "model_training_data_refresh": ModelTrainingDataRefreshTransaction,
        "attestation": AttestationTransaction,
        "post_training_task": PostTrainingTaskTransaction,
        "cancel_training_task": CancelTrainingTaskTransaction,
    }
    
    @classmethod
    def create_transaction(cls, tx_type: str, **kwargs) -> BaseTransaction:
        """Create a transaction of the specified type"""
        if tx_type not in cls.TRANSACTION_TYPES:
            raise ValueError(f"Unknown transaction type: {tx_type}")
            
        # Add common fields if not provided
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = time.time()
            
        return cls.TRANSACTION_TYPES[tx_type](**kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseTransaction:
        """Create transaction from dictionary data"""
        tx_data = data.get("transaction_data", {})
        tx_type = tx_data.get("type")
        
        if tx_type not in cls.TRANSACTION_TYPES:
            raise ValueError(f"Unknown transaction type: {tx_type}")
            
        # Remove type field and create transaction
        tx_data_copy = tx_data.copy()
        tx_data_copy.pop("type", None)
        
        return cls.TRANSACTION_TYPES[tx_type](**data)
    
    @classmethod
    def sign_transaction(cls, transaction: BaseTransaction, private_key: str) -> str:
        """Sign a transaction with private key"""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chaincraft", "examples"))
        from blockchain import BlockchainUtils
        
        tx_data = transaction.get_transaction_data()
        return BlockchainUtils.sign_transaction(tx_data, private_key)
    
    @classmethod
    def verify_transaction(cls, transaction: BaseTransaction) -> bool:
        """Verify a transaction signature"""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chaincraft", "examples"))
        from blockchain import BlockchainUtils
        
        tx_data = transaction.get_transaction_data()
        return BlockchainUtils.verify_signature(
            tx_data, transaction.signature, transaction.public_key
        )
    
    @classmethod
    def calculate_tx_id(cls, transaction: BaseTransaction) -> str:
        """Calculate transaction ID from transaction data"""
        tx_data = transaction.get_transaction_data()
        tx_data["signature"] = transaction.signature
        tx_data["public_key"] = transaction.public_key
        
        tx_str = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_str.encode()).hexdigest()
    
    @classmethod
    def create_signed_transaction(
        cls, 
        tx_type: str, 
        sender: str,
        private_key: str, 
        public_key: str,
        **kwargs
    ) -> BaseTransaction:
        """Create and sign a transaction in one step"""
        # Create the transaction
        tx = cls.create_transaction(
            tx_type, 
            sender=sender, 
            public_key=public_key,
            signature="",  # temporary
            tx_id="",  # temporary
            **kwargs
        )
        
        # Sign the transaction
        signature = cls.sign_transaction(tx, private_key)
        tx.signature = signature
        
        # Calculate transaction ID
        tx.tx_id = cls.calculate_tx_id(tx)
        
        return tx
