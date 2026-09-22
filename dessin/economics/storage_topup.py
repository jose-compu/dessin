"""
Storage Top-Up System
=====================

Allows anyone to top-up storage for a model, not just the owner.
Only owners can manage model parameters and training datasets.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class StorageTopUp:
    """Represents a storage top-up payment"""
    model_id: str
    payer_address: str
    amount_paid: float
    blocks_added: int
    new_expiration_block: int
    timestamp: float
    tx_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "payer_address": self.payer_address,
            "amount_paid": self.amount_paid,
            "blocks_added": self.blocks_added,
            "new_expiration_block": self.new_expiration_block,
            "timestamp": self.timestamp,
            "tx_hash": self.tx_hash
        }


class StorageTopUpManager:
    """Manages storage top-ups for models"""
    
    def __init__(self):
        """Initialize storage top-up manager"""
        # Track all top-ups: model_id -> List[StorageTopUp]
        self.topup_history: Dict[str, List[StorageTopUp]] = {}
        
        # Track contributors: model_id -> {address: total_contribution}
        self.contributors: Dict[str, Dict[str, float]] = {}
        
        print("✓ Storage top-up manager initialized")
    
    def topup_storage(
        self,
        model_id: str,
        model_owner: str,
        payer_address: str,
        model_size_gb: float,
        additional_blocks: int,
        payment: float,
        current_expiration: int,
        storage_price_per_gb_per_block: float,
        tx_hash: str = ""
    ) -> Optional[StorageTopUp]:
        """
        Top-up storage for a model (can be done by anyone)
        
        Args:
            model_id: Model to top-up
            model_owner: Owner of the model
            payer_address: Address paying for top-up
            model_size_gb: Model size in GB
            additional_blocks: Number of blocks to add
            payment: Payment amount
            current_expiration: Current expiration block
            storage_price_per_gb_per_block: Current storage price
            tx_hash: Transaction hash
        
        Returns:
            StorageTopUp record if successful
        """
        # Calculate required payment
        required_payment = model_size_gb * additional_blocks * storage_price_per_gb_per_block
        
        if payment < required_payment:
            print(f"⚠️  Insufficient payment for storage top-up")
            print(f"   Required: {required_payment:.4f} DESSIN")
            print(f"   Provided: {payment:.4f} DESSIN")
            return None
        
        # Create top-up record
        new_expiration = current_expiration + additional_blocks
        
        topup = StorageTopUp(
            model_id=model_id,
            payer_address=payer_address,
            amount_paid=payment,
            blocks_added=additional_blocks,
            new_expiration_block=new_expiration,
            timestamp=time.time(),
            tx_hash=tx_hash
        )
        
        # Record in history
        if model_id not in self.topup_history:
            self.topup_history[model_id] = []
        self.topup_history[model_id].append(topup)
        
        # Track contributor
        if model_id not in self.contributors:
            self.contributors[model_id] = {}
        
        self.contributors[model_id][payer_address] = \
            self.contributors[model_id].get(payer_address, 0.0) + payment
        
        # Log the top-up
        if payer_address == model_owner:
            print(f"💾 Owner topped up storage for model {model_id[:16]}...")
        else:
            print(f"💾 Storage topped up for model {model_id[:16]}...")
            print(f"   Payer: {payer_address[:20]}... (not owner)")
        
        print(f"   Amount: {payment:.4f} DESSIN")
        print(f"   Blocks added: {additional_blocks}")
        print(f"   New expiration: block {new_expiration}")
        
        return topup
    
    def get_topup_history(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[StorageTopUp]:
        """
        Get top-up history for a model
        
        Args:
            model_id: Model ID
            limit: Maximum number of records
        
        Returns:
            List of top-ups, most recent first
        """
        if model_id not in self.topup_history:
            return []
        
        history = self.topup_history[model_id][::-1]  # Reverse for most recent first
        return history[:limit]
    
    def get_contributors(
        self,
        model_id: str
    ) -> Dict[str, float]:
        """
        Get all contributors and their total contributions for a model
        
        Args:
            model_id: Model ID
        
        Returns:
            Dict mapping addresses to total contributions
        """
        return self.contributors.get(model_id, {})
    
    def get_total_topup_amount(
        self,
        model_id: str
    ) -> float:
        """
        Get total amount topped up for a model
        
        Args:
            model_id: Model ID
        
        Returns:
            Total amount in DESSIN
        """
        if model_id not in self.topup_history:
            return 0.0
        
        return sum(topup.amount_paid for topup in self.topup_history[model_id])
    
    def get_user_contribution(
        self,
        model_id: str,
        user_address: str
    ) -> float:
        """
        Get a specific user's contribution to a model's storage
        
        Args:
            model_id: Model ID
            user_address: User address
        
        Returns:
            Total contribution in DESSIN
        """
        if model_id not in self.contributors:
            return 0.0
        
        return self.contributors[model_id].get(user_address, 0.0)
    
    def get_storage_statistics(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics for a model
        
        Args:
            model_id: Model ID
        
        Returns:
            Dict with statistics
        """
        if model_id not in self.topup_history:
            return {
                "model_id": model_id,
                "total_topups": 0,
                "total_amount": 0.0,
                "total_blocks_added": 0,
                "unique_contributors": 0,
                "contributors": {}
            }
        
        history = self.topup_history[model_id]
        contributors = self.contributors[model_id]
        
        return {
            "model_id": model_id,
            "total_topups": len(history),
            "total_amount": sum(t.amount_paid for t in history),
            "total_blocks_added": sum(t.blocks_added for t in history),
            "unique_contributors": len(contributors),
            "contributors": contributors.copy(),
            "latest_topup": history[-1].to_dict() if history else None
        }
