"""
Model Lifecycle Transactions for DeSSIN
========================================

Blockchain transactions for managing model lifecycle:
- Deprecate models
- Pause/Resume training
- Withdraw remaining fees
- Update model status
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..consensus.transactions import BaseTransaction


@dataclass
class DeprecateModelTransaction(BaseTransaction):
    """Transaction to deprecate a model"""
    
    model_id: str
    reason: str = "Model deprecated by owner"
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "deprecate_model",
            "sender": self.sender,
            "model_id": self.model_id,
            "reason": self.reason,
            "fee": self.fee,
            "timestamp": self.timestamp
        }
    
    def validate(self) -> bool:
        """Validate deprecation transaction"""
        if not self.model_id:
            return False
        if len(self.reason) > 256:
            return False
        return True


@dataclass
class PauseModelTransaction(BaseTransaction):
    """Transaction to pause model training"""
    
    model_id: str
    pause: bool = True  # True = pause, False = resume
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "pause_model",
            "sender": self.sender,
            "model_id": self.model_id,
            "pause": self.pause,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class WithdrawModelBalanceTransaction(BaseTransaction):
    """Transaction to withdraw remaining model balance"""
    
    model_id: str
    withdrawal_address: Optional[str] = None  # If None, send to sender
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "withdraw_model_balance",
            "sender": self.sender,
            "model_id": self.model_id,
            "withdrawal_address": self.withdrawal_address or self.sender,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


class ModelLifecycleManager:
    """Manages model lifecycle operations on the blockchain"""
    
    def __init__(self, model_manager, training_scheduler, economic_system):
        self.model_manager = model_manager
        self.training_scheduler = training_scheduler
        self.economic_system = economic_system
        
        print("✓ Model lifecycle manager initialized")
    
    def process_deprecate_transaction(
        self, 
        transaction: DeprecateModelTransaction
    ) -> bool:
        """Process a model deprecation transaction"""
        
        # Verify ownership
        model_info = self.model_manager.get_model_info(transaction.model_id)
        if not model_info:
            print(f"Model not found: {transaction.model_id}")
            return False
        
        if model_info.owner != transaction.sender:
            print(f"Not owner: {transaction.sender} cannot deprecate {transaction.model_id}")
            return False
        
        # Deprecate in scheduler
        success = self.training_scheduler.deprecate_model(
            transaction.model_id,
            reason=transaction.reason
        )
        
        if success:
            print(f"✓ Model {transaction.model_id[:20]}... deprecated on blockchain")
        
        return success
    
    def process_pause_transaction(
        self, 
        transaction: PauseModelTransaction
    ) -> bool:
        """Process a pause/resume transaction"""
        
        # Verify ownership
        model_info = self.model_manager.get_model_info(transaction.model_id)
        if not model_info:
            print(f"Model not found: {transaction.model_id}")
            return False
        
        if model_info.owner != transaction.sender:
            print(f"Not owner: {transaction.sender} cannot pause {transaction.model_id}")
            return False
        
        # Pause or resume
        if transaction.pause:
            success = self.training_scheduler.pause_model(transaction.model_id)
        else:
            success = self.training_scheduler.resume_model(transaction.model_id)
        
        if success:
            action = "paused" if transaction.pause else "resumed"
            print(f"✓ Model {transaction.model_id[:20]}... {action} on blockchain")
        
        return success
    
    def process_withdraw_transaction(
        self, 
        transaction: WithdrawModelBalanceTransaction
    ) -> bool:
        """Process a balance withdrawal transaction"""
        
        # Verify ownership
        model_info = self.model_manager.get_model_info(transaction.model_id)
        if not model_info:
            print(f"Model not found: {transaction.model_id}")
            return False
        
        if model_info.owner != transaction.sender:
            print(f"Not owner: {transaction.sender} cannot withdraw from {transaction.model_id}")
            return False
        
        # Withdraw from scheduler
        success, amount = self.training_scheduler.withdraw_balance(
            transaction.model_id,
            transaction.sender
        )
        
        if success and amount > 0:
            # Transfer funds back to owner
            withdrawal_address = transaction.withdrawal_address or transaction.sender
            
            if hasattr(self.economic_system, 'transfer'):
                self.economic_system.transfer(
                    from_address="model_escrow",
                    to_address=withdrawal_address,
                    amount=amount,
                    memo=f"Withdrawal from model {transaction.model_id[:16]}"
                )
            else:
                # Direct balance update
                self.economic_system.balances[withdrawal_address] = \
                    self.economic_system.balances.get(withdrawal_address, 0.0) + amount
            
            print(f"✓ Withdrew {amount:.2f} DESSIN from model {transaction.model_id[:20]}...")
            print(f"  To: {withdrawal_address[:20]}...")
        
        return success
    
    def auto_deprecate_old_models(
        self, 
        max_age_blocks: int = 100000,
        current_block: int = 0
    ) -> List[str]:
        """Automatically deprecate very old models"""
        
        deprecated_models = []
        
        for model_id, model_info in self.model_manager.models.items():
            model_age = current_block - model_info.upload_block
            
            # Check if model is too old
            if model_age > max_age_blocks:
                # Check if already deprecated
                stats = self.training_scheduler.get_model_stats(model_id)
                if stats and not stats['is_deprecated']:
                    self.training_scheduler.deprecate_model(
                        model_id,
                        reason=f"Auto-deprecated after {max_age_blocks} blocks"
                    )
                    deprecated_models.append(model_id)
        
        if deprecated_models:
            print(f"🗑️  Auto-deprecated {len(deprecated_models)} old models")
        
        return deprecated_models
    
    def auto_pause_unfunded_models(self) -> List[str]:
        """Automatically pause models that have run out of funds"""
        
        paused_models = []
        
        all_stats = self.training_scheduler.get_all_model_stats()
        
        for stats in all_stats:
            if stats['is_active'] and not stats['is_paid_up']:
                model_id = stats['model_id']
                self.training_scheduler.pause_model(model_id)
                paused_models.append(model_id)
        
        if paused_models:
            print(f"⏸️  Auto-paused {len(paused_models)} unfunded models")
        
        return paused_models
