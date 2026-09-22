"""
Model Training Scheduler for DeSSIN
====================================

Intelligently schedules training iterations across multiple models
based on payment status, priority, and fair rotation principles.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ModelPriority(Enum):
    """Priority levels for model training"""
    HIGH = "high"      # Paid premium
    NORMAL = "normal"  # Standard payment
    LOW = "low"        # Minimal payment
    PAUSED = "paused"  # Payment expired


@dataclass
class ModelTrainingSlot:
    """Represents a model's training slot in the schedule"""
    model_id: str
    owner: str
    payment_per_iteration: float
    priority: ModelPriority
    
    # Statistics
    total_iterations: int = 0
    iterations_this_round: int = 0
    last_trained_time: float = 0.0
    total_payments: float = 0.0
    
    # Payment tracking
    payment_balance: float = 0.0
    min_balance_required: float = 1.0
    
    # Status
    is_active: bool = True
    is_paid_up: bool = True
    is_deprecated: bool = False
    deprecation_reason: str = ""
    deprecation_time: float = 0.0
    
    def has_sufficient_payment(self) -> bool:
        """Check if model has sufficient payment balance"""
        return self.payment_balance >= self.min_balance_required
    
    def deduct_payment(self, amount: float) -> bool:
        """Deduct payment for an iteration"""
        if self.payment_balance >= amount:
            self.payment_balance -= amount
            self.total_payments += amount
            return True
        return False
    
    def add_payment(self, amount: float):
        """Add payment to balance"""
        self.payment_balance += amount


class ModelTrainingScheduler:
    """Schedules training iterations across multiple models"""
    
    def __init__(self):
        self.training_slots: Dict[str, ModelTrainingSlot] = {}
        self.rotation_queue: List[str] = []
        self.current_round = 0
        
        # Scheduling parameters
        self.round_iterations = 10  # Iterations per round before rebalancing
        self.fair_rotation = True   # Enable fair rotation
        
        print("✓ Model training scheduler initialized")
        print(f"  Fair rotation: {self.fair_rotation}")
        print(f"  Round size: {self.round_iterations} iterations")
    
    def register_model(
        self,
        model_id: str,
        owner: str,
        payment_per_iteration: float,
        initial_payment: float = 10.0,
        priority: ModelPriority = ModelPriority.NORMAL
    ) -> bool:
        """Register a model for training"""
        
        if model_id in self.training_slots:
            print(f"Model {model_id} already registered")
            return False
        
        slot = ModelTrainingSlot(
            model_id=model_id,
            owner=owner,
            payment_per_iteration=payment_per_iteration,
            priority=priority,
            payment_balance=initial_payment
        )
        
        self.training_slots[model_id] = slot
        self._update_rotation_queue()
        
        print(f"✓ Registered model {model_id[:20]}...")
        print(f"  Owner: {owner[:20]}...")
        print(f"  Payment/iter: {payment_per_iteration:.4f} DESSIN")
        print(f"  Initial balance: {initial_payment:.2f} DESSIN")
        print(f"  Priority: {priority.value}")
        
        return True
    
    def _update_rotation_queue(self):
        """Update the rotation queue based on active, paid models"""
        self.rotation_queue = []
        
        # Group by priority
        high_priority = []
        normal_priority = []
        low_priority = []
        
        for model_id, slot in self.training_slots.items():
            if not slot.is_active:
                continue
            
            if not slot.has_sufficient_payment():
                slot.is_paid_up = False
                continue
            
            slot.is_paid_up = True
            
            if slot.priority == ModelPriority.HIGH:
                high_priority.append(model_id)
            elif slot.priority == ModelPriority.NORMAL:
                normal_priority.append(model_id)
            elif slot.priority == ModelPriority.LOW:
                low_priority.append(model_id)
        
        # Build queue with fair rotation
        if self.fair_rotation:
            # Round-robin within each priority tier
            # High priority models get more slots
            max_len = max(
                len(high_priority) * 3,  # 3x weight
                len(normal_priority) * 2,  # 2x weight
                len(low_priority) * 1,     # 1x weight
                1
            )
            
            for i in range(max_len):
                if high_priority:
                    self.rotation_queue.append(high_priority[i % len(high_priority)])
                if i % 2 == 0 and normal_priority:
                    self.rotation_queue.append(normal_priority[i % len(normal_priority)])
                if i % 3 == 0 and low_priority:
                    self.rotation_queue.append(low_priority[i % len(low_priority)])
        else:
            # Simple concatenation
            self.rotation_queue = high_priority + normal_priority + low_priority
    
    def get_next_model_for_training(self) -> Optional[str]:
        """Get the next model that should be trained"""
        
        if not self.rotation_queue:
            self._update_rotation_queue()
        
        if not self.rotation_queue:
            return None
        
        # Round-robin through queue
        if self.fair_rotation:
            # Rotate through all models in queue
            for _ in range(len(self.rotation_queue)):
                model_id = self.rotation_queue[0]
                self.rotation_queue = self.rotation_queue[1:] + [model_id]
                
                slot = self.training_slots.get(model_id)
                if slot and slot.is_active and slot.has_sufficient_payment():
                    return model_id
            
            # No valid models found
            return None
        else:
            # Take first available
            model_id = self.rotation_queue[0]
            return model_id if self.training_slots[model_id].has_sufficient_payment() else None
    
    def record_training_iteration(
        self,
        model_id: str,
        iteration_cost: Optional[float] = None
    ) -> bool:
        """Record that a model was trained for one iteration"""
        
        if model_id not in self.training_slots:
            return False
        
        slot = self.training_slots[model_id]
        
        # Deduct payment
        cost = iteration_cost if iteration_cost is not None else slot.payment_per_iteration
        if not slot.deduct_payment(cost):
            print(f"⚠️  Model {model_id[:20]}... insufficient funds")
            slot.is_paid_up = False
            self._update_rotation_queue()
            return False
        
        # Update stats
        slot.total_iterations += 1
        slot.iterations_this_round += 1
        slot.last_trained_time = time.time()
        
        # Check if round completed
        if slot.iterations_this_round >= self.round_iterations:
            slot.iterations_this_round = 0
            self.current_round += 1
            self._update_rotation_queue()
        
        return True
    
    def add_model_payment(self, model_id: str, amount: float) -> bool:
        """Add payment for a model"""
        if model_id not in self.training_slots:
            return False
        
        slot = self.training_slots[model_id]
        slot.add_payment(amount)
        
        print(f"✓ Payment added to {model_id[:20]}...")
        print(f"  Amount: {amount:.2f} DESSIN")
        print(f"  New balance: {slot.payment_balance:.2f} DESSIN")
        
        # Reactivate if was paused
        if not slot.is_paid_up and slot.has_sufficient_payment():
            slot.is_paid_up = True
            self._update_rotation_queue()
            print(f"  Model reactivated for training")
        
        return True
    
    def pause_model(self, model_id: str) -> bool:
        """Pause training for a model"""
        if model_id not in self.training_slots:
            return False
        
        slot = self.training_slots[model_id]
        slot.is_active = False
        self._update_rotation_queue()
        
        print(f"⏸️  Paused model {model_id[:20]}...")
        return True
    
    def resume_model(self, model_id: str) -> bool:
        """Resume training for a model"""
        if model_id not in self.training_slots:
            return False
        
        slot = self.training_slots[model_id]
        slot.is_active = True
        self._update_rotation_queue()
        
        print(f"▶️  Resumed model {model_id[:20]}...")
        return True
    
    def get_schedule_status(self) -> Dict:
        """Get current scheduling status"""
        active_models = sum(1 for s in self.training_slots.values() if s.is_active)
        paid_models = sum(1 for s in self.training_slots.values() 
                         if s.is_active and s.is_paid_up)
        
        return {
            "total_models": len(self.training_slots),
            "active_models": active_models,
            "paid_models": paid_models,
            "rotation_queue_size": len(self.rotation_queue),
            "current_round": self.current_round,
            "fair_rotation": self.fair_rotation
        }
    
    def get_model_stats(self, model_id: str) -> Optional[Dict]:
        """Get statistics for a specific model"""
        if model_id not in self.training_slots:
            return None
        
        slot = self.training_slots[model_id]
        
        return {
            "model_id": model_id,
            "owner": slot.owner,
            "total_iterations": slot.total_iterations,
            "iterations_this_round": slot.iterations_this_round,
            "payment_balance": slot.payment_balance,
            "total_payments": slot.total_payments,
            "payment_per_iteration": slot.payment_per_iteration,
            "priority": slot.priority.value,
            "is_active": slot.is_active,
            "is_paid_up": slot.is_paid_up,
            "is_deprecated": slot.is_deprecated,
            "deprecation_reason": slot.deprecation_reason
        }
    
    def get_all_model_stats(self) -> List[Dict]:
        """Get statistics for all models"""
        return [self.get_model_stats(mid) for mid in self.training_slots.keys()]
    
    def deprecate_model(
        self, 
        model_id: str, 
        reason: str = "Model deprecated by owner"
    ) -> bool:
        """Deprecate a model (owner marks it as obsolete)"""
        if model_id not in self.training_slots:
            print(f"Model not found: {model_id}")
            return False
        
        slot = self.training_slots[model_id]
        slot.is_deprecated = True
        slot.is_active = False
        slot.deprecation_reason = reason
        slot.deprecation_time = time.time()
        
        self._update_rotation_queue()
        
        print(f"🗑️  Deprecated model {model_id[:20]}...")
        print(f"  Reason: {reason}")
        print(f"  Remaining balance: {slot.payment_balance:.2f} DESSIN (available for withdrawal)")
        
        return True
    
    def withdraw_balance(self, model_id: str, owner: str) -> Tuple[bool, float]:
        """Withdraw remaining payment balance from a model"""
        if model_id not in self.training_slots:
            print(f"Model not found: {model_id}")
            return False, 0.0
        
        slot = self.training_slots[model_id]
        
        # Verify ownership
        if slot.owner != owner:
            print(f"❌ Not owner: {owner[:20]}... cannot withdraw")
            return False, 0.0
        
        # Can only withdraw from paused or deprecated models
        if slot.is_active and not slot.is_deprecated:
            print(f"⚠️  Model still active. Pause or deprecate before withdrawing.")
            return False, 0.0
        
        # Withdraw all remaining balance
        withdrawal_amount = slot.payment_balance
        slot.payment_balance = 0.0
        
        print(f"💰 Withdrawal successful")
        print(f"  Model: {model_id[:20]}...")
        print(f"  Owner: {owner[:20]}...")
        print(f"  Amount: {withdrawal_amount:.2f} DESSIN")
        
        return True, withdrawal_amount
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model completely from the scheduler"""
        if model_id not in self.training_slots:
            return False
        
        slot = self.training_slots[model_id]
        
        # Warn if balance remaining
        if slot.payment_balance > 0:
            print(f"⚠️  Model has {slot.payment_balance:.2f} DESSIN remaining")
            print(f"   Consider withdrawing balance first")
        
        del self.training_slots[model_id]
        self._update_rotation_queue()
        
        print(f"🗑️  Removed model {model_id[:20]}... from scheduler")
        
        return True
    
    def print_rotation_summary(self):
        """Print a summary of the current rotation state"""
        print()
        print("=" * 80)
        print("Training Rotation Summary")
        print("=" * 80)
        
        status = self.get_schedule_status()
        print(f"Total models: {status['total_models']}")
        print(f"Active models: {status['active_models']}")
        print(f"Paid models: {status['paid_models']}")
        print(f"Current round: {status['current_round']}")
        print()
        
        print("Model Status:")
        print("-" * 80)
        
        for slot in sorted(self.training_slots.values(), 
                          key=lambda s: s.total_iterations, reverse=True):
            if slot.is_deprecated:
                status_icon = "🗑"
            elif slot.is_active and slot.is_paid_up:
                status_icon = "✓"
            else:
                status_icon = "⏸"
            
            status_suffix = " [DEPRECATED]" if slot.is_deprecated else ""
            
            print(f"{status_icon} {slot.model_id[:30]:32s} | "
                  f"Iters: {slot.total_iterations:4d} | "
                  f"Balance: {slot.payment_balance:6.2f} | "
                  f"Paid: {slot.total_payments:6.2f}{status_suffix}")
        
        print("=" * 80)
        print()
