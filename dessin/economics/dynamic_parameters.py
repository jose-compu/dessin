"""
Dynamic Network Parameters
===========================

Allows block leaders to adjust network parameters like block rewards and storage pricing
by small increments (±0.1%) per block to respond to market conditions.

Inspired by blockchain designs where validators can propose parameter adjustments.
"""

import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum


class ParameterType(Enum):
    """Types of adjustable parameters"""
    BLOCK_REWARD = "block_reward"
    STORAGE_PRICE = "storage_price"
    TRAINING_PAYMENT = "training_payment"


@dataclass
class ParameterAdjustment:
    """Represents a parameter adjustment by a block leader"""
    parameter_type: ParameterType
    block_index: int
    block_leader: str
    old_value: float
    new_value: float
    adjustment_percent: float
    timestamp: float
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_type": self.parameter_type.value,
            "block_index": self.block_index,
            "block_leader": self.block_leader,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "adjustment_percent": self.adjustment_percent,
            "timestamp": self.timestamp,
            "reason": self.reason
        }


class DynamicParameterManager:
    """Manages dynamic parameter adjustments by block leaders"""
    
    def __init__(
        self,
        initial_block_reward: float = 10.0,
        initial_storage_price: float = 0.001,
        initial_training_payment: float = 0.001,
        max_adjustment_percent: float = 0.1,  # 0.1% max adjustment per block
        min_block_reward: float = 0.1,
        max_block_reward: float = 100.0,
        min_storage_price: float = 0.0001,
        max_storage_price: float = 0.1,
        min_training_payment: float = 0.0001,
        max_training_payment: float = 0.1
    ):
        """
        Initialize dynamic parameter manager
        
        Args:
            initial_block_reward: Starting block reward
            initial_storage_price: Starting storage price per GB per block
            initial_training_payment: Starting training payment per iteration
            max_adjustment_percent: Maximum adjustment per block (0.1% = 0.001)
            min/max values: Bounds for each parameter
        """
        # Current parameter values
        self.block_reward = initial_block_reward
        self.storage_price = initial_storage_price
        self.training_payment = initial_training_payment
        
        # Adjustment limits
        self.max_adjustment_percent = max_adjustment_percent / 100  # Convert to decimal
        
        # Parameter bounds
        self.min_block_reward = min_block_reward
        self.max_block_reward = max_block_reward
        self.min_storage_price = min_storage_price
        self.max_storage_price = max_storage_price
        self.min_training_payment = min_training_payment
        self.max_training_payment = max_training_payment
        
        # History of adjustments
        self.adjustment_history: List[ParameterAdjustment] = []
        
        print("✓ Dynamic parameter manager initialized")
        print(f"  Block reward: {self.block_reward} DESSIN")
        print(f"  Storage price: {self.storage_price} DESSIN/GB/block")
        print(f"  Training payment: {self.training_payment} DESSIN/iteration")
        print(f"  Max adjustment per block: ±{max_adjustment_percent}%")
    
    def adjust_block_reward(
        self,
        block_index: int,
        block_leader: str,
        increase: bool,
        reason: str = ""
    ) -> Optional[ParameterAdjustment]:
        """
        Adjust block reward by ±0.1% (or configured max)
        
        Args:
            block_index: Current block index
            block_leader: Address of block leader making adjustment
            increase: True to increase, False to decrease
            reason: Optional reason for adjustment
        
        Returns:
            ParameterAdjustment if successful, None if out of bounds
        """
        old_value = self.block_reward
        
        # Calculate adjustment
        adjustment = old_value * self.max_adjustment_percent
        new_value = old_value + adjustment if increase else old_value - adjustment
        
        # Check bounds
        if new_value < self.min_block_reward or new_value > self.max_block_reward:
            print(f"⚠️  Block reward adjustment rejected: {new_value:.6f} out of bounds")
            return None
        
        # Apply adjustment
        self.block_reward = new_value
        
        adjustment_percent = ((new_value - old_value) / old_value) * 100
        
        adj = ParameterAdjustment(
            parameter_type=ParameterType.BLOCK_REWARD,
            block_index=block_index,
            block_leader=block_leader,
            old_value=old_value,
            new_value=new_value,
            adjustment_percent=adjustment_percent,
            timestamp=time.time(),
            reason=reason
        )
        
        self.adjustment_history.append(adj)
        
        direction = "↑" if increase else "↓"
        print(f"📊 {direction} Block reward adjusted: {old_value:.4f} → {new_value:.4f} DESSIN ({adjustment_percent:+.2f}%)")
        if reason:
            print(f"   Reason: {reason}")
        
        return adj
    
    def adjust_storage_price(
        self,
        block_index: int,
        block_leader: str,
        increase: bool,
        reason: str = ""
    ) -> Optional[ParameterAdjustment]:
        """
        Adjust storage price by ±0.1% (or configured max)
        
        Args:
            block_index: Current block index
            block_leader: Address of block leader making adjustment
            increase: True to increase, False to decrease
            reason: Optional reason for adjustment
        
        Returns:
            ParameterAdjustment if successful, None if out of bounds
        """
        old_value = self.storage_price
        
        # Calculate adjustment
        adjustment = old_value * self.max_adjustment_percent
        new_value = old_value + adjustment if increase else old_value - adjustment
        
        # Check bounds
        if new_value < self.min_storage_price or new_value > self.max_storage_price:
            print(f"⚠️  Storage price adjustment rejected: {new_value:.6f} out of bounds")
            return None
        
        # Apply adjustment
        self.storage_price = new_value
        
        adjustment_percent = ((new_value - old_value) / old_value) * 100
        
        adj = ParameterAdjustment(
            parameter_type=ParameterType.STORAGE_PRICE,
            block_index=block_index,
            block_leader=block_leader,
            old_value=old_value,
            new_value=new_value,
            adjustment_percent=adjustment_percent,
            timestamp=time.time(),
            reason=reason
        )
        
        self.adjustment_history.append(adj)
        
        direction = "↑" if increase else "↓"
        print(f"📊 {direction} Storage price adjusted: {old_value:.6f} → {new_value:.6f} DESSIN/GB/block ({adjustment_percent:+.2f}%)")
        if reason:
            print(f"   Reason: {reason}")
        
        return adj
    
    def adjust_training_payment(
        self,
        block_index: int,
        block_leader: str,
        increase: bool,
        reason: str = ""
    ) -> Optional[ParameterAdjustment]:
        """
        Adjust training payment by ±0.1% (or configured max)
        
        Args:
            block_index: Current block index
            block_leader: Address of block leader making adjustment
            increase: True to increase, False to decrease
            reason: Optional reason for adjustment
        
        Returns:
            ParameterAdjustment if successful, None if out of bounds
        """
        old_value = self.training_payment
        
        # Calculate adjustment
        adjustment = old_value * self.max_adjustment_percent
        new_value = old_value + adjustment if increase else old_value - adjustment
        
        # Check bounds
        if new_value < self.min_training_payment or new_value > self.max_training_payment:
            print(f"⚠️  Training payment adjustment rejected: {new_value:.6f} out of bounds")
            return None
        
        # Apply adjustment
        self.training_payment = new_value
        
        adjustment_percent = ((new_value - old_value) / old_value) * 100
        
        adj = ParameterAdjustment(
            parameter_type=ParameterType.TRAINING_PAYMENT,
            block_index=block_index,
            block_leader=block_leader,
            old_value=old_value,
            new_value=new_value,
            adjustment_percent=adjustment_percent,
            timestamp=time.time(),
            reason=reason
        )
        
        self.adjustment_history.append(adj)
        
        direction = "↑" if increase else "↓"
        print(f"📊 {direction} Training payment adjusted: {old_value:.6f} → {new_value:.6f} DESSIN/iter ({adjustment_percent:+.2f}%)")
        if reason:
            print(f"   Reason: {reason}")
        
        return adj
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values"""
        return {
            "block_reward": self.block_reward,
            "storage_price": self.storage_price,
            "training_payment": self.training_payment
        }
    
    def get_adjustment_history(
        self,
        parameter_type: Optional[ParameterType] = None,
        limit: int = 100
    ) -> List[ParameterAdjustment]:
        """
        Get adjustment history
        
        Args:
            parameter_type: Filter by parameter type (None for all)
            limit: Maximum number of records to return
        
        Returns:
            List of adjustments, most recent first
        """
        history = self.adjustment_history[::-1]  # Reverse for most recent first
        
        if parameter_type:
            history = [adj for adj in history if adj.parameter_type == parameter_type]
        
        return history[:limit]
    
    def get_parameter_trend(
        self,
        parameter_type: ParameterType,
        blocks: int = 100
    ) -> Dict[str, Any]:
        """
        Get trend analysis for a parameter over recent blocks
        
        Args:
            parameter_type: Parameter to analyze
            blocks: Number of recent blocks to analyze
        
        Returns:
            Dict with trend statistics
        """
        history = [
            adj for adj in self.adjustment_history
            if adj.parameter_type == parameter_type
        ]
        
        if not history:
            return {"error": "No history available"}
        
        # Get recent adjustments
        recent = history[-blocks:] if len(history) > blocks else history
        
        if not recent:
            return {"error": "No recent adjustments"}
        
        # Calculate statistics
        increases = sum(1 for adj in recent if adj.adjustment_percent > 0)
        decreases = sum(1 for adj in recent if adj.adjustment_percent < 0)
        total_change = sum(adj.adjustment_percent for adj in recent)
        
        start_value = recent[0].old_value
        end_value = recent[-1].new_value
        net_change_percent = ((end_value - start_value) / start_value) * 100
        
        return {
            "parameter_type": parameter_type.value,
            "blocks_analyzed": len(recent),
            "increases": increases,
            "decreases": decreases,
            "start_value": start_value,
            "end_value": end_value,
            "net_change_percent": net_change_percent,
            "total_adjustments": len(recent),
            "average_adjustment_percent": total_change / len(recent) if recent else 0
        }
    
    def suggest_adjustment(
        self,
        parameter_type: ParameterType,
        network_metrics: Dict[str, Any]
    ) -> Optional[bool]:
        """
        Suggest whether to increase or decrease a parameter based on network metrics
        
        Args:
            parameter_type: Parameter to adjust
            network_metrics: Dict with metrics like storage_utilization, mining_rate, etc.
        
        Returns:
            True to increase, False to decrease, None for no change
        """
        if parameter_type == ParameterType.STORAGE_PRICE:
            # Increase price if storage utilization is high
            utilization = network_metrics.get("storage_utilization", 0.5)
            if utilization > 0.8:
                return True  # Increase to reduce demand
            elif utilization < 0.3:
                return False  # Decrease to increase demand
        
        elif parameter_type == ParameterType.BLOCK_REWARD:
            # Adjust based on mining rate
            mining_rate = network_metrics.get("blocks_per_hour", 60)
            target_rate = network_metrics.get("target_blocks_per_hour", 60)
            
            if mining_rate < target_rate * 0.9:
                return True  # Increase reward to attract miners
            elif mining_rate > target_rate * 1.1:
                return False  # Decrease reward to reduce mining pressure
        
        elif parameter_type == ParameterType.TRAINING_PAYMENT:
            # Adjust based on training demand
            training_utilization = network_metrics.get("training_utilization", 0.5)
            if training_utilization > 0.8:
                return True  # Increase to attract more trainers
            elif training_utilization < 0.3:
                return False  # Decrease to reduce costs
        
        return None  # No change suggested
