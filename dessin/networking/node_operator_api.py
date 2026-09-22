"""
Node Operator API for DeSSIN
============================

Allows node operators to configure their pricing preferences on-the-fly.
Each node can set its own desired prices within safe bounds.

Operators can set target prices that will be applied when they become block leader,
since they can only influence network-wide prices when selected as leader.
"""

import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum


class PricingParameter(Enum):
    """Types of pricing parameters node operators can configure"""
    STORAGE_PRICE = "storage_price"
    BLOCK_REWARD_TARGET = "block_reward_target"
    TRAINING_PRICE = "training_price"
    QUERY_PRICE = "query_price"


class PricingStrategy(Enum):
    """Types of pricing strategies"""
    ABSOLUTE_TARGET = "absolute_target"  # Set specific price target
    RELATIVE_INCREASE = "relative_increase"  # Increase by percentage
    RELATIVE_DECREASE = "relative_decrease"  # Decrease by percentage
    TEMPORARY_ADJUSTMENT = "temporary_adjustment"  # Adjust for N blocks
    NO_TARGET = "no_target"  # Let network determine prices


@dataclass
class PricingTarget:
    """Pricing target that will be applied when node becomes block leader"""
    parameter: PricingParameter
    strategy: PricingStrategy
    
    # For absolute targets
    target_value: Optional[float] = None
    
    # For relative adjustments
    percentage_change: Optional[float] = None  # e.g., 0.5 for 50% increase
    
    # For temporary adjustments
    duration_blocks: Optional[int] = None
    blocks_remaining: Optional[int] = None
    original_value: Optional[float] = None
    
    # Metadata
    created_at: float = 0.0
    applied_count: int = 0  # How many times applied as leader
    last_applied: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if temporary adjustment has expired"""
        if self.strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            return self.blocks_remaining is not None and self.blocks_remaining <= 0
        return False
    
    def apply_as_leader(self, current_value: float, increase: bool = True) -> Optional[float]:
        """
        Calculate what adjustment to make as block leader
        
        Args:
            current_value: Current network value
            increase: Direction to move (for relative strategies)
        
        Returns:
            Target value to move towards, or None if no action
        """
        if self.strategy == PricingStrategy.NO_TARGET:
            return None
        
        elif self.strategy == PricingStrategy.ABSOLUTE_TARGET:
            # Move towards target value
            if self.target_value is None:
                return None
            
            # Determine if we should increase or decrease
            if current_value < self.target_value:
                return current_value  # Signal to increase
            elif current_value > self.target_value:
                return current_value  # Signal to decrease
            else:
                return None  # Already at target
        
        elif self.strategy == PricingStrategy.RELATIVE_INCREASE:
            # Always increase
            return current_value  # Signal to increase
        
        elif self.strategy == PricingStrategy.RELATIVE_DECREASE:
            # Always decrease
            return current_value  # Signal to decrease
        
        elif self.strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            # Apply for limited blocks
            if self.blocks_remaining is not None and self.blocks_remaining > 0:
                if self.percentage_change and self.percentage_change > 0:
                    return current_value  # Signal to increase
                elif self.percentage_change and self.percentage_change < 0:
                    return current_value  # Signal to decrease
        
        return None
    
    def should_increase(self, current_value: float) -> Optional[bool]:
        """
        Determine if should increase (True), decrease (False), or no action (None)
        
        Returns:
            True to increase, False to decrease, None for no action
        """
        if self.strategy == PricingStrategy.NO_TARGET:
            return None
        
        elif self.strategy == PricingStrategy.ABSOLUTE_TARGET:
            if self.target_value is None:
                return None
            if current_value < self.target_value:
                return True  # Increase towards target
            elif current_value > self.target_value:
                return False  # Decrease towards target
            else:
                return None  # At target
        
        elif self.strategy == PricingStrategy.RELATIVE_INCREASE:
            return True
        
        elif self.strategy == PricingStrategy.RELATIVE_DECREASE:
            return False
        
        elif self.strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            if self.is_expired():
                return None
            if self.percentage_change:
                return self.percentage_change > 0
        
        return None
    
    def mark_applied(self):
        """Mark that this target was applied as block leader"""
        self.applied_count += 1
        self.last_applied = time.time()
        
        # Decrement blocks remaining for temporary adjustments
        if self.strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            if self.blocks_remaining is not None:
                self.blocks_remaining -= 1


@dataclass
class PricingConfig:
    """Node operator's pricing configuration"""
    
    # Storage pricing (per GB per block)
    storage_price: float = 0.001
    min_storage_price: float = 0.0001  # 0.1x minimum
    max_storage_price: float = 0.01    # 10x maximum
    
    # Block reward target (what node expects to earn)
    block_reward_target: float = 10.0
    min_block_reward_target: float = 1.0
    max_block_reward_target: float = 100.0
    
    # Training pricing (per iteration)
    training_price: float = 0.001
    min_training_price: float = 0.0001
    max_training_price: float = 0.01
    
    # Query pricing (per token)
    query_price: float = 0.001
    min_query_price: float = 0.0001
    max_query_price: float = 0.01
    
    # Last update tracking
    last_updated: float = 0.0
    update_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """
        Validate all pricing parameters are within bounds
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not (self.min_storage_price <= self.storage_price <= self.max_storage_price):
            errors.append(
                f"Storage price {self.storage_price} outside bounds "
                f"[{self.min_storage_price}, {self.max_storage_price}]"
            )
        
        if not (self.min_block_reward_target <= self.block_reward_target <= self.max_block_reward_target):
            errors.append(
                f"Block reward target {self.block_reward_target} outside bounds "
                f"[{self.min_block_reward_target}, {self.max_block_reward_target}]"
            )
        
        if not (self.min_training_price <= self.training_price <= self.max_training_price):
            errors.append(
                f"Training price {self.training_price} outside bounds "
                f"[{self.min_training_price}, {self.max_training_price}]"
            )
        
        if not (self.min_query_price <= self.query_price <= self.max_query_price):
            errors.append(
                f"Query price {self.query_price} outside bounds "
                f"[{self.min_query_price}, {self.max_query_price}]"
            )
        
        return errors


@dataclass
class PricingHistory:
    """Record of a pricing change"""
    timestamp: float
    parameter: PricingParameter
    old_value: float
    new_value: float
    change_percent: float
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['parameter'] = self.parameter.value
        return result


class NodeOperatorAPI:
    """
    API for node operators to configure their node's pricing preferences
    
    This allows operators to:
    - Set their desired storage prices
    - Set their target block rewards
    - Adjust training and query pricing
    - Set pricing targets that apply when they become block leader
    - All changes are immediate (on-the-fly)
    - All changes are bounded for safety
    
    Note: Network-wide price adjustments only take effect when the node
    is selected as block leader via VRF randomness.
    """
    
    def __init__(
        self,
        initial_config: Optional[PricingConfig] = None,
        enable_bounds: bool = True
    ):
        """
        Initialize node operator API
        
        Args:
            initial_config: Initial pricing configuration
            enable_bounds: Enable safety bounds (recommended)
        """
        self.config = initial_config or PricingConfig()
        self.enable_bounds = enable_bounds
        
        # Pricing history for transparency
        self.pricing_history: List[PricingHistory] = []
        
        # Pricing targets (applied when node becomes block leader)
        self.pricing_targets: Dict[PricingParameter, PricingTarget] = {}
        
        # Block leader statistics
        self.times_selected_as_leader = 0
        self.adjustments_made_as_leader = 0
        self.last_leader_selection = 0.0
        
        # Validate initial configuration
        if self.enable_bounds:
            errors = self.config.validate()
            if errors:
                print(f"⚠️  Initial configuration has errors:")
                for error in errors:
                    print(f"   - {error}")
                print(f"   Using safe defaults instead")
                self.config = PricingConfig()
        
        print("✓ Node operator API initialized")
        print(f"  Storage price: {self.config.storage_price:.6f} DESSIN/GB/block")
        print(f"  Block reward target: {self.config.block_reward_target:.4f} DESSIN")
        print(f"  Training price: {self.config.training_price:.6f} DESSIN/iteration")
        print(f"  Query price: {self.config.query_price:.6f} DESSIN/token")
        print(f"  Safety bounds: {'enabled' if enable_bounds else 'disabled'}")
        print(f"  Block leader targets: 0 configured")
    
    def set_storage_price(
        self,
        new_price: float,
        reason: str = ""
    ) -> bool:
        """
        Set storage price (per GB per block)
        
        Args:
            new_price: New storage price
            reason: Reason for change
        
        Returns:
            True if successful, False if invalid
        """
        # Validate bounds
        if self.enable_bounds:
            if not (self.config.min_storage_price <= new_price <= self.config.max_storage_price):
                print(f"⚠️  Storage price rejected: {new_price:.6f}")
                print(f"   Must be between {self.config.min_storage_price:.6f} and {self.config.max_storage_price:.6f}")
                print(f"   Current: {self.config.storage_price:.6f}")
                return False
        
        # Record history
        old_price = self.config.storage_price
        change_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
        
        history = PricingHistory(
            timestamp=time.time(),
            parameter=PricingParameter.STORAGE_PRICE,
            old_value=old_price,
            new_value=new_price,
            change_percent=change_percent,
            reason=reason
        )
        self.pricing_history.append(history)
        
        # Apply change
        self.config.storage_price = new_price
        self.config.last_updated = time.time()
        self.config.update_count += 1
        
        direction = "↑" if new_price > old_price else "↓" if new_price < old_price else "="
        print(f"📊 {direction} Storage price updated: {old_price:.6f} → {new_price:.6f} DESSIN/GB/block")
        print(f"   Change: {change_percent:+.2f}%")
        if reason:
            print(f"   Reason: {reason}")
        
        return True
    
    def set_block_reward_target(
        self,
        new_target: float,
        reason: str = ""
    ) -> bool:
        """
        Set target block reward (what node expects to earn per block)
        
        Args:
            new_target: New block reward target
            reason: Reason for change
        
        Returns:
            True if successful, False if invalid
        """
        # Validate bounds
        if self.enable_bounds:
            if not (self.config.min_block_reward_target <= new_target <= self.config.max_block_reward_target):
                print(f"⚠️  Block reward target rejected: {new_target:.4f}")
                print(f"   Must be between {self.config.min_block_reward_target:.4f} and {self.config.max_block_reward_target:.4f}")
                print(f"   Current: {self.config.block_reward_target:.4f}")
                return False
        
        # Record history
        old_target = self.config.block_reward_target
        change_percent = ((new_target - old_target) / old_target * 100) if old_target > 0 else 0
        
        history = PricingHistory(
            timestamp=time.time(),
            parameter=PricingParameter.BLOCK_REWARD_TARGET,
            old_value=old_target,
            new_value=new_target,
            change_percent=change_percent,
            reason=reason
        )
        self.pricing_history.append(history)
        
        # Apply change
        self.config.block_reward_target = new_target
        self.config.last_updated = time.time()
        self.config.update_count += 1
        
        direction = "↑" if new_target > old_target else "↓" if new_target < old_target else "="
        print(f"📊 {direction} Block reward target updated: {old_target:.4f} → {new_target:.4f} DESSIN")
        print(f"   Change: {change_percent:+.2f}%")
        if reason:
            print(f"   Reason: {reason}")
        
        return True
    
    def set_training_price(
        self,
        new_price: float,
        reason: str = ""
    ) -> bool:
        """
        Set training price (per iteration)
        
        Args:
            new_price: New training price
            reason: Reason for change
        
        Returns:
            True if successful, False if invalid
        """
        # Validate bounds
        if self.enable_bounds:
            if not (self.config.min_training_price <= new_price <= self.config.max_training_price):
                print(f"⚠️  Training price rejected: {new_price:.6f}")
                print(f"   Must be between {self.config.min_training_price:.6f} and {self.config.max_training_price:.6f}")
                print(f"   Current: {self.config.training_price:.6f}")
                return False
        
        # Record history
        old_price = self.config.training_price
        change_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
        
        history = PricingHistory(
            timestamp=time.time(),
            parameter=PricingParameter.TRAINING_PRICE,
            old_value=old_price,
            new_value=new_price,
            change_percent=change_percent,
            reason=reason
        )
        self.pricing_history.append(history)
        
        # Apply change
        self.config.training_price = new_price
        self.config.last_updated = time.time()
        self.config.update_count += 1
        
        direction = "↑" if new_price > old_price else "↓" if new_price < old_price else "="
        print(f"📊 {direction} Training price updated: {old_price:.6f} → {new_price:.6f} DESSIN/iteration")
        print(f"   Change: {change_percent:+.2f}%")
        if reason:
            print(f"   Reason: {reason}")
        
        return True
    
    def set_query_price(
        self,
        new_price: float,
        reason: str = ""
    ) -> bool:
        """
        Set query price (per token)
        
        Args:
            new_price: New query price
            reason: Reason for change
        
        Returns:
            True if successful, False if invalid
        """
        # Validate bounds
        if self.enable_bounds:
            if not (self.config.min_query_price <= new_price <= self.config.max_query_price):
                print(f"⚠️  Query price rejected: {new_price:.6f}")
                print(f"   Must be between {self.config.min_query_price:.6f} and {self.config.max_query_price:.6f}")
                print(f"   Current: {self.config.query_price:.6f}")
                return False
        
        # Record history
        old_price = self.config.query_price
        change_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
        
        history = PricingHistory(
            timestamp=time.time(),
            parameter=PricingParameter.QUERY_PRICE,
            old_value=old_price,
            new_value=new_price,
            change_percent=change_percent,
            reason=reason
        )
        self.pricing_history.append(history)
        
        # Apply change
        self.config.query_price = new_price
        self.config.last_updated = time.time()
        self.config.update_count += 1
        
        direction = "↑" if new_price > old_price else "↓" if new_price < old_price else "="
        print(f"📊 {direction} Query price updated: {old_price:.6f} → {new_price:.6f} DESSIN/token")
        print(f"   Change: {change_percent:+.2f}%")
        if reason:
            print(f"   Reason: {reason}")
        
        return True
    
    def update_bounds(
        self,
        parameter: PricingParameter,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """
        Update safety bounds for a parameter (advanced operation)
        
        Args:
            parameter: Parameter to update bounds for
            min_value: New minimum value
            max_value: New maximum value
        
        Returns:
            True if successful
        """
        if not self.enable_bounds:
            print("⚠️  Bounds are disabled for this node")
            return False
        
        if parameter == PricingParameter.STORAGE_PRICE:
            if min_value is not None:
                self.config.min_storage_price = min_value
            if max_value is not None:
                self.config.max_storage_price = max_value
            print(f"✓ Storage price bounds updated: [{self.config.min_storage_price:.6f}, {self.config.max_storage_price:.6f}]")
        
        elif parameter == PricingParameter.BLOCK_REWARD_TARGET:
            if min_value is not None:
                self.config.min_block_reward_target = min_value
            if max_value is not None:
                self.config.max_block_reward_target = max_value
            print(f"✓ Block reward bounds updated: [{self.config.min_block_reward_target:.4f}, {self.config.max_block_reward_target:.4f}]")
        
        elif parameter == PricingParameter.TRAINING_PRICE:
            if min_value is not None:
                self.config.min_training_price = min_value
            if max_value is not None:
                self.config.max_training_price = max_value
            print(f"✓ Training price bounds updated: [{self.config.min_training_price:.6f}, {self.config.max_training_price:.6f}]")
        
        elif parameter == PricingParameter.QUERY_PRICE:
            if min_value is not None:
                self.config.min_query_price = min_value
            if max_value is not None:
                self.config.max_query_price = max_value
            print(f"✓ Query price bounds updated: [{self.config.min_query_price:.6f}, {self.config.max_query_price:.6f}]")
        
        return True
    
    def get_pricing_config(self) -> PricingConfig:
        """Get current pricing configuration"""
        return self.config
    
    def get_pricing_history(
        self,
        parameter: Optional[PricingParameter] = None,
        limit: int = 100
    ) -> List[PricingHistory]:
        """
        Get pricing change history
        
        Args:
            parameter: Filter by parameter type (None for all)
            limit: Maximum number of records
        
        Returns:
            List of pricing changes, most recent first
        """
        history = self.pricing_history[::-1]  # Reverse for recent first
        
        if parameter:
            history = [h for h in history if h.parameter == parameter]
        
        return history[:limit]
    
    def get_pricing_summary(self) -> Dict[str, Any]:
        """Get comprehensive pricing summary"""
        return {
            "current_prices": {
                "storage": self.config.storage_price,
                "block_reward_target": self.config.block_reward_target,
                "training": self.config.training_price,
                "query": self.config.query_price
            },
            "bounds": {
                "storage": [self.config.min_storage_price, self.config.max_storage_price],
                "block_reward_target": [self.config.min_block_reward_target, self.config.max_block_reward_target],
                "training": [self.config.min_training_price, self.config.max_training_price],
                "query": [self.config.min_query_price, self.config.max_query_price]
            },
            "metadata": {
                "last_updated": self.config.last_updated,
                "update_count": self.config.update_count,
                "history_records": len(self.pricing_history),
                "bounds_enabled": self.enable_bounds
            }
        }
    
    def bulk_update_prices(
        self,
        updates: Dict[str, float],
        reason: str = ""
    ) -> Dict[str, bool]:
        """
        Update multiple prices at once
        
        Args:
            updates: Dict mapping parameter names to new values
            reason: Reason for changes
        
        Returns:
            Dict mapping parameter names to success status
        """
        results = {}
        
        for param_name, new_value in updates.items():
            if param_name == "storage_price" or param_name == "storage":
                results["storage_price"] = self.set_storage_price(new_value, reason)
            
            elif param_name == "block_reward_target" or param_name == "block_reward":
                results["block_reward_target"] = self.set_block_reward_target(new_value, reason)
            
            elif param_name == "training_price" or param_name == "training":
                results["training_price"] = self.set_training_price(new_value, reason)
            
            elif param_name == "query_price" or param_name == "query":
                results["query_price"] = self.set_query_price(new_value, reason)
            
            else:
                print(f"⚠️  Unknown parameter: {param_name}")
                results[param_name] = False
        
        return results
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all prices to default values
        
        Returns:
            True if successful
        """
        old_config = self.config
        self.config = PricingConfig()
        
        print("🔄 Pricing reset to defaults")
        print(f"  Storage: {old_config.storage_price:.6f} → {self.config.storage_price:.6f}")
        print(f"  Block reward: {old_config.block_reward_target:.4f} → {self.config.block_reward_target:.4f}")
        print(f"  Training: {old_config.training_price:.6f} → {self.config.training_price:.6f}")
        print(f"  Query: {old_config.query_price:.6f} → {self.config.query_price:.6f}")
        
        return True
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration for backup/sharing"""
        return {
            "config": self.config.to_dict(),
            "history": [h.to_dict() for h in self.pricing_history],
            "summary": self.get_pricing_summary()
        }
    
    def import_config(self, data: Dict[str, Any]) -> bool:
        """
        Import configuration from backup
        
        Args:
            data: Configuration data (from export_config)
        
        Returns:
            True if successful
        """
        try:
            if "config" in data:
                new_config = PricingConfig(**data["config"])
                
                # Validate
                if self.enable_bounds:
                    errors = new_config.validate()
                    if errors:
                        print(f"⚠️  Import failed: invalid configuration")
                        for error in errors:
                            print(f"   - {error}")
                        return False
                
                self.config = new_config
                print("✓ Configuration imported successfully")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Import failed: {e}")
            return False
    
    # ============================================================================
    # Block Leader Pricing Targets
    # ============================================================================
    
    def set_pricing_target(
        self,
        parameter: PricingParameter,
        strategy: PricingStrategy,
        target_value: Optional[float] = None,
        percentage_change: Optional[float] = None,
        duration_blocks: Optional[int] = None
    ) -> bool:
        """
        Set a pricing target that will be applied when node becomes block leader
        
        Args:
            parameter: Which parameter to target
            strategy: Strategy type (absolute, relative, etc.)
            target_value: Target value (for ABSOLUTE_TARGET)
            percentage_change: Percentage change (for RELATIVE_* strategies)
            duration_blocks: Duration in blocks (for TEMPORARY_ADJUSTMENT)
        
        Returns:
            True if successful
        
        Examples:
            # Set absolute target
            api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                PricingStrategy.ABSOLUTE_TARGET,
                target_value=0.002
            )
            
            # Increase by 50%
            api.set_pricing_target(
                PricingParameter.BLOCK_REWARD_TARGET,
                PricingStrategy.RELATIVE_INCREASE,
                percentage_change=0.5
            )
            
            # Decrease for 100 blocks
            api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                PricingStrategy.TEMPORARY_ADJUSTMENT,
                percentage_change=-0.2,  # 20% decrease
                duration_blocks=100
            )
            
            # Remove target
            api.set_pricing_target(
                PricingParameter.STORAGE_PRICE,
                PricingStrategy.NO_TARGET
            )
        """
        # Create target
        target = PricingTarget(
            parameter=parameter,
            strategy=strategy,
            target_value=target_value,
            percentage_change=percentage_change,
            duration_blocks=duration_blocks,
            blocks_remaining=duration_blocks,
            created_at=time.time()
        )
        
        # Store target
        self.pricing_targets[parameter] = target
        
        # Print confirmation
        print(f"✓ Pricing target set: {parameter.value}")
        print(f"  Strategy: {strategy.value}")
        
        if strategy == PricingStrategy.ABSOLUTE_TARGET and target_value:
            print(f"  Target: {target_value:.6f}")
        
        elif strategy == PricingStrategy.RELATIVE_INCREASE and percentage_change:
            print(f"  Increase: {percentage_change*100:.1f}%")
        
        elif strategy == PricingStrategy.RELATIVE_DECREASE and percentage_change:
            print(f"  Decrease: {abs(percentage_change)*100:.1f}%")
        
        elif strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            if percentage_change:
                direction = "increase" if percentage_change > 0 else "decrease"
                print(f"  {direction.capitalize()}: {abs(percentage_change)*100:.1f}%")
            if duration_blocks:
                print(f"  Duration: {duration_blocks} blocks")
        
        elif strategy == PricingStrategy.NO_TARGET:
            print(f"  Target removed")
        
        print(f"  Note: Will apply when selected as block leader")
        
        return True
    
    def get_pricing_targets(self) -> Dict[PricingParameter, PricingTarget]:
        """Get all configured pricing targets"""
        return self.pricing_targets.copy()
    
    def remove_pricing_target(self, parameter: PricingParameter) -> bool:
        """Remove a pricing target"""
        if parameter in self.pricing_targets:
            del self.pricing_targets[parameter]
            print(f"✓ Pricing target removed: {parameter.value}")
            return True
        return False
    
    def apply_as_block_leader(
        self,
        current_network_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Called when this node is selected as block leader
        
        Applies configured pricing targets to adjust network prices
        
        Args:
            current_network_prices: Current network-wide prices
                {
                    'storage_price': 0.001,
                    'block_reward': 10.0,
                    'training_price': 0.001
                }
        
        Returns:
            List of adjustments to make:
                [
                    {
                        'parameter': 'storage_price',
                        'increase': True,  # or False
                        'reason': 'Moving towards target 0.002'
                    }
                ]
        """
        self.times_selected_as_leader += 1
        self.last_leader_selection = time.time()
        
        adjustments = []
        
        # Check each pricing target
        for param, target in self.pricing_targets.items():
            # Skip expired targets
            if target.is_expired():
                continue
            
            # Get current network value
            param_name = param.value
            if param_name == 'block_reward_target':
                param_name = 'block_reward'
            
            current_value = current_network_prices.get(param_name)
            if current_value is None:
                continue
            
            # Determine adjustment
            should_increase = target.should_increase(current_value)
            
            if should_increase is not None:
                # Prepare adjustment
                reason = self._get_adjustment_reason(target, current_value)
                
                adjustment = {
                    'parameter': param,
                    'increase': should_increase,
                    'reason': reason
                }
                
                adjustments.append(adjustment)
                
                # Mark as applied
                target.mark_applied()
                self.adjustments_made_as_leader += 1
                
                print(f"📊 Block leader adjustment: {param.value}")
                print(f"   Direction: {'increase' if should_increase else 'decrease'}")
                print(f"   Reason: {reason}")
        
        return adjustments
    
    def _get_adjustment_reason(self, target: PricingTarget, current_value: float) -> str:
        """Generate reason for adjustment"""
        if target.strategy == PricingStrategy.ABSOLUTE_TARGET:
            return f"Moving towards target {target.target_value:.6f} (current: {current_value:.6f})"
        
        elif target.strategy == PricingStrategy.RELATIVE_INCREASE:
            if target.percentage_change:
                return f"Increasing by {target.percentage_change*100:.1f}% (strategy)"
            return "Increasing (strategy)"
        
        elif target.strategy == PricingStrategy.RELATIVE_DECREASE:
            if target.percentage_change:
                return f"Decreasing by {abs(target.percentage_change)*100:.1f}% (strategy)"
            return "Decreasing (strategy)"
        
        elif target.strategy == PricingStrategy.TEMPORARY_ADJUSTMENT:
            direction = "Increasing" if target.percentage_change and target.percentage_change > 0 else "Decreasing"
            blocks_left = target.blocks_remaining or 0
            return f"{direction} (temporary, {blocks_left} blocks remaining)"
        
        return "Adjustment per operator strategy"
    
    def get_leader_statistics(self) -> Dict[str, Any]:
        """Get statistics about block leader selections"""
        return {
            "times_selected": self.times_selected_as_leader,
            "adjustments_made": self.adjustments_made_as_leader,
            "last_selection": self.last_leader_selection,
            "active_targets": len([t for t in self.pricing_targets.values() if not t.is_expired()]),
            "expired_targets": len([t for t in self.pricing_targets.values() if t.is_expired()])
        }
