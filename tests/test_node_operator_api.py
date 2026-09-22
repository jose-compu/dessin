"""
Tests for Node Operator API
"""

import pytest
from dessin.networking.node_operator_api import (
    NodeOperatorAPI,
    PricingConfig,
    PricingParameter,
    PricingHistory
)


@pytest.fixture
def operator_api():
    """Create a node operator API"""
    return NodeOperatorAPI()


def test_operator_api_initialization(operator_api):
    """Test operator API initializes with defaults"""
    config = operator_api.config
    
    assert config.storage_price == 0.001
    assert config.block_reward_target == 10.0
    assert config.training_price == 0.001
    assert config.query_price == 0.001
    assert operator_api.enable_bounds is True


def test_set_storage_price(operator_api):
    """Test setting storage price"""
    old_price = operator_api.config.storage_price
    
    success = operator_api.set_storage_price(
        0.002,
        reason="Increased demand for storage"
    )
    
    assert success is True
    assert operator_api.config.storage_price == 0.002
    assert len(operator_api.pricing_history) == 1


def test_set_storage_price_out_of_bounds(operator_api):
    """Test that out-of-bounds prices are rejected"""
    # Try to set too high
    success = operator_api.set_storage_price(0.1)  # Way above max
    assert success is False
    assert operator_api.config.storage_price == 0.001  # Unchanged
    
    # Try to set too low
    success = operator_api.set_storage_price(0.00001)  # Way below min
    assert success is False
    assert operator_api.config.storage_price == 0.001  # Unchanged


def test_set_block_reward_target(operator_api):
    """Test setting block reward target"""
    success = operator_api.set_block_reward_target(
        15.0,
        reason="Need higher rewards to justify mining costs"
    )
    
    assert success is True
    assert operator_api.config.block_reward_target == 15.0


def test_set_training_price(operator_api):
    """Test setting training price"""
    success = operator_api.set_training_price(
        0.002,
        reason="Higher GPU costs"
    )
    
    assert success is True
    assert operator_api.config.training_price == 0.002


def test_set_query_price(operator_api):
    """Test setting query price"""
    success = operator_api.set_query_price(
        0.0015,
        reason="Premium inference quality"
    )
    
    assert success is True
    assert operator_api.config.query_price == 0.0015


def test_pricing_history(operator_api):
    """Test pricing history tracking"""
    # Make several changes
    operator_api.set_storage_price(0.002, "Test 1")
    operator_api.set_storage_price(0.0025, "Test 2")
    operator_api.set_training_price(0.002, "Test 3")
    
    # Check total history
    history = operator_api.get_pricing_history()
    assert len(history) == 3
    
    # Check filtered history
    storage_history = operator_api.get_pricing_history(
        parameter=PricingParameter.STORAGE_PRICE
    )
    assert len(storage_history) == 2
    
    training_history = operator_api.get_pricing_history(
        parameter=PricingParameter.TRAINING_PRICE
    )
    assert len(training_history) == 1


def test_pricing_summary(operator_api):
    """Test getting pricing summary"""
    summary = operator_api.get_pricing_summary()
    
    assert "current_prices" in summary
    assert "bounds" in summary
    assert "metadata" in summary
    
    assert summary["current_prices"]["storage"] == 0.001
    assert summary["metadata"]["bounds_enabled"] is True


def test_bulk_update(operator_api):
    """Test bulk updating multiple prices"""
    updates = {
        "storage": 0.002,
        "training": 0.0015,
        "query": 0.0012
    }
    
    results = operator_api.bulk_update_prices(
        updates,
        reason="Market adjustment"
    )
    
    assert results["storage_price"] is True
    assert results["training_price"] is True
    assert results["query_price"] is True
    
    assert operator_api.config.storage_price == 0.002
    assert operator_api.config.training_price == 0.0015
    assert operator_api.config.query_price == 0.0012


def test_update_bounds(operator_api):
    """Test updating safety bounds"""
    # Update storage price bounds
    success = operator_api.update_bounds(
        PricingParameter.STORAGE_PRICE,
        min_value=0.0005,
        max_value=0.005
    )
    
    assert success is True
    assert operator_api.config.min_storage_price == 0.0005
    assert operator_api.config.max_storage_price == 0.005
    
    # Now can set price within new bounds
    success = operator_api.set_storage_price(0.004)
    assert success is True


def test_bounds_enforcement_prevents_chaos(operator_api):
    """Test that bounds prevent extreme pricing that could break the network"""
    # Try to set absurdly high storage price
    success = operator_api.set_storage_price(1000.0)
    assert success is False  # Rejected
    
    # Try to set absurdly low storage price
    success = operator_api.set_storage_price(0.0000001)
    assert success is False  # Rejected
    
    # Try to set negative price
    success = operator_api.set_storage_price(-1.0)
    assert success is False  # Rejected
    
    # Config should be unchanged
    assert operator_api.config.storage_price == 0.001


def test_bounds_prevent_malicious_block_rewards(operator_api):
    """Test bounds prevent malicious block reward manipulation"""
    # Try to set absurdly high reward target
    success = operator_api.set_block_reward_target(1000000.0)
    assert success is False
    
    # Try to set zero reward
    success = operator_api.set_block_reward_target(0.0)
    assert success is False
    
    # Should remain at default
    assert operator_api.config.block_reward_target == 10.0


def test_reset_to_defaults(operator_api):
    """Test resetting to default prices"""
    # Change prices
    operator_api.set_storage_price(0.005)
    operator_api.set_training_price(0.008)
    
    # Reset
    success = operator_api.reset_to_defaults()
    assert success is True
    
    # Should be back to defaults
    assert operator_api.config.storage_price == 0.001
    assert operator_api.config.training_price == 0.001


def test_config_validation(operator_api):
    """Test configuration validation"""
    # Valid config
    errors = operator_api.config.validate()
    assert len(errors) == 0
    
    # Invalid config
    invalid_config = PricingConfig()
    invalid_config.storage_price = 100.0  # Way above max
    errors = invalid_config.validate()
    assert len(errors) > 0


def test_export_import_config(operator_api):
    """Test exporting and importing configuration"""
    # Make changes
    operator_api.set_storage_price(0.002)
    operator_api.set_training_price(0.0015)
    
    # Export
    data = operator_api.export_config()
    assert "config" in data
    assert "history" in data
    
    # Create new API and import
    new_api = NodeOperatorAPI()
    success = new_api.import_config(data)
    
    assert success is True
    assert new_api.config.storage_price == 0.002
    assert new_api.config.training_price == 0.0015


def test_update_tracking(operator_api):
    """Test that updates are tracked"""
    assert operator_api.config.update_count == 0
    
    operator_api.set_storage_price(0.002)
    assert operator_api.config.update_count == 1
    
    operator_api.set_training_price(0.0015)
    assert operator_api.config.update_count == 2
    
    operator_api.set_query_price(0.0012)
    assert operator_api.config.update_count == 3


def test_bounds_disabled_mode():
    """Test API with bounds disabled (not recommended)"""
    api = NodeOperatorAPI(enable_bounds=False)
    
    # Should allow any value (dangerous!)
    success = api.set_storage_price(1000.0)
    # Will still work, but not recommended
    assert api.config.storage_price == 1000.0


def test_pricing_history_change_percent(operator_api):
    """Test that change percentages are calculated correctly"""
    operator_api.set_storage_price(0.002)  # 100% increase
    
    history = operator_api.get_pricing_history(limit=1)
    assert len(history) == 1
    assert history[0].change_percent == pytest.approx(100.0)


def test_pricing_target_absolute():
    """Test setting absolute pricing target"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    success = api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.ABSOLUTE_TARGET,
        target_value=0.002
    )
    
    assert success is True
    targets = api.get_pricing_targets()
    assert PricingParameter.STORAGE_PRICE in targets
    assert targets[PricingParameter.STORAGE_PRICE].target_value == 0.002


def test_pricing_target_relative_increase():
    """Test setting relative increase target"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    success = api.set_pricing_target(
        PricingParameter.BLOCK_REWARD_TARGET,
        PricingStrategy.RELATIVE_INCREASE,
        percentage_change=0.5
    )
    
    assert success is True
    targets = api.get_pricing_targets()
    assert PricingParameter.BLOCK_REWARD_TARGET in targets
    assert targets[PricingParameter.BLOCK_REWARD_TARGET].percentage_change == 0.5


def test_pricing_target_temporary():
    """Test temporary adjustment"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    success = api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.TEMPORARY_ADJUSTMENT,
        percentage_change=-0.2,
        duration_blocks=100
    )
    
    assert success is True
    targets = api.get_pricing_targets()
    target = targets[PricingParameter.STORAGE_PRICE]
    assert target.blocks_remaining == 100
    assert target.percentage_change == -0.2


def test_apply_as_block_leader():
    """Test applying targets as block leader"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    # Set target
    api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.ABSOLUTE_TARGET,
        target_value=0.002
    )
    
    # Current network price is lower
    current_prices = {
        'storage_price': 0.001,
        'block_reward': 10.0
    }
    
    # Apply as leader
    adjustments = api.apply_as_block_leader(current_prices)
    
    # Should get adjustment to increase
    assert len(adjustments) == 1
    assert adjustments[0]['parameter'] == PricingParameter.STORAGE_PRICE
    assert adjustments[0]['increase'] is True


def test_block_leader_statistics():
    """Test block leader statistics tracking"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    # Set target
    api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.ABSOLUTE_TARGET,
        target_value=0.002
    )
    
    # Apply multiple times
    current_prices = {'storage_price': 0.001, 'block_reward': 10.0}
    
    for _ in range(3):
        api.apply_as_block_leader(current_prices)
    
    # Check statistics
    stats = api.get_leader_statistics()
    assert stats['times_selected'] == 3
    assert stats['adjustments_made'] == 3
    assert stats['active_targets'] == 1


def test_temporary_adjustment_expiry():
    """Test that temporary adjustments expire"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    # Set temporary adjustment for 2 blocks
    api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.TEMPORARY_ADJUSTMENT,
        percentage_change=-0.2,
        duration_blocks=2
    )
    
    current_prices = {'storage_price': 0.001, 'block_reward': 10.0}
    
    # Apply twice
    adjustments1 = api.apply_as_block_leader(current_prices)
    assert len(adjustments1) == 1  # Should apply
    
    adjustments2 = api.apply_as_block_leader(current_prices)
    assert len(adjustments2) == 1  # Should apply
    
    # Third time should be expired
    adjustments3 = api.apply_as_block_leader(current_prices)
    assert len(adjustments3) == 0  # Expired


def test_remove_pricing_target():
    """Test removing a pricing target"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    # Set target
    api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.ABSOLUTE_TARGET,
        target_value=0.002
    )
    
    # Remove target
    success = api.remove_pricing_target(PricingParameter.STORAGE_PRICE)
    assert success is True
    
    # Should be gone
    targets = api.get_pricing_targets()
    assert PricingParameter.STORAGE_PRICE not in targets


def test_multiple_pricing_targets():
    """Test setting multiple pricing targets"""
    from dessin.networking.node_operator_api import PricingStrategy
    
    api = NodeOperatorAPI()
    
    # Set multiple targets
    api.set_pricing_target(
        PricingParameter.STORAGE_PRICE,
        PricingStrategy.ABSOLUTE_TARGET,
        target_value=0.002
    )
    
    api.set_pricing_target(
        PricingParameter.BLOCK_REWARD_TARGET,
        PricingStrategy.RELATIVE_INCREASE,
        percentage_change=0.3
    )
    
    # Both should be active
    targets = api.get_pricing_targets()
    assert len(targets) == 2
    
    # Apply as leader
    current_prices = {'storage_price': 0.001, 'block_reward': 10.0}
    adjustments = api.apply_as_block_leader(current_prices)
    
    # Should get both adjustments
    assert len(adjustments) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
