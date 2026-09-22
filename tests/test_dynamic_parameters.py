"""
Tests for Dynamic Parameters
"""

import pytest
from dessin.economics.dynamic_parameters import (
    DynamicParameterManager,
    ParameterType,
    ParameterAdjustment
)


@pytest.fixture
def param_manager():
    """Create a parameter manager"""
    return DynamicParameterManager(
        initial_block_reward=10.0,
        initial_storage_price=0.001,
        max_adjustment_percent=0.1  # 0.1% per block
    )


def test_parameter_manager_initialization(param_manager):
    """Test parameter manager initializes correctly"""
    assert param_manager.block_reward == 10.0
    assert param_manager.storage_price == 0.001
    assert param_manager.max_adjustment_percent == 0.001


def test_adjust_block_reward_increase(param_manager):
    """Test increasing block reward"""
    old_reward = param_manager.block_reward
    
    adjustment = param_manager.adjust_block_reward(
        block_index=1,
        block_leader="0x1234",
        increase=True,
        reason="Attract more miners"
    )
    
    assert adjustment is not None
    assert param_manager.block_reward > old_reward
    assert adjustment.parameter_type == ParameterType.BLOCK_REWARD
    assert adjustment.adjustment_percent > 0


def test_adjust_block_reward_decrease(param_manager):
    """Test decreasing block reward"""
    old_reward = param_manager.block_reward
    
    adjustment = param_manager.adjust_block_reward(
        block_index=1,
        block_leader="0x1234",
        increase=False,
        reason="Reduce inflation"
    )
    
    assert adjustment is not None
    assert param_manager.block_reward < old_reward
    assert adjustment.adjustment_percent < 0


def test_adjust_storage_price_increase(param_manager):
    """Test increasing storage price"""
    old_price = param_manager.storage_price
    
    adjustment = param_manager.adjust_storage_price(
        block_index=1,
        block_leader="0x1234",
        increase=True,
        reason="High storage utilization"
    )
    
    assert adjustment is not None
    assert param_manager.storage_price > old_price


def test_adjust_storage_price_decrease(param_manager):
    """Test decreasing storage price"""
    old_price = param_manager.storage_price
    
    adjustment = param_manager.adjust_storage_price(
        block_index=1,
        block_leader="0x1234",
        increase=False,
        reason="Low storage utilization"
    )
    
    assert adjustment is not None
    assert param_manager.storage_price < old_price


def test_parameter_bounds(param_manager):
    """Test that parameters respect bounds"""
    # Increase many times
    for i in range(10000):
        param_manager.adjust_block_reward(i, "0x1234", True)
    
    # Should hit max bound
    assert param_manager.block_reward <= param_manager.max_block_reward
    
    # Decrease many times
    for i in range(20000):
        param_manager.adjust_block_reward(i, "0x1234", False)
    
    # Should hit min bound
    assert param_manager.block_reward >= param_manager.min_block_reward


def test_adjustment_history(param_manager):
    """Test adjustment history tracking"""
    # Make several adjustments
    for i in range(5):
        param_manager.adjust_block_reward(i, "0x1234", True)
    
    for i in range(5, 10):
        param_manager.adjust_storage_price(i, "0x1234", False)
    
    # Check total history
    history = param_manager.get_adjustment_history()
    assert len(history) == 10
    
    # Check filtered history
    block_reward_history = param_manager.get_adjustment_history(
        parameter_type=ParameterType.BLOCK_REWARD
    )
    assert len(block_reward_history) == 5
    
    storage_price_history = param_manager.get_adjustment_history(
        parameter_type=ParameterType.STORAGE_PRICE
    )
    assert len(storage_price_history) == 5


def test_parameter_trend(param_manager):
    """Test trend analysis"""
    # Increase block reward 10 times
    for i in range(10):
        param_manager.adjust_block_reward(i, "0x1234", True)
    
    trend = param_manager.get_parameter_trend(
        ParameterType.BLOCK_REWARD,
        blocks=10
    )
    
    assert trend["increases"] == 10
    assert trend["decreases"] == 0
    assert trend["net_change_percent"] > 0


def test_get_current_parameters(param_manager):
    """Test getting current parameter values"""
    params = param_manager.get_current_parameters()
    
    assert "block_reward" in params
    assert "storage_price" in params
    assert "training_payment" in params
    assert params["block_reward"] == 10.0


def test_suggest_adjustment():
    """Test adjustment suggestions based on metrics"""
    manager = DynamicParameterManager()
    
    # High storage utilization should suggest increase
    metrics = {"storage_utilization": 0.9}
    suggestion = manager.suggest_adjustment(
        ParameterType.STORAGE_PRICE,
        metrics
    )
    assert suggestion is True
    
    # Low storage utilization should suggest decrease
    metrics = {"storage_utilization": 0.2}
    suggestion = manager.suggest_adjustment(
        ParameterType.STORAGE_PRICE,
        metrics
    )
    assert suggestion is False
    
    # Normal utilization should suggest no change
    metrics = {"storage_utilization": 0.5}
    suggestion = manager.suggest_adjustment(
        ParameterType.STORAGE_PRICE,
        metrics
    )
    assert suggestion is None


def test_small_adjustment_magnitude(param_manager):
    """Test that adjustments are small (0.1%)"""
    old_reward = param_manager.block_reward
    
    param_manager.adjust_block_reward(1, "0x1234", True)
    
    new_reward = param_manager.block_reward
    change_percent = abs((new_reward - old_reward) / old_reward * 100)
    
    # Should be approximately 0.1%
    assert 0.09 < change_percent < 0.11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
