"""
Tests for Block Time Scaling Flexibility
"""

import math

import pytest
from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager
from dessin.runtime.config import ConsensusConfig


@pytest.fixture
def manager():
    """Create a block time manager with logarithmic scaling"""
    config = ConsensusConfig()
    return DynamicBlockTimeManager(config, use_logarithmic_scaling=True)


@pytest.fixture
def step_manager():
    """Create a block time manager with step-based scaling"""
    config = ConsensusConfig()
    return DynamicBlockTimeManager(config, use_logarithmic_scaling=False)


def test_logarithmic_scaling_tiny_models(manager):
    """Test scaling for very small models (1-10 MB)"""
    MB = 1024 * 1024
    
    # 1 MB
    factor = manager.calculate_model_size_factor(1 * MB)
    assert factor == 1.0
    
    # 5 MB
    factor = manager.calculate_model_size_factor(5 * MB)
    assert factor == 1.0
    
    # 10 MB (baseline)
    factor = manager.calculate_model_size_factor(10 * MB)
    assert factor == 1.0


def test_logarithmic_scaling_small_models(manager):
    """Test scaling for small models (10-100 MB)"""
    MB = 1024 * 1024
    
    # 50 MB
    factor = manager.calculate_model_size_factor(50 * MB)
    assert 1.5 <= factor <= 3.0
    
    # 100 MB
    factor = manager.calculate_model_size_factor(100 * MB)
    assert 2.5 <= factor <= 5.0


def test_logarithmic_scaling_medium_models(manager):
    """Test scaling for medium models (100 MB - 10 GB)"""
    MB = 1024 * 1024
    GB = 1024 * MB
    
    # 1 GB
    factor_1gb = manager.calculate_model_size_factor(1 * GB)
    assert 8.0 <= factor_1gb <= 12.0
    
    # 5 GB
    factor_5gb = manager.calculate_model_size_factor(5 * GB)
    assert 14.0 <= factor_5gb <= 20.0
    
    # 10 GB
    factor_10gb = manager.calculate_model_size_factor(10 * GB)
    assert 18.0 <= factor_10gb <= 25.0
    
    # Should increase with size
    assert factor_5gb > factor_1gb
    assert factor_10gb > factor_5gb


def test_logarithmic_scaling_large_models(manager):
    """Test scaling for large models (10-100 GB)"""
    GB = 1024 * 1024 * 1024
    
    # 50 GB
    factor_50gb = manager.calculate_model_size_factor(50 * GB)
    assert 25.0 <= factor_50gb <= 35.0
    
    # 100 GB
    factor_100gb = manager.calculate_model_size_factor(100 * GB)
    assert 30.0 <= factor_100gb <= 40.0
    
    assert factor_100gb > factor_50gb


def test_logarithmic_scaling_huge_models(manager):
    """Test scaling for huge models (100 GB - 1 TB)"""
    GB = 1024 * 1024 * 1024
    TB = 1024 * GB
    
    # 500 GB
    factor_500gb = manager.calculate_model_size_factor(500 * GB)
    assert 40.0 <= factor_500gb <= 50.0
    
    # 1 TB
    factor_1tb = manager.calculate_model_size_factor(1 * TB)
    assert 45.0 <= factor_1tb <= 55.0
    
    assert factor_1tb > factor_500gb


def test_logarithmic_scaling_massive_models(manager):
    """Test scaling for massive models (1-10 TB)"""
    TB = 1024 * 1024 * 1024 * 1024
    
    # 5 TB
    factor_5tb = manager.calculate_model_size_factor(5 * TB)
    assert 55.0 <= factor_5tb <= 70.0
    
    # 10 TB
    factor_10tb = manager.calculate_model_size_factor(10 * TB)
    assert 63.0 <= factor_10tb <= 75.0
    
    assert factor_10tb > factor_5tb


def test_logarithmic_scaling_extreme_models(manager):
    """Test scaling for extreme models (10+ TB)"""
    TB = 1024 * 1024 * 1024 * 1024
    
    # 50 TB
    factor_50tb = manager.calculate_model_size_factor(50 * TB)
    assert 75.0 <= factor_50tb <= 95.0
    
    # 100 TB
    factor_100tb = manager.calculate_model_size_factor(100 * TB)
    assert 83.0 <= factor_100tb <= 105.0
    
    # 1 PB (petabyte!)
    factor_1pb = manager.calculate_model_size_factor(1024 * TB)
    assert 100.0 <= factor_1pb <= 150.0


def test_max_scaling_factor_cap(manager):
    """Test that scaling factor is capped at maximum"""
    TB = 1024 * 1024 * 1024 * 1024
    
    # Even for absurdly large models, should not exceed max
    factor = manager.calculate_model_size_factor(10000 * TB)
    assert factor <= manager.max_scaling_factor


def test_smooth_scaling_progression(manager):
    """Test that scaling increases smoothly, no sudden jumps"""
    MB = 1024 * 1024
    GB = 1024 * MB
    
    # Test at various points
    sizes = [
        10 * MB, 50 * MB, 100 * MB, 500 * MB,
        1 * GB, 5 * GB, 10 * GB, 50 * GB, 100 * GB
    ]
    
    factors = [manager.calculate_model_size_factor(size) for size in sizes]
    
    # Each factor should be larger than the previous
    for i in range(1, len(factors)):
        assert factors[i] > factors[i-1], f"Factor should increase: {factors[i-1]} -> {factors[i]}"
        
        # Check that increases are reasonable (no more than 2x per step)
        ratio = factors[i] / factors[i-1]
        assert ratio < 2.5, f"Jump too large: {factors[i-1]} -> {factors[i]} (ratio: {ratio})"


def test_step_based_scaling(step_manager):
    """Test step-based scaling (backward compatibility)"""
    MB = 1024 * 1024
    GB = 1024 * MB
    TB = 1024 * GB
    
    # Test various sizes
    assert step_manager.calculate_model_size_factor(5 * MB, use_logarithmic=False) == 1.0
    assert step_manager.calculate_model_size_factor(50 * MB, use_logarithmic=False) == 1.2
    assert step_manager.calculate_model_size_factor(200 * MB, use_logarithmic=False) == 1.5
    assert step_manager.calculate_model_size_factor(2 * GB, use_logarithmic=False) == 3.0
    assert step_manager.calculate_model_size_factor(20 * GB, use_logarithmic=False) == 8.0
    assert step_manager.calculate_model_size_factor(75 * GB, use_logarithmic=False) == 15.0
    assert step_manager.calculate_model_size_factor(200 * GB, use_logarithmic=False) == 30.0
    assert step_manager.calculate_model_size_factor(2 * TB, use_logarithmic=False) == 100.0


def test_zero_size_handling(manager):
    """Test handling of zero or negative sizes"""
    assert manager.calculate_model_size_factor(0) == 1.0
    assert manager.calculate_model_size_factor(-100) == 1.0


def test_scaling_with_different_configs():
    """Test scaling with different configuration parameters"""
    config = ConsensusConfig()
    
    # Manager with lower max scaling
    manager_low = DynamicBlockTimeManager(config, max_scaling_factor=80.0)
    
    # Manager with higher max scaling
    manager_high = DynamicBlockTimeManager(config, max_scaling_factor=5000.0)
    
    TB = 1024 * 1024 * 1024 * 1024
    
    # For massive models, low cap should limit scaling
    factor_low = manager_low.calculate_model_size_factor(1000 * TB, max_factor=80.0)
    factor_high = manager_high.calculate_model_size_factor(1000 * TB, max_factor=5000.0)
    
    assert factor_low <= 80.0
    assert factor_high <= 5000.0
    assert factor_high >= factor_low


def test_realistic_model_sizes():
    """Test with realistic model sizes"""
    config = ConsensusConfig()
    manager = DynamicBlockTimeManager(config)
    
    MB = 1024 * 1024
    GB = 1024 * MB
    
    # GPT-2 Small (~500 MB)
    gpt2_small = manager.calculate_model_size_factor(500 * MB)
    assert 6.0 <= gpt2_small <= 9.0
    
    # GPT-2 XL (~6 GB)
    gpt2_xl = manager.calculate_model_size_factor(6 * GB)
    assert 15.0 <= gpt2_xl <= 21.0
    
    # GPT-3 (~350 GB)
    gpt3 = manager.calculate_model_size_factor(350 * GB)
    assert 38.0 <= gpt3 <= 46.0
    
    # GPT-4 (estimated ~1.8 TB)
    gpt4 = manager.calculate_model_size_factor(1800 * GB)
    assert 49.0 <= gpt4 <= 57.0
    
    # Scaling should reflect complexity
    assert gpt2_xl > gpt2_small
    assert gpt3 > gpt2_xl
    assert gpt4 > gpt3


def test_scaling_formula_documentation():
    """Test and document the scaling formula behavior"""
    config = ConsensusConfig()
    manager = DynamicBlockTimeManager(config)
    
    MB = 1024 * 1024
    GB = 1024 * MB
    TB = 1024 * GB
    
    test_cases = [
        (1 * MB, "1 MB (tiny)"),
        (10 * MB, "10 MB (baseline)"),
        (100 * MB, "100 MB (small)"),
        (1 * GB, "1 GB (medium)"),
        (10 * GB, "10 GB (large)"),
        (100 * GB, "100 GB (huge)"),
        (1 * TB, "1 TB (massive)"),
        (10 * TB, "10 TB (extreme)"),
        (100 * TB, "100 TB (petascale)"),
    ]
    
    print("\nScaling Factor Table:")
    print("-" * 50)
    for size, description in test_cases:
        factor = manager.calculate_model_size_factor(size)
        print(f"{description:25s} -> {factor:6.2f}x")


def test_adaptive_timing_increase():
    """Test adaptive timing increases block time when validation is slow"""
    config = ConsensusConfig()
    config.training_block_time_minutes = 60.0  # 3600 seconds
    config.max_block_time_seconds = 10_000  # allow +5% above 3600 without clamping to default 3600 cap
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    # Initial block time
    initial_time = 3600.0
    manager.current_block_time_seconds = initial_time
    
    # Validation takes 85% of block time (> 80% threshold)
    validation_time = initial_time * 0.85
    
    new_time, reason = manager.adjust_block_time_adaptive(validation_time)
    
    # Strong lengthening under stress (default +15%)
    expected_time = initial_time * (1 + manager.ramp_down_capacity_pct)
    assert new_time == pytest.approx(expected_time)
    assert "85.0%" in reason


def test_adaptive_timing_decrease():
    """Test adaptive timing decreases block time when validation is fast"""
    config = ConsensusConfig()
    config.training_block_time_minutes = 60.0
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    # Initial block time
    initial_time = 3600.0
    manager.current_block_time_seconds = initial_time
    
    # Validation takes 30% of block time (< 40% threshold)
    validation_time = initial_time * 0.30
    
    new_time, reason = manager.adjust_block_time_adaptive(validation_time)
    
    # Gentle shortening when slack (default −5%)
    expected_time = initial_time * (1 - manager.ramp_up_capacity_pct)
    assert new_time == pytest.approx(expected_time)
    assert "30.0%" in reason


def test_adaptive_timing_maintain():
    """Test adaptive timing maintains block time when validation is optimal"""
    config = ConsensusConfig()
    config.training_block_time_minutes = 60.0
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    # Initial block time
    initial_time = 3600.0
    manager.current_block_time_seconds = initial_time
    
    # Validation takes 60% of block time (between 40% and 80%)
    validation_time = initial_time * 0.60
    
    new_time, reason = manager.adjust_block_time_adaptive(validation_time)
    
    # Should maintain current time
    assert new_time == initial_time
    assert "60.0%" in reason
    assert "optimal" in reason.lower()


def test_adaptive_timing_minimum_floor():
    """Test adaptive timing respects minimum block time"""
    config = ConsensusConfig()
    config.training_block_time_minutes = 60.0
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    # Set block time very close to minimum (would decrease below minimum)
    manager.current_block_time_seconds = 42.0
    
    # Validation is fast (20% of block time)
    validation_time = 42.0 * 0.20
    
    new_time, reason = manager.adjust_block_time_adaptive(validation_time)
    
    # Should hit minimum floor (40 seconds): 42 * (1 - 0.05) < 40
    assert new_time == 40.0
    assert "minimum" in reason.lower()


def test_adaptive_timing_multiple_adjustments():
    """Test adaptive timing handles multiple consecutive adjustments"""
    config = ConsensusConfig()
    config.training_block_time_minutes = 60.0
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    initial_time = 100.0
    manager.current_block_time_seconds = initial_time
    
    ru = manager.ramp_down_capacity_pct
    rd = manager.ramp_up_capacity_pct
    # Slow validation -> conservative lengthen
    new_time1, _ = manager.adjust_block_time_adaptive(100.0 * 0.85)
    assert new_time1 == pytest.approx(float(math.ceil(100.0 * (1 + ru) - 1e-9)))
    
    new_time2, _ = manager.adjust_block_time_adaptive(new_time1 * 0.85)
    assert new_time2 == pytest.approx(float(math.ceil(new_time1 * (1 + ru) - 1e-9)))
    
    # Fast validation -> aggressive shorten
    new_time3, _ = manager.adjust_block_time_adaptive(new_time2 * 0.30)
    assert new_time3 == pytest.approx(float(math.ceil(new_time2 * (1 - rd) - 1e-9)))


def test_adaptive_vs_model_based():
    """Test that adaptive and model-based can coexist"""
    config = ConsensusConfig()
    
    # Adaptive manager
    adaptive = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True
    )
    
    # Model-based manager
    model_based = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=False,
        use_logarithmic_scaling=True
    )
    
    # Both should initialize successfully
    assert adaptive.use_adaptive_timing is True
    assert model_based.use_adaptive_timing is False
    
    # Model-based should still have scaling
    MB = 1024 * 1024
    GB = 1024 * MB
    factor = model_based.calculate_model_size_factor(1 * GB)
    assert factor > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
