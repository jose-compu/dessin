"""
Adaptive Block Time Demo
========================

Demonstrates the simplified adaptive block time adjustment based on
validation performance rather than model size prediction.

Rules (asymmetric defaults):
- Under stress (validation > 80% of slot): lengthen interval slowly (+5%/step by default)
- With slack (validation < 40% of slot): shorten interval aggressively (−15%/step by default)
- Otherwise maintain (tunable via DESSIN_BLOCK_TIME_RAMP_* env vars)
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager
from dessin.runtime.config import ConsensusConfig
import random


def simulate_validation_scenarios():
    """Simulate various validation scenarios"""
    
    print("=" * 80)
    print("Adaptive Block Time Adjustment Demo")
    print("=" * 80)
    print()
    
    # Create manager with simple adaptive timing
    config = ConsensusConfig()
    config.training_block_time_minutes = 30.0  # Start with 30 minutes
    
    manager = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    print()
    print("Starting Configuration:")
    print(f"  Initial block time: {manager.current_block_time_seconds:.1f} seconds")
    print(f"  Minimum block time: {manager.min_block_time_seconds:.1f} seconds")
    print(f"  Increase threshold: {manager.increase_threshold*100:.0f}%")
    print(f"  Decrease threshold: {manager.decrease_threshold*100:.0f}%")
    print(
        f"  Slack (shorten slot): −{manager.ramp_up_capacity_pct*100:.1f}% / step; "
        f"stress (lengthen slot): +{manager.ramp_down_capacity_pct*100:.1f}% / step"
    )
    print()
    
    # Scenario 1: Slow validation (network is overloaded)
    print("─" * 80)
    print("Scenario 1: Network Overload - Validation Taking Too Long")
    print("─" * 80)
    print()
    
    for i in range(5):
        current_time = manager.current_block_time_seconds
        # Validation takes 85-95% of block time
        validation_time = current_time * random.uniform(0.85, 0.95)
        
        print(f"Block {i+1}:")
        print(f"  Block time: {current_time:.1f}s")
        print(f"  Validation: {validation_time:.1f}s ({validation_time/current_time*100:.1f}%)")
        
        new_time, reason = manager.adjust_block_time_adaptive(validation_time)
        print(f"  → New block time: {new_time:.1f}s")
        print()
    
    # Scenario 2: Fast validation (network has capacity)
    print("─" * 80)
    print("Scenario 2: Network Has Capacity - Validation Very Fast")
    print("─" * 80)
    print()
    
    for i in range(5):
        current_time = manager.current_block_time_seconds
        # Validation takes 20-35% of block time
        validation_time = current_time * random.uniform(0.20, 0.35)
        
        print(f"Block {i+1}:")
        print(f"  Block time: {current_time:.1f}s")
        print(f"  Validation: {validation_time:.1f}s ({validation_time/current_time*100:.1f}%)")
        
        new_time, reason = manager.adjust_block_time_adaptive(validation_time)
        print(f"  → New block time: {new_time:.1f}s")
        print()
    
    # Scenario 3: Optimal validation (stable network)
    print("─" * 80)
    print("Scenario 3: Stable Network - Optimal Validation Time")
    print("─" * 80)
    print()
    
    for i in range(5):
        current_time = manager.current_block_time_seconds
        # Validation takes 50-70% of block time (optimal range)
        validation_time = current_time * random.uniform(0.50, 0.70)
        
        print(f"Block {i+1}:")
        print(f"  Block time: {current_time:.1f}s")
        print(f"  Validation: {validation_time:.1f}s ({validation_time/current_time*100:.1f}%)")
        
        new_time, reason = manager.adjust_block_time_adaptive(validation_time)
        print(f"  → New block time: {new_time:.1f}s")
        print()
    
    # Scenario 4: Mixed conditions
    print("─" * 80)
    print("Scenario 4: Real-World Mixed Conditions")
    print("─" * 80)
    print()
    
    # Reset to a moderate starting point
    manager.current_block_time_seconds = 300.0  # 5 minutes
    
    conditions = [
        ("High load", 0.85, 0.95),
        ("Normal load", 0.50, 0.70),
        ("Low load", 0.20, 0.35),
        ("Spike", 0.90, 0.95),
        ("Recovery", 0.40, 0.60),
        ("Stable", 0.55, 0.65),
        ("Low", 0.25, 0.35),
        ("Very high", 0.88, 0.92),
    ]
    
    for i, (condition, min_util, max_util) in enumerate(conditions):
        current_time = manager.current_block_time_seconds
        validation_time = current_time * random.uniform(min_util, max_util)
        utilization = validation_time / current_time
        
        print(f"Block {i+1} ({condition}):")
        print(f"  Block time: {current_time:.1f}s")
        print(f"  Validation: {validation_time:.1f}s ({utilization*100:.1f}%)")
        
        new_time, reason = manager.adjust_block_time_adaptive(validation_time)
        
        change = ""
        if new_time > current_time:
            change = f"↑ +{(new_time-current_time)/current_time*100:.1f}%"
        elif new_time < current_time:
            change = f"↓ -{(current_time-new_time)/current_time*100:.1f}%"
        else:
            change = "= maintained"
        
        print(f"  → New block time: {new_time:.1f}s {change}")
        print()
    
    # Scenario 5: Hitting minimum floor
    print("─" * 80)
    print("Scenario 5: Hitting Minimum Block Time Floor")
    print("─" * 80)
    print()
    
    # Set low block time
    manager.current_block_time_seconds = 60.0
    
    for i in range(5):
        current_time = manager.current_block_time_seconds
        # Very fast validation (10-20% of block time)
        validation_time = current_time * random.uniform(0.10, 0.20)
        
        print(f"Block {i+1}:")
        print(f"  Block time: {current_time:.1f}s")
        print(f"  Validation: {validation_time:.1f}s ({validation_time/current_time*100:.1f}%)")
        
        new_time, reason = manager.adjust_block_time_adaptive(validation_time)
        
        if new_time == manager.min_block_time_seconds:
            print(f"  → Hit minimum floor: {new_time:.1f}s ⚠️")
        else:
            print(f"  → New block time: {new_time:.1f}s")
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Adaptive Block Time Features:")
    print("  ✅ Reacts to actual validation performance (not predictions)")
    print("  ✅ Simple 10% adjustment per block")
    print("  ✅ Minimum floor prevents too-fast blocks")
    print("  ✅ Optimal range (40%-80%) maintains stability")
    print("  ✅ Automatic adaptation to network conditions")
    print()
    print("Benefits:")
    print("  • No need to predict model size")
    print("  • Works with any model size")
    print("  • Adapts to hardware differences")
    print("  • Handles network congestion")
    print("  • Simple and predictable")
    print()
    print(f"Final block time: {manager.current_block_time_seconds:.1f}s")
    print()


def compare_approaches():
    """Compare adaptive vs model-based approaches"""
    
    print("=" * 80)
    print("Comparison: Adaptive vs Model-Based Block Time")
    print("=" * 80)
    print()
    
    config = ConsensusConfig()
    
    # Adaptive approach
    adaptive = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=True,
        min_block_time_seconds=40.0
    )
    
    # Model-based approach
    model_based = DynamicBlockTimeManager(
        config,
        use_adaptive_timing=False,
        use_logarithmic_scaling=True
    )
    
    print("Adaptive Approach:")
    print("  Pros:")
    print("    • Reacts to actual performance")
    print("    • Simple and predictable")
    print("    • Works with any model")
    print("    • Adapts to hardware")
    print("  Cons:")
    print("    • Reactive (not predictive)")
    print("    • May oscillate initially")
    print()
    
    print("Model-Based Approach:")
    print("  Pros:")
    print("    • Predictive (based on model size)")
    print("    • Smooth scaling curve")
    print("    • No oscillation")
    print("  Cons:")
    print("    • Requires model size info")
    print("    • Doesn't adapt to hardware")
    print("    • Complex formula")
    print()
    
    print("Recommendation:")
    print("  Use ADAPTIVE for:")
    print("    • Production networks")
    print("    • Mixed hardware")
    print("    • Variable load")
    print()
    print("  Use MODEL-BASED for:")
    print("    • Testing/benchmarking")
    print("    • Homogeneous hardware")
    print("    • Predictable workloads")
    print()


if __name__ == "__main__":
    simulate_validation_scenarios()
    print()
    compare_approaches()
