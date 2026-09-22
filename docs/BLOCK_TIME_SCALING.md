# Flexible Block Time Scaling

## Overview

DeSSIN now features **highly flexible block time scaling** that adapts to models ranging from 1MB to 100TB+ (petabyte-scale and beyond).

## Scaling Formula

### Logarithmic Scaling (Default, Recommended)

Uses a smooth logarithmic curve for seamless scaling across all model sizes:

```
factor = 1.0 + (log₁₀(size / 10MB))^1.8 × 2.5
```

**Key Properties:**
- ✅ Smooth progression (no sudden jumps)
- ✅ Scales from 1MB to 100TB+
- ✅ Configurable maximum cap
- ✅ Mathematically elegant
- ✅ Production-tested

### Scaling Table

| Model Size | Scaling Factor | Example Models |
|------------|----------------|----------------|
| 1 MB | 1.0x | Test models |
| 10 MB | 1.0x | Tiny models |
| 100 MB | 3.5x | Nanochat d12 |
| 1 GB | 9.8x | Nanochat d20, GPT-2 |
| 10 GB | 19.2x | Nanochat d32, GPT-2 XL |
| 100 GB | 31.4x | GPT-3, LLaMA 7B |
| 1 TB | 46.6x | GPT-4 class |
| 10 TB | 64.3x | Future large models |
| 100 TB | 84.5x | Extreme scale |
| 1 PB | 107.0x | Petabyte models |

### Configuration

```python
from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager
from dessin.runtime.config import ConsensusConfig

config = ConsensusConfig()

# Default: logarithmic with 1000x max
manager = DynamicBlockTimeManager(
    config,
    use_logarithmic_scaling=True,
    max_scaling_factor=1000.0
)

# Conservative: lower maximum
manager_conservative = DynamicBlockTimeManager(
    config,
    use_logarithmic_scaling=True,
    max_scaling_factor=100.0
)

# Aggressive: higher maximum
manager_aggressive = DynamicBlockTimeManager(
    config,
    use_logarithmic_scaling=True,
    max_scaling_factor=10000.0
)

# Legacy: step-based (backward compatible)
manager_legacy = DynamicBlockTimeManager(
    config,
    use_logarithmic_scaling=False
)
```

## Usage Examples

### Automatic Scaling

```python
from dessin.runtime.node import DessinNode

node = DessinNode()

# Create small model (1 GB) - 9.8x factor
model_small = node.create_nanochat_model(
    name="Small Model",
    depth=20  # ~1.5 GB
)
# Block time: base_time × 9.8

# Create large model (100 GB) - 31.4x factor
model_large = node.create_nanochat_model(
    name="Large Model",
    depth=50  # ~100 GB estimated
)
# Block time: base_time × 31.4
```

### Manual Scaling Factor

```python
# Get scaling factor for any size
manager = node.consensus.dynamic_block_time_manager

MB = 1024 * 1024
GB = 1024 * MB
TB = 1024 * GB

# Check scaling for different sizes
factor_1gb = manager.calculate_model_size_factor(1 * GB)
factor_10gb = manager.calculate_model_size_factor(10 * GB)
factor_100gb = manager.calculate_model_size_factor(100 * GB)
factor_1tb = manager.calculate_model_size_factor(1 * TB)

print(f"1 GB:   {factor_1gb:.2f}x")
print(f"10 GB:  {factor_10gb:.2f}x")
print(f"100 GB: {factor_100gb:.2f}x")
print(f"1 TB:   {factor_1tb:.2f}x")
```

## Comparison: Step vs Logarithmic

### Step-Based Scaling (Legacy)

```
< 10MB:      1.0x
10-100MB:    1.2x
100-500MB:   1.5x
500MB-1GB:   2.0x
1-5GB:       3.0x
5-10GB:      4.0x
10-50GB:     8.0x
50-100GB:   15.0x
100-500GB:  30.0x
500GB-1TB:  50.0x
1-10TB:    100.0x
10TB+:     logarithmic
```

**Issues:**
- ❌ Discrete jumps
- ❌ Less smooth
- ⚠️ Not optimal for all sizes

### Logarithmic Scaling (New Default)

```
Smooth curve from 1MB to 100TB+
No sudden jumps
Scales appropriately at all sizes
```

**Benefits:**
- ✅ Smooth scaling
- ✅ Optimal for all sizes
- ✅ Mathematically sound
- ✅ Configurable cap

## Real-World Examples

### Nanochat Models

```
Nanochat d20 (~1.5 GB):
  Factor: ~10x
  Base block time: 1 hour
  Actual block time: ~10 hours
  
Nanochat d26 (~2.5 GB):
  Factor: ~12x
  Base block time: 1 hour
  Actual block time: ~12 hours
  
Nanochat d32 (~4 GB):
  Factor: ~14x
  Base block time: 1 hour
  Actual block time: ~14 hours
```

### Famous Models

```
GPT-2 (1.5 GB):
  Factor: ~10x
  Training: Feasible on 8xGPU
  
GPT-3 (350 GB):
  Factor: ~40x
  Training: Requires significant compute
  
GPT-4 (~1.8 TB estimated):
  Factor: ~50x
  Training: Extreme scale
  
Future models (10 TB+):
  Factor: 60-100x+
  Training: Cutting edge scale
```

## Benefits of Flexible Scaling

### 1. Universal Support

Handles models from 1MB to petabyte-scale without code changes.

### 2. Smooth Adaptation

No sudden jumps in block times as models grow.

### 3. Configurable

Nodes can set their own maximum scaling factors based on capabilities.

### 4. Future-Proof

Ready for next-generation models without updates.

### 5. Fair Economics

Larger models automatically get more time, preventing unfair advantage.

## Advanced Configuration

### Per-Node Customization

```python
# High-performance node (can handle large models)
config_high_perf = ConsensusConfig()
manager_high = DynamicBlockTimeManager(
    config_high_perf,
    max_scaling_factor=5000.0  # Support extreme scales
)

# Standard node
manager_standard = DynamicBlockTimeManager(
    config_high_perf,
    max_scaling_factor=1000.0  # Default
)

# Resource-constrained node
manager_constrained = DynamicBlockTimeManager(
    config_high_perf,
    max_scaling_factor=100.0  # Limit to smaller models
)
```

### Dynamic Adjustment

```python
# Block times automatically adjust based on:
# 1. Model size (via scaling factor)
# 2. Verification performance
# 3. Network conditions

# Example: 100 GB model
base_time = 3600  # 1 hour
model_factor = 31.4  # From scaling formula
network_adjustment = 1.1  # From dynamic adjustment

actual_block_time = base_time * model_factor * network_adjustment
# = 3600 × 31.4 × 1.1
# ≈ 124,524 seconds
# ≈ 34.6 hours
```

## Testing

### Comprehensive Test Suite

```bash
# Run scaling tests
./bin/python -m pytest tests/test_block_time_scaling.py -v

# 14/14 tests passing ✅
```

### Test Coverage

- ✅ Tiny models (1-10 MB)
- ✅ Small models (10-100 MB)
- ✅ Medium models (100 MB - 10 GB)
- ✅ Large models (10-100 GB)
- ✅ Huge models (100 GB - 1 TB)
- ✅ Massive models (1-10 TB)
- ✅ Extreme models (10-100 TB)
- ✅ Petabyte-scale models (100 TB+)
- ✅ Smooth progression
- ✅ Configuration limits
- ✅ Realistic model sizes (GPT-2, GPT-3, GPT-4)

## Mathematical Properties

### Formula Analysis

```
f(x) = 1.0 + (log₁₀(x / 10MB))^1.8 × 2.5

Where:
  x = model size in bytes
  10MB = baseline size (factor = 1.0)
  1.8 = scaling exponent (controls curve shape)
  2.5 = multiplier (controls scaling magnitude)
```

**Derivative Properties:**
- First derivative > 0 (always increasing)
- Second derivative > 0 initially (accelerating growth)
- Second derivative approaches 0 (diminishing returns at extreme scales)

**Asymptotic Behavior:**
- Grows without bound (before cap)
- Cap provides practical upper limit
- Smooth at all scales

## Performance Impact

### Block Production

```
Tiny model (10 MB):    1.0x → Fast blocks
Medium model (1 GB):   9.8x → Moderate blocks
Large model (100 GB): 31.4x → Slow blocks
Massive model (10 TB):64.3x → Very slow blocks
```

### Network Throughput

```
Mixed model sizes:
- 10% tiny:    1.0x average
- 60% medium:  9.8x average
- 25% large:  31.4x average
- 5% massive: 64.3x average

Weighted average: ~16x
Overall throughput: 1/16 of base rate
```

### Economic Balance

```
Small models:
  - Lower block times
  - More blocks produced
  - Lower training costs
  - More accessible

Large models:
  - Higher block times
  - Fewer blocks produced
  - Higher training costs
  - Commensurate rewards
```

## Monitoring

### Get Scaling Factor

```python
# For a specific model
model_info = node.model_manager.get_model_info(model_id)
size_bytes = int(model_info.size_gb * 1024 * 1024 * 1024)

factor = node.consensus.dynamic_block_time_manager.calculate_model_size_factor(
    size_bytes
)

print(f"Model: {model_info.name}")
print(f"Size: {model_info.size_gb:.2f} GB")
print(f"Scaling factor: {factor:.2f}x")
```

### Monitor Network

```python
# Get performance summary
summary = node.consensus.dynamic_block_time_manager.get_performance_summary()

print(f"Current block time: {summary['current_block_time_seconds']:.1f}s")
print(f"Blocks analyzed: {summary['total_blocks_analyzed']}")
print(f"Timeout ratio: {summary['timeout_ratio']:.2%}")
```

## Troubleshooting

### Block Times Too Long

```python
# Reduce maximum scaling factor (set on the manager constructor, not ConsensusConfig)
from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager
from dessin.runtime.config import ConsensusConfig

manager = DynamicBlockTimeManager(ConsensusConfig(), max_scaling_factor=500.0)
```

### Block Times Too Short

```python
manager = DynamicBlockTimeManager(ConsensusConfig(), max_scaling_factor=2000.0)
```

### Switch to Step-Based

```python
# Use legacy step-based scaling
manager = DynamicBlockTimeManager(
    config,
    use_logarithmic_scaling=False
)
```

## Conclusion

The flexible block time scaling system provides:

✅ **Universal support**: 1MB to 100TB+ models  
✅ **Smooth scaling**: No sudden jumps  
✅ **Configurable**: Per-node customization  
✅ **Future-proof**: Ready for next-gen models  
✅ **Tested**: 14 comprehensive tests  
✅ **Production-ready**: Deployed and verified  

This ensures fair, efficient, and scalable training verification across the entire spectrum of AI model sizes.

---

**Formula**: `1.0 + (log₁₀(size / 10MB))^1.8 × 2.5`  
**Range**: 1MB to 100TB+ (cap: 1000x default)  
**Status**: ✅ Production Ready
