# Dynamic Economics & Governance Features

## Overview

DeSSIN now features a dynamic economic system where block leaders can adjust network parameters gradually to respond to market conditions, inspired by other blockchain designs like Tezos and Polkadot.

## Key Features

### 1. Block Leader Parameter Adjustments

Block leaders can adjust key network parameters by **±0.1% per block** (configurable):

- **Block Reward**: Adjust mining incentives
- **Storage Price**: Adjust storage rental costs
- **Training Payment**: Adjust training iteration costs

This allows the network to adapt to:
- Mining rate changes (attract/reduce miners)
- Storage utilization (encourage/discourage storage usage)
- Training demand (incentivize/reduce training costs)

### 2. Open Storage Top-Up

**Anyone can top-up storage for any model**, not just the owner:

- Community members can support models they value
- Prevents models from expiring due to owner oversight
- Creates community-funded model economy
- Complete tracking of all contributors

### 3. Owner-Only Model Management

While storage can be topped up by anyone, **only owners control**:

- Model upload
- Training dataset selection
- Training parameters (batch size, learning rate, etc.)
- Device configuration (CPU/GPU)

This prevents unauthorized modification while allowing community support.

### 4. Minimum Storage Requirements

Model owners must pay for a **minimum number of blocks** (default: 100):

- Prevents spam models
- Ensures serious commitment
- Configurable per node

### 5. CPU/GPU Training Configuration

Nodes can configure training device preferences:

- `CPU`: Force CPU training
- `GPU`: Force GPU training (if available)
- `AUTO`: Automatically select best device

## API Reference

### Block Leader Operations

```python
from dessin.economics.dynamic_parameters import ParameterType

# Increase block reward
node.adjust_network_parameter(
    ParameterType.BLOCK_REWARD,
    increase=True,
    reason="Low mining rate, need to attract miners"
)

# Decrease storage price
node.adjust_network_parameter(
    ParameterType.STORAGE_PRICE,
    increase=False,
    reason="Low storage utilization, reducing price"
)

# Increase training payment
node.adjust_network_parameter(
    ParameterType.TRAINING_PAYMENT,
    increase=True,
    reason="High training demand"
)
```

### Storage Top-Up (Anyone)

```python
# Anyone can top-up storage for any model
success = node.topup_model_storage(
    model_id="nanochat_abc123",
    additional_blocks=100,
    payment=1.5
)

# View contributors
contributors = node.storage_topup.get_contributors(model_id)
# {
#     "0xowner...": 10.0,
#     "0xcontributor1...": 5.0,
#     "0xcontributor2...": 2.5
# }

# View storage statistics
stats = node.storage_topup.get_storage_statistics(model_id)
# {
#     "total_topups": 15,
#     "total_amount": 50.5,
#     "total_blocks_added": 5000,
#     "unique_contributors": 8,
#     "contributors": {...}
# }
```

### Model Creation (Owner Only)

```python
from dessin.nanochat.nanochat_integration import TrainingDevice

# Create model with specific training device
model_id = node.create_nanochat_model(
    name="My Chat Model",
    depth=20,
    dataset_name="fineweb",  # Owner only
    device_batch_size=32,    # Owner only
    training_device=TrainingDevice.GPU,  # Owner only
    storage_blocks=200,      # Minimum 100 enforced
    storage_payment=10.0
)
```

### Parameter Monitoring

```python
# Get current parameters
params = node.dynamic_params.get_current_parameters()
# {
#     "block_reward": 10.015,
#     "storage_price": 0.000998,
#     "training_payment": 0.001002
# }

# Get adjustment history
history = node.dynamic_params.get_adjustment_history(
    parameter_type=ParameterType.STORAGE_PRICE,
    limit=100
)

# Get parameter trend
trend = node.dynamic_params.get_parameter_trend(
    ParameterType.BLOCK_REWARD,
    blocks=100
)
# {
#     "increases": 55,
#     "decreases": 45,
#     "net_change_percent": +0.5,
#     "start_value": 10.0,
#     "end_value": 10.05
# }
```

## Configuration

### Dynamic Parameter Manager

```python
from dessin.economics.dynamic_parameters import DynamicParameterManager

manager = DynamicParameterManager(
    initial_block_reward=10.0,
    initial_storage_price=0.001,
    initial_training_payment=0.001,
    max_adjustment_percent=0.1,  # 0.1% max per block
    min_block_reward=0.1,
    max_block_reward=100.0,
    min_storage_price=0.0001,
    max_storage_price=0.1
)
```

### Nanochat Integration

```python
from dessin.nanochat.nanochat_integration import NanochatIntegration, TrainingDevice

integration = NanochatIntegration(
    model_manager=model_manager,
    default_training_device=TrainingDevice.GPU
)

# Set minimum storage blocks
integration.min_storage_blocks = 100
```

## Economic Examples

### Example 1: Market Adaptation

Network responds to increased demand:

```
Block 1000: Storage utilization = 85%
  → Block leader increases storage price by 0.1%
  → Price: 0.001000 → 0.001001

Block 1001: Storage utilization = 87%
  → Block leader increases storage price by 0.1%
  → Price: 0.001001 → 0.001002

... (continues adjusting)

Block 1100: Storage utilization = 65% (decreased)
  → Block leader decreases storage price by 0.1%
  → Price: 0.001010 → 0.001009
```

### Example 2: Community-Funded Model

Popular model supported by community:

```
Initial: Owner pays for 100 blocks (0.158 DESSIN)

Block 50: Community member tops up 50 blocks (0.079 DESSIN)
Block 75: Another member tops up 25 blocks (0.040 DESSIN)
Block 90: Owner extends 100 blocks (0.158 DESSIN)
Block 120: New contributor tops up 200 blocks (0.316 DESSIN)

Total contributors: 4
Total storage paid: 0.751 DESSIN
Total blocks: 475
```

### Example 3: Training Device Economics

Different nodes optimize costs:

```
Node A (No GPU):
  - Uses CPU for all training
  - Lower hardware costs
  - Slower training (longer block times)

Node B (GPU Available):
  - Uses GPU for training
  - Higher hardware costs
  - Faster training (shorter block times, more blocks mined)

Economic balance:
  - GPU nodes mine more blocks (higher earnings)
  - CPU nodes have lower operational costs
  - Network supports both, providing accessibility
```

## Security Considerations

### Parameter Adjustment Limits

- Maximum ±0.1% per block prevents sudden shocks
- Bounds prevent extreme values
- Gradual adjustments allow market adaptation
- History tracking provides transparency

### Storage Top-Up Security

- Payment validation prevents free riding
- Owner identity verification on model creation
- Transaction hashing for audit trail
- Expiration enforcement prevents indefinite storage

### Owner Authorization

- Model parameters immutable except by owner
- Dataset selection controlled by owner
- Prevents malicious model modification
- Training configuration locked to owner

## Testing

All features are comprehensively tested:

```bash
# Run dynamic parameter tests
./bin/python -m pytest tests/test_dynamic_parameters.py -v
# 11/11 passed ✅

# Run storage top-up tests
./bin/python -m pytest tests/test_storage_topup.py -v
# 10/10 passed ✅

# Run nanochat integration tests
./bin/python -m pytest tests/test_nanochat_integration.py -v
# 14/14 passed ✅

# Run demo
./bin/python e2e/scripts/dynamic_economics_demo.py
```

## Future Enhancements

- [ ] Governance voting on parameter bounds
- [ ] Automated parameter adjustment algorithms
- [ ] Storage insurance pools
- [ ] Model donation/transfer mechanisms
- [ ] Training compute credits
- [ ] Storage marketplace
- [ ] Parameter prediction markets

## References

- **Tezos**: Self-amendment and on-chain governance
- **Polkadot**: Adaptive parameters via governance
- **Cosmos**: Dynamic fee markets
- **Ethereum EIP-1559**: Base fee adjustments
- **Bitcoin**: Difficulty adjustment algorithm

## Migration Guide

### For Existing Deployments

1. **Parameter Initialization**: Default values maintain backward compatibility
2. **Storage Top-Up**: Existing models can be topped up by anyone
3. **Owner Rights**: Unchanged - owners retain full control
4. **Minimum Blocks**: Enforced on new models only

### Code Updates

```python
# Old (still works)
model_id = node.create_nanochat_model(
    name="My Model",
    depth=20
)

# New (with additional features)
model_id = node.create_nanochat_model(
    name="My Model",
    depth=20,
    training_device=TrainingDevice.GPU,  # Optional
    storage_blocks=200  # Optional (minimum enforced)
)

# New: Anyone can top-up
node.topup_model_storage(model_id, 100, 1.0)

# New: Block leaders adjust parameters
node.adjust_network_parameter(
    ParameterType.STORAGE_PRICE,
    increase=True
)
```

## Conclusion

The dynamic economics system provides:

✅ **Market responsiveness** through gradual parameter adjustments  
✅ **Community support** via open storage top-ups  
✅ **Owner control** over model parameters  
✅ **Resource optimization** with CPU/GPU configuration  
✅ **Economic sustainability** through minimum requirements  
✅ **Transparency** with complete tracking and history  

This creates a flexible, adaptive network that can respond to changing conditions while maintaining security and decentralization.
