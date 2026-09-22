# Nanochat Integration for DeSSIN

## Prerequisites

Inference and training paths that call into Karpathy's repo expect **`external/nanochat`** at the workspace root. From the repo root you can run **`bash scripts/setup_nanochat.sh`** to clone and pin the revision. Pin metadata is documented inline in `dessin/nanochat_wrapper.py`. If that tree is absent, wrappers fall back to **simulation** mode; CI and lightweight nodes may run without GPU nanochat binaries.

## Overview

DeSSIN now supports [nanochat](https://github.com/karpathy/nanochat) models - Andrej Karpathy's minimal, hackable ChatGPT implementation. This integration enables decentralized training, storage, and inference of nanochat models on the DeSSIN blockchain.

## Key Features

### 1. Nanochat Model Support
- **Multiple Depths**: Support for d20, d26, d32, and custom depth models
- **Configurable Training**: Customizable batch size, learning rate, dataset selection
- **Parameter Tracking**: Automatic parameter count estimation based on model architecture

### 2. Storage Rental System
- **Upfront Payment**: Model owners pay for storage in advance for a specified number of blocks
- **Dynamic Pricing**: Nodes can adjust storage prices (0.1x to 10x base rate)
- **Storage Extension**: Owners can extend storage by paying for additional blocks
- **Auto-Expiration**: Models are automatically deleted when storage expires

### 3. Training Payments
- **Per-Iteration Pricing**: Miners are paid for each training iteration
- **Owner Funding**: Model owners fund training operations
- **Economic Tracking**: Complete tracking of training costs and storage fees

### 4. Dynamic Block Time Scaling
- **Model Size Aware**: Block times automatically scale based on model size using the logarithmic formula `1.0 + (log₁₀(size / 10MB))^1.8 × 2.5` — see [BLOCK_TIME_SCALING.md](BLOCK_TIME_SCALING.md) for the full table.
- **Adaptive Mode (Default)**: In production nodes, utilization-based adaptive timing overrides the size formula; see [ADAPTIVE_BLOCK_TIME.md](ADAPTIVE_BLOCK_TIME.md).
- **Representative scaling factors** (logarithmic):
  - 10 MB: 1.0× (baseline)
  - 100 MB: ~3.5×
  - 1 GB: ~9.8×
  - 10 GB: ~19.2×
  - 100 GB: ~31.4×

### 5. Web Interface
- **ChatGPT-style UI**: Beautiful web interface inspired by nanochat's ui.html
- **Model Selection**: Easy switching between different deployed models
- **Real-time Chat**: Interactive chat with your trained models
- **Model Management**: View model info, training status, and costs

## Quick Start

### Creating a Nanochat Model

```python
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig

# Initialize node
config = DessinConfig.default()
node = DessinNode(config)
node.start()

# Create a nanochat model
model_id = node.create_nanochat_model(
    name="My Chat Model",
    depth=20,  # d20 model (~220M parameters)
    device_batch_size=32,
    dataset_name="fineweb",
    storage_blocks=1000,  # Pay for 1000 blocks
    storage_payment=10.0,  # 10 DESSIN upfront
    training_payment_per_iteration=0.001  # 0.001 DESSIN per iteration
)

print(f"Model created: {model_id}")
```

### Starting the Web Interface

```python
# Start web interface on port 8000
url = node.start_web_interface(port=8000, host="0.0.0.0")
print(f"Web interface: {url}")

# Access at http://localhost:8000
```

### Training a Model

```python
# Start training
node.nanochat_integration.start_training(
    model_id=model_id,
    miner_address=node.address,
    block_index=1
)

# Perform training iterations
for i in range(100):
    success, metrics = node.nanochat_integration.train_iteration(model_id)
    print(f"Iteration {i}: Loss {metrics['loss_after']:.4f}")
```

### Extending Storage

```python
# Get model info
model_info = node.model_manager.get_model_info(model_id)

# Calculate extension cost
extension_cost = node.model_manager.get_storage_cost(
    model_size_gb=model_info.size_gb,
    storage_blocks=500  # Extend by 500 blocks
)

# Extend storage
node.model_manager.extend_storage(
    model_id=model_id,
    additional_blocks=500,
    payment=extension_cost
)
```

## Architecture

### Components

1. **NanochatIntegration** (`dessin/nanochat_integration.py`)
   - Main integration class
   - Handles model creation, training, and lifecycle

2. **NanochatWebServer** (`dessin/nanochat_web.py`)
   - HTTP server for web interface
   - REST API for model queries and management

3. **ModelManager** (updated)
   - Extended to support nanochat model metadata
   - Storage rental and expiration tracking

4. **EconomicSystem** (updated)
   - Training iteration payments
   - Storage rental and extension payments
   - Dynamic storage pricing

5. **DynamicBlockTimeManager** (updated)
   - Model size-aware block time adjustments
   - Scaling from 10M to 100G models

### Data Flow

```
User → Node.create_nanochat_model()
  ↓
NanochatIntegration.create_model()
  ↓
ModelManager.register_model()
  ↓
EconomicSystem.process_storage_rental()
  ↓
Blockchain Storage
```

## Model Configurations

### Small Model (d20)
- **Depth**: 20 layers
- **Parameters**: ~140M
- **Size**: ~1.5 GB
- **Use Case**: Fast inference, limited compute
- **Recommended Batch Size**: 32

### Medium Model (d26)
- **Depth**: 26 layers
- **Parameters**: ~240M
- **Size**: ~2.5 GB
- **Use Case**: Better quality, moderate compute
- **Recommended Batch Size**: 16

### Large Model (d32)
- **Depth**: 32 layers
- **Parameters**: ~370M
- **Size**: ~4 GB
- **Use Case**: Best quality, high compute
- **Recommended Batch Size**: 8

## Economic Model

### Storage Pricing
- **Base Rate**: 0.001 DESSIN per GB per block
- **Adjustable**: Nodes can set multiplier (0.1x to 10x)
- **Upfront Payment**: Required for initial storage
- **Extension**: Pay for additional blocks anytime

### Training Pricing
- **Per Iteration**: Model owner pays miner
- **Default Rate**: 0.001 DESSIN per iteration
- **Configurable**: Set per model
- **Tracked**: Complete cost tracking

### Cost Examples

**Small Model (1.5 GB) for 1000 blocks:**
- Storage: 1.5 × 1000 × 0.001 = 1.5 DESSIN
- Training (1000 iterations): 1000 × 0.001 = 1.0 DESSIN
- **Total**: 2.5 DESSIN

**Large Model (4 GB) for 1000 blocks:**
- Storage: 4.0 × 1000 × 0.001 = 4.0 DESSIN
- Training (1000 iterations): 1000 × 0.003 = 3.0 DESSIN
- **Total**: 7.0 DESSIN

## Running the Demo

```bash
# Activate environment
source bin/activate

# Run demo
python e2e/scripts/nanochat_demo.py
```

The demo demonstrates:
- Creating models of different sizes
- Training iterations
- Storage rental and extension
- Dynamic block time scaling
- Web interface deployment

## Testing

```bash
# Run nanochat integration tests
./bin/python -m pytest tests/test_nanochat_integration.py -v

# Run all tests
./bin/python -m pytest tests/ -v
```

Run counts change as the suite grows; confirm with `pytest tests/ -v`. Relevant modules include `tests/test_nanochat_integration.py`, `tests/test_nanochat_wrapper.py`, and consensus or transaction tests covering your change.

## Web Interface

The web interface provides a ChatGPT-like experience:

- **Model Selection**: Dropdown to choose between deployed models
- **Chat Interface**: Real-time chat with typing indicators
- **Model Info**: Display model details (depth, parameters, etc.)
- **Responsive Design**: Works on desktop and mobile
- **Beautiful UI**: Gradient background, smooth animations

### API Endpoints

- `GET /` - Web interface HTML
- `GET /api/models` - List all models
- `GET /api/model/{id}` - Get model information
- `POST /api/chat` - Send chat message
- `POST /api/create_model` - Create new model

## Future Enhancements

- [ ] Expand production hardening around real nanochat when `external/nanochat` is present (resource limits, deterministic builds)
- [ ] Broader inference paths using the nanochat engine in all deployment modes
- [ ] Fine-tuning support with custom datasets
- [ ] Model quantization (4-bit, 8-bit)
- [ ] Distributed training across nodes
- [ ] Model marketplace
- [ ] Identity customization (per nanochat guide)

## References

- [Nanochat Repository](https://github.com/karpathy/nanochat)
- [Documentation index](README.md)
- [PoGO Protocol Specification](POGO_PROTOCOL_SPECIFICATION.md)
