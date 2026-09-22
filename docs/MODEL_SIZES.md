# DeSSIN GPT Model Sizes

Quick reference for model sizes in the DeSSIN blockchain network.

## Model Size Table

All models use the GPT-2 architecture with standard vocabulary (50,257 tokens) and context length (1024 tokens).

| Depth | Layers | Hidden Size | Parameters | Size Description | Use Case |
|-------|--------|-------------|------------|------------------|----------|
| d12   | 12     | 512         | 89.7M      | Small            | Fast testing, development |
| d16   | 16     | 640         | 143.6M     | Medium-Small     | Quick experiments |
| d20   | 20     | 768         | 219.6M     | Medium           | **Default for e2e tests** |
| d26   | 26     | 960         | 385.1M     | Large            | Production quality |
| d32   | 32     | 1152        | 626.7M     | Very Large       | High performance |

## Parameter Calculation

Parameters are calculated as follows for a GPT model:

```
hidden_size = 768 + (depth - 20) × 32

Parameters = 
  + Token embeddings: vocab_size × hidden_size
  + Position embeddings: context_length × hidden_size
  + Transformer blocks: depth × (4 × hidden_size² + 8 × hidden_size² + 2 × hidden_size)
  + Output layer: hidden_size × vocab_size
```

Where:
- **Attention**: 4 weight matrices (Q, K, V, Output) = 4 × hidden_size²
- **FFN**: 2 layers with 4× expansion = 2 × hidden_size × (4 × hidden_size) = 8 × hidden_size²
- **Layer Norms**: 2 per block = 2 × hidden_size

## Model Architecture Details

### Default Configuration (d20)

```python
depth = 20
hidden_size = 768
attention_heads = 12
vocab_size = 50257  # GPT-2 tokenizer
context_length = 1024
total_parameters = 219,569,664  # ~220M
```

**Architecture Breakdown:**
- Embeddings: 39.4M parameters
- Transformer blocks (20 layers): 141.6M parameters
- Output layer: 38.6M parameters

## Training Characteristics

| Depth | Params | Est. Training Time* | Memory (FP32) | Storage |
|-------|--------|---------------------|---------------|---------|
| d12   | 90M    | ~1-2 hours          | ~1.4 GB       | ~360 MB |
| d16   | 144M   | ~2-3 hours          | ~2.2 GB       | ~575 MB |
| d20   | 220M   | ~3-5 hours          | ~3.3 GB       | ~880 MB |
| d26   | 385M   | ~8-12 hours         | ~5.8 GB       | ~1.5 GB |
| d32   | 627M   | ~15-20 hours        | ~9.5 GB       | ~2.5 GB |

\* On a single H100 GPU with device_batch_size=16

## Dataset Options

The DeSSIN network supports various datasets for language model training:

1. **FineWeb** (default) - High-quality web text dataset
2. **OpenWebText** - Reddit-sourced web content
3. **WikiText** - Wikipedia articles
4. **Custom** - User-provided training data

## Verification

To verify integrations in CI, run the pytest suites (nanochat integration and consensus):

```bash
pytest tests/test_nanochat_integration.py tests/test_consensus.py -v
```

For a manual multi-node bring-up, use the scripts referenced in `README.md` (for example `e2e/scripts/run_4node_network.py`).

## Example Usage in Tests

### Multi-node demo (`e2e/scripts/run_4node_network.py`)

```python
model_id = node.nanochat_integration.create_model(
    name="test-nano-d20",
    depth=20,  # ~220M parameters
    device_batch_size=8,
    dataset_name="FineWeb",
    storage_blocks=100,
    training_payment_per_iteration=0.1
)
```

### Quick web UI demo (`e2e/scripts/start_web_ui.py`)

Use `./bin/python e2e/scripts/start_web_ui.py --quick-train` (or `e2e/scripts/run_network_with_web_ui.py`) for local training-plus-UI workflows; tune `depth`, `vocab_size`, and `context_length` to match sizing goals.

```python
model_id = node.nanochat_integration.create_model(
    name="tiny-chat-model",
    depth=12,  # ~90M parameters (faster for testing)
    device_batch_size=4,
    vocab_size=1024,  # Reduced for speed
    context_length=128
)
```

## Sound GPT Model Requirements

✅ All models in the network meet the following criteria:

1. **Parameter Count**: All models have 100K+ parameters (smallest is 90M)
2. **Architecture**: Standard GPT-2 style transformer
3. **Training**: Proper gradient descent on language modeling
4. **Dataset**: Real language data (FineWeb, WikiText, etc.)
5. **Use Case**: Text generation and language understanding

## Recommended Configurations

- **Development/Testing**: d12 or d16 (fast iteration)
- **Default E2E Tests**: d20 (good balance of size and speed)
- **Production**: d26 or d32 (better quality)
- **Research**: Custom configurations supported

## Notes

- All models use the nanochat implementation (Karpathy's GPT)
- Models can be fine-tuned on custom data after pre-training
- 4-bit quantization available for inference (reduces memory by ~75%)
- Distributed training supported across multiple nodes
