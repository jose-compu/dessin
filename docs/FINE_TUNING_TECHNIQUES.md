# Modern Fine-Tuning Techniques

## Overview

DeSSIN supports state-of-the-art fine-tuning techniques, allowing model owners to configure and experiment with different methods on a per-model basis. The community can explore various approaches to find optimal configurations for their specific use cases.

## Supported Techniques

### 1. LoRA (Low-Rank Adaptation)

**Best for**: Most use cases, efficient training

LoRA adds small trainable rank decomposition matrices to specific layers while freezing the base model weights.

**Advantages**:
- Only 0.1-2% of parameters need training
- Fast training and inference
- Easy to swap between different LoRA adapters
- Minimal memory overhead

**Configuration**:
```python
lora_config = LoRAConfig(
    rank=16,                    # LoRA rank (r) - higher = more capacity
    alpha=32,                   # Scaling factor (typically 2*rank)
    dropout=0.1,               # Dropout for regularization
    target_modules=[           # Which layers to apply LoRA
        "q_proj", "v_proj",
        "k_proj", "o_proj"
    ]
)
```

**Recommended ranks**:
- 8-16: Efficient, good for most tasks
- 32-64: Higher quality, more parameters
- 128+: Maximum quality, approaching full fine-tuning

### 2. QLoRA (Quantized LoRA)

**Best for**: Training large models on limited hardware

QLoRA combines LoRA with 4-bit quantization, enabling fine-tuning of large models on consumer GPUs.

**Advantages**:
- Massive memory savings (4x-8x reduction)
- Can train larger models on same hardware
- Minimal quality degradation
- Uses NormalFloat4 (NF4) quantization

**Configuration**:
```python
lora_config = LoRAConfig(
    rank=32,
    alpha=64,
    use_quantization=True,
    quantization_type=QuantizationType.INT4,
    use_double_quant=True      # Double quantization for extra savings
)

training_config = TrainingConfig(
    bf16=True,                 # Use BF16 for compute
    batch_size=8               # Can use larger batches
)
```

### 3. Full Fine-Tuning

**Best for**: Maximum quality, sufficient compute available

Fine-tunes all model parameters.

**Advantages**:
- Highest quality results
- Full model adaptation
- Best for domain-specific tasks

**Disadvantages**:
- Requires more memory and compute
- Slower training
- Need to store full model weights

**Configuration**:
```python
training_config = TrainingConfig(
    learning_rate=5e-5,        # Lower LR for stability
    batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,  # Save memory
    bf16=True
)
```

### 4. Adapter Layers

**Best for**: Lightweight fine-tuning, multi-task learning

Adds small bottleneck layers between transformer blocks.

**Advantages**:
- Only 0.5-3% parameters added
- Fast training
- Can train multiple adapters for different tasks

**Configuration**:
```python
adapter_config = AdapterConfig(
    adapter_size=64,           # Bottleneck dimension
    adapter_dropout=0.1,
    adapter_non_linearity="gelu",
    adapter_residual=True
)
```

### 5. Prompt Tuning

**Best for**: Task-specific adaptation, minimal parameters

Learns continuous prompt embeddings prepended to input.

**Advantages**:
- Extremely parameter efficient
- Very fast training
- Good for few-shot learning

**Configuration**:
```python
prompt_config = PromptTuningConfig(
    num_virtual_tokens=20,     # Number of learned tokens
    prompt_tuning_init="random"
)

training_config = TrainingConfig(
    learning_rate=3e-2,        # Higher LR for prompts
    batch_size=16
)
```

### 6. Prefix Tuning

**Best for**: Generation tasks, conditional generation

Similar to prompt tuning but learns prefix vectors for both keys and values.

**Configuration**:
```python
prompt_config = PromptTuningConfig(
    num_virtual_tokens=20,
    prefix_projection=True     # Use MLP for reparameterization
)
```

### 7. P-Tuning v2

**Best for**: Better than prompt tuning, across all layers

Adds learnable prompts to every layer's input.

### 8. IA3 (Infused Adapter)

**Best for**: Even more parameter efficient than LoRA

Learns scalar vectors to rescale activations.

## Optimizers

### AdamW (Recommended)
Default optimizer, works well for most cases.
```python
optimizer=OptimizerType.ADAMW
```

### AdaFactor
Memory-efficient alternative to Adam.
```python
optimizer=OptimizerType.ADAFACTOR
```

### Lion
Newer optimizer, often faster convergence.
```python
optimizer=OptimizerType.LION
```

### Sophia
Second-order optimizer, can be faster for large models.
```python
optimizer=OptimizerType.SOPHIA
```

## Learning Rate Schedulers

### Cosine Annealing (Recommended)
```python
lr_scheduler=LRSchedulerType.COSINE
warmup_steps=100
```

### Linear Decay
```python
lr_scheduler=LRSchedulerType.LINEAR
warmup_ratio=0.1
```

### Cosine with Restarts
```python
lr_scheduler=LRSchedulerType.COSINE_WITH_RESTARTS
```

### Reduce on Plateau
Adaptive scheduler based on metrics.
```python
lr_scheduler=LRSchedulerType.REDUCE_ON_PLATEAU
```

## Presets

Quick-start configurations for common scenarios:

### `lora_efficient`
Fast and efficient LoRA (rank 8)
```python
config_manager.create_preset_configuration(
    model_id=model_id,
    owner=owner_address,
    preset="lora_efficient"
)
```

### `lora_quality`
High-quality LoRA (rank 64)
```python
preset="lora_quality"
```

### `qlora_4bit`
Memory-efficient 4-bit QLoRA
```python
preset="qlora_4bit"
```

### `full_finetuning`
Full parameter fine-tuning
```python
preset="full_finetuning"
```

### `adapter_lightweight`
Lightweight adapter layers
```python
preset="adapter_lightweight"
```

### `prompt_tuning`
Prompt tuning with 20 virtual tokens
```python
preset="prompt_tuning"
```

## Usage Examples

### Creating a Configuration

```python
from dessin.fine_tuning import (
    FineTuningConfigurationManager,
    FineTuningMethod,
    LoRAConfig,
    TrainingConfig
)

config_manager = FineTuningConfigurationManager()

# Create LoRA configuration
config_id = config_manager.create_configuration(
    model_id="model_123",
    owner="owner_address",
    method=FineTuningMethod.LORA,
    lora_config=LoRAConfig(rank=16, alpha=32),
    training_config=TrainingConfig(
        learning_rate=3e-4,
        num_epochs=3,
        batch_size=4
    ),
    description="Baseline LoRA configuration",
    experiment_name="lora_r16_baseline"
)
```

### Using Blockchain Transactions

```python
from dessin.fine_tuning.fine_tuning_transactions import (
    CreateFineTuningConfigTransaction,
    SetModelFineTuningPresetTransaction
)

# Create configuration via transaction
tx = CreateFineTuningConfigTransaction(
    sender=owner_address,
    model_id=model_id,
    method="lora",
    config_data={
        "rank": 32,
        "alpha": 64,
        "learning_rate": 2e-4
    },
    description="Production LoRA config",
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)

# Or use a preset
preset_tx = SetModelFineTuningPresetTransaction(
    sender=owner_address,
    model_id=model_id,
    preset_name="qlora_4bit",
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)
```

### Updating Configuration

```python
from dessin.fine_tuning.fine_tuning_transactions import UpdateFineTuningConfigTransaction

update_tx = UpdateFineTuningConfigTransaction(
    sender=owner_address,
    config_id=config_id,
    updates={
        "description": "Increased rank for better quality",
    },
    reason="Experimentation with higher capacity",
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)
```

## Experimentation Workflow

1. **Choose Method**: Select based on compute/quality tradeoff
2. **Configure**: Set hyperparameters (rank, LR, etc.)
3. **Tag Experiments**: Use `experiment_name` and `tags` for tracking
4. **Monitor**: Track trainable parameters, loss, and metrics
5. **Iterate**: Update configuration based on results
6. **Share**: Successful configurations can be shared with community

## Best Practices

### For Limited Hardware
- Use QLoRA with 4-bit quantization
- Enable gradient checkpointing
- Use smaller batch sizes with gradient accumulation
- Consider adapter or prompt tuning

### For Maximum Quality
- Use full fine-tuning or high-rank LoRA (64+)
- Longer training (more epochs)
- Careful learning rate tuning
- Use cosine annealing scheduler

### For Fast Iteration
- Use LoRA with rank 8-16
- Smaller models (depth 10-12)
- Fewer epochs initially
- Quick validation

### For Multi-Task
- Train multiple LoRA adapters
- Share base model, swap adapters
- Use adapter layers for parallel tasks

## Community Experimentation

The configuration system is designed for community experimentation:

- **Versioning**: Each configuration update increments version
- **Tags**: Label experiments for easy filtering
- **History**: All configurations are tracked per model
- **Sharing**: Successful configs can be documented and reused
- **Transparency**: All configurations on blockchain

## Parameter Efficiency Comparison

| Method | Trainable Params | Memory | Training Speed | Quality |
|--------|-----------------|---------|----------------|---------|
| Full | 100% | High | Slow | Highest |
| LoRA (r=8) | ~0.1% | Low | Fast | High |
| LoRA (r=64) | ~1% | Low | Fast | Very High |
| QLoRA | ~0.5% | Very Low | Fast | High |
| Adapter | ~2% | Low | Fast | High |
| Prompt Tuning | <0.01% | Very Low | Very Fast | Medium |
| Prefix Tuning | <0.1% | Very Low | Very Fast | Medium-High |

## Advanced Features

### Gradient Accumulation
Train with effective larger batch sizes:
```python
batch_size=2
gradient_accumulation_steps=8  # Effective batch size = 16
```

### Mixed Precision
Faster training with FP16/BF16:
```python
bf16=True  # Recommended
# or
fp16=True  # Older GPUs
```

### Gradient Checkpointing
Trade compute for memory:
```python
gradient_checkpointing=True  # Saves ~30% memory
```

### Warmup
Stabilize early training:
```python
warmup_steps=100
# or
warmup_ratio=0.1  # 10% of total steps
```

## Demo Script

Run the demo to see all techniques:
```bash
./bin/python e2e/scripts/fine_tuning_techniques_demo.py
```

This demonstrates:
- Creating configurations for different methods
- Using presets
- Updating configurations
- Tracking experiments
- Comparing parameter efficiency

## Future Enhancements

Potential additions:
- ReLoRA (periodic LoRA reinitialization)
- DyLoRA (dynamic rank selection)
- AdaLoRA (adaptive rank allocation)
- LoRA+ (separate LR for A and B matrices)
- DoRA (weight decomposition LoRA)
- Flash Attention integration
- Deepspeed/FSDP support for distributed training

The configuration system is extensible to support these as they become standard practice.
