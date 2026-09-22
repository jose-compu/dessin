# Multi-Layer Fine-Tuning Pipeline

## Overview

The DeSSIN blockchain supports **multi-layer fine-tuning pipelines** that allow model owners to define multiple sequential fine-tuning stages with automatic or manual progression between layers.

## Features

### Multiple Fine-Tuning Layers

- Define arbitrary number of fine-tuning layers
- Each layer has its own dataset and training parameters
- Sequential execution from layer 0 → layer N

### Automatic Progression

Model owners can set automatic conditions for progressing to the next layer:

1. **`iteration_count`** - Progress after N training iterations
2. **`loss_threshold`** - Progress when loss drops below threshold
3. **`accuracy_threshold`** - Progress when accuracy exceeds threshold
4. **`time_elapsed`** - Progress after N minutes of training
5. **`manual_only`** - No automatic progression (manual only)

### Manual Progression

Model owners can send blockchain transactions to manually progress to the next layer:

```python
# Send LayerProgressionTransaction
transaction = LayerProgressionTransaction(
    sender=owner_address,
    pipeline_id=pipeline_id,
    current_layer_index=1,
    force_progression=False  # or True to force
)
```

## Architecture

### Pipeline Structure

```
Pipeline
├── Layer 0: General Adaptation
│   ├── Dataset: general_domain
│   ├── Iterations: 50
│   └── Condition: iteration_count >= 50 → AUTO-PROGRESS
├── Layer 1: Domain Specialization
│   ├── Dataset: domain_specific
│   ├── Iterations: 100
│   └── Condition: loss < 7.0 → AUTO-PROGRESS
└── Layer 2: Task-Specific
    ├── Dataset: task_data
    ├── Iterations: 200
    └── Condition: manual_only → WAIT FOR TRANSACTION
```

### Layer Configuration

Each layer has:

```python
{
    'name': 'Layer Name',
    'description': 'What this layer does',
    'dataset_id': 'dataset_xyz',
    'learning_rate': 3e-4,
    'num_iterations': 100,
    'batch_size': 4,
    'progression_condition': 'loss_threshold',
    'condition_value': 7.0
}
```

## Usage

### Creating a Pipeline

```python
from dessin.fine_tuning.fine_tuning_pipeline import (
    FineTuningPipelineManager,
    ProgressionCondition
)

pipeline_manager = FineTuningPipelineManager(
    model_manager,
    fine_tuning_manager
)

# Define layers
layer_configs = [
    {
        'name': 'Layer 1: Broad Adaptation',
        'dataset_id': 'dataset_001',
        'learning_rate': 3e-4,
        'num_iterations': 50,
        'batch_size': 4,
        'progression_condition': 'iteration_count',
        'condition_value': 50  # Auto-progress after 50 iterations
    },
    {
        'name': 'Layer 2: Domain Focus',
        'dataset_id': 'dataset_002',
        'learning_rate': 1e-4,
        'num_iterations': 100,
        'batch_size': 4,
        'progression_condition': 'loss_threshold',
        'condition_value': 7.0  # Auto-progress when loss < 7.0
    },
    {
        'name': 'Layer 3: Final Polish',
        'dataset_id': 'dataset_003',
        'learning_rate': 5e-5,
        'num_iterations': 200,
        'batch_size': 4,
        'progression_condition': 'manual_only',
        'condition_value': 0  # Owner controls progression
    }
]

# Create pipeline
pipeline_id = pipeline_manager.create_pipeline(
    name="Production Pipeline",
    base_model_id=base_model_id,
    owner=owner_address,
    layer_configs=layer_configs,
    max_cost_dessin=200.0
)
```

### Starting Training

```python
# Start the pipeline
pipeline_manager.start_pipeline(pipeline_id)

# Training loop
while True:
    status = pipeline_manager.get_pipeline_status(pipeline_id)
    
    if status['status'] == 'completed':
        break
    
    # Train one iteration
    success, metrics = pipeline_manager.train_iteration(pipeline_id)
    
    print(f"Layer {metrics['layer_index']}: {metrics['layer_name']}")
    print(f"Loss: {metrics['loss_after']:.4f}")
    
    # Check for auto-progression
    if 'progressed_to_layer' in metrics:
        print(f"Auto-progressed to layer {metrics['progressed_to_layer']}")
```

### Manual Progression

```python
# Model owner decides to progress manually
pipeline_manager.manual_progress_layer(
    pipeline_id=pipeline_id,
    force=False  # Check conditions first
)

# Or force progression regardless of conditions
pipeline_manager.manual_progress_layer(
    pipeline_id=pipeline_id,
    force=True  # Skip condition check
)
```

## Blockchain Transactions

### Create Pipeline Transaction

```python
CreatePipelineTransaction(
    sender=owner_address,
    pipeline_name="My Pipeline",
    base_model_id=base_model_id,
    layer_configs=[...],
    max_cost_dessin=100.0,
    fee=0.01,
    timestamp=time.time()
)
```

### Layer Progression Transaction

```python
LayerProgressionTransaction(
    sender=owner_address,
    pipeline_id=pipeline_id,
    current_layer_index=1,
    force_progression=False,
    fee=0.01,
    timestamp=time.time()
)
```

## Example Demo

Run the demo to see automatic and manual progression:

```bash
./bin/python e2e/scripts/fine_tuning_pipeline_demo.py
```

This demonstrates:
- ✓ Layer 1: Auto-progresses after 50 iterations
- ✓ Layer 2: Auto-progresses when loss < 7.0
- ✓ Layer 3: Manually progressed via transaction

## Progression Conditions

### Iteration Count

```python
{
    'progression_condition': 'iteration_count',
    'condition_value': 50  # Progress after 50 iterations
}
```

Progress automatically when the layer completes N training iterations.

### Loss Threshold

```python
{
    'progression_condition': 'loss_threshold',
    'condition_value': 7.0  # Progress when loss < 7.0
}
```

Progress automatically when the best loss drops below the threshold.

### Accuracy Threshold

```python
{
    'progression_condition': 'accuracy_threshold',
    'condition_value': 0.95  # Progress when accuracy > 95%
}
```

Progress automatically when accuracy exceeds the threshold.

### Time Elapsed

```python
{
    'progression_condition': 'time_elapsed',
    'condition_value': 60.0  # Progress after 60 minutes
}
```

Progress automatically after N minutes of training the layer.

### Manual Only

```python
{
    'progression_condition': 'manual_only',
    'condition_value': 0  # No auto-progression
}
```

No automatic progression. Owner must send a `LayerProgressionTransaction` to progress.

## Benefits

### For Model Owners

1. **Gradual Fine-Tuning** - Progressively refine models through stages
2. **Cost Control** - Set budget limits per layer
3. **Flexible Progression** - Automatic or manual control
4. **Quality Gates** - Only progress when conditions are met

### For the Network

1. **Predictable Training** - Clear stages and progression rules
2. **Verification** - Each layer checkpoint is verified
3. **Economic Model** - Payments per layer iteration
4. **Decentralized** - Multiple miners can contribute

## Pipeline Status

Check pipeline status at any time:

```python
status = pipeline_manager.get_pipeline_status(pipeline_id)

print(f"Pipeline: {status['name']}")
print(f"Status: {status['status']}")
print(f"Current Layer: {status['current_layer_index']}/{status['total_layers']}")
print(f"Total Cost: {status['total_cost']:.2f} DESSIN")

for layer in status['layers']:
    print(f"  Layer {layer['index']}: {layer['name']}")
    print(f"    Status: {layer['status']}")
    print(f"    Progress: {layer['iterations_completed']}/{layer['max_iterations']}")
    print(f"    Best Loss: {layer['best_loss']:.4f}")
    print(f"    Condition Met: {layer['condition_met']}")
```

## Use Cases

### Staged Domain Adaptation

```
Layer 0: General language → Broad domain adaptation
Layer 1: Domain-specific → Specialized vocabulary/concepts
Layer 2: Task-specific → Final task optimization
```

### Curriculum Learning

```
Layer 0: Easy examples → Build foundation
Layer 1: Medium difficulty → Expand capabilities  
Layer 2: Hard examples → Master edge cases
```

### Progressive Specialization

```
Layer 0: General chat → Conversational ability
Layer 1: Technical docs → Domain knowledge
Layer 2: API calls → Tool use and execution
```

## Economics

- Each iteration costs DESSIN (paid by model owner)
- Pipeline has max_cost_dessin budget
- Training stops if budget exceeded
- Progression transactions have small fee
- Model owner controls all parameters

## Integration with PoGO

- Each layer checkpoint is verified via PoGO
- Merkle roots computed for layer transitions
- Block rewards for miners training layers
- Economic system handles all payments
- Consensus validates progression conditions
