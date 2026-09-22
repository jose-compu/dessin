# Multi-Model Training Rotation

## Overview

The DeSSIN blockchain supports **simultaneous training of multiple models** with intelligent scheduling and fair rotation based on payment status.

## Key Features

### Fair Rotation

- All models that have paid fees get equal training opportunities
- Round-robin scheduling across active, paid models
- No model is starved of training time

### Payment-Based Scheduling

- Models must maintain payment balance to stay active
- Payment deducted per training iteration
- Models automatically paused when payment exhausted
- Reactivated when owner adds more funds

### Priority Tiers

Models can have different priority levels:

1. **HIGH** - Premium models get 3× rotation slots
2. **NORMAL** - Standard models get 2× rotation slots
3. **LOW** - Economy models get 1× rotation slot
4. **PAUSED** - No training (payment expired)

## Architecture

### Model Training Scheduler

The `ModelTrainingScheduler` manages rotation across all models:

```python
from dessin.model_training_scheduler import ModelTrainingScheduler, ModelPriority

# Create scheduler
scheduler = ModelTrainingScheduler()

# Register models
scheduler.register_model(
    model_id="model_123",
    owner="owner_address",
    payment_per_iteration=0.1,
    initial_payment=10.0,
    priority=ModelPriority.NORMAL
)

# Get next model to train
model_id = scheduler.get_next_model_for_training()

# Record training iteration
scheduler.record_training_iteration(model_id)
```

### Rotation Algorithm

1. **Queue Building**
   - Group models by priority (HIGH, NORMAL, LOW)
   - Create weighted rotation queue
   - High priority models appear more frequently

2. **Fair Selection**
   - Round-robin through queue
   - Skip models without sufficient payment
   - Rebalance queue periodically

3. **Payment Tracking**
   - Deduct payment per iteration
   - Track total payments
   - Pause when balance < threshold

## Example: 3 Models, 30 Iterations

```
Model A: chat-model          → 11 iterations
Model B: code-model          → 9 iterations  
Model C: reasoning-model     → 10 iterations

Total: 30 iterations
Fair: Each got ~10 iterations (±1 variation is normal)
```

### Iteration Log

```
Iter  1: chat-model
Iter  2: code-model
Iter  3: reasoning-model
Iter  4: chat-model
Iter  5: code-model
Iter  6: reasoning-model
...
```

## Payment Management

### Adding Funds

Owners can add payment to keep models training:

```python
scheduler.add_model_payment(
    model_id="model_123",
    amount=10.0  # Add 10 DESSIN
)
```

### Payment Exhaustion

When a model runs out of funds:

```
⚠️  Model model_123 insufficient funds
⏸️  Model automatically paused
```

After adding funds:

```
✓ Payment added to model_123
✓ New balance: 10.00 DESSIN
✓ Model reactivated for training
```

## Testing

The e2e test demonstrates multi-model rotation:

```bash
./bin/python e2e/scripts/test_e2e_gpt_model.py
```

**Test Results:**

```
✅ SUCCESS: All tests passed!

Summary:
  ✓ Created 3 models with proper GPT architecture
  ✓ Models have ~70M parameters
  ✓ 30 training iterations completed successfully
  ✓ Fair rotation: Each model got 10 iterations
  ✓ Loss reduced for all models

The network intelligently rotates training between multiple paying models!
```

## Scheduler Status

Get real-time status of the training schedule:

```python
status = scheduler.get_schedule_status()

# Returns:
{
    "total_models": 3,
    "active_models": 3,
    "paid_models": 3,
    "rotation_queue_size": 6,
    "current_round": 2,
    "fair_rotation": True
}
```

## Model Statistics

Track statistics for each model:

```python
stats = scheduler.get_model_stats("model_123")

# Returns:
{
    "model_id": "model_123",
    "owner": "owner_address",
    "total_iterations": 10,
    "iterations_this_round": 3,
    "payment_balance": 8.70,
    "total_payments": 1.30,
    "payment_per_iteration": 0.1,
    "priority": "normal",
    "is_active": True,
    "is_paid_up": True
}
```

## Rotation Summary

View current rotation state:

```
================================================================================
Training Rotation Summary
================================================================================
Total models: 3
Active models: 3
Paid models: 3
Current round: 2

Model Status:
--------------------------------------------------------------------------------
✓ model_chat_001        | Iters:   11 | Balance:   8.90 | Paid:   1.10
✓ model_code_002        | Iters:   10 | Balance:   9.00 | Paid:   1.00
✓ model_reasoning_003   | Iters:    9 | Balance:   9.10 | Paid:   0.90
================================================================================
```

## Priority-Based Rotation

### Example with Mixed Priorities

```python
# Register models with different priorities
scheduler.register_model("premium_model", owner, 0.2, 20.0, ModelPriority.HIGH)
scheduler.register_model("standard_model", owner, 0.1, 10.0, ModelPriority.NORMAL)
scheduler.register_model("economy_model", owner, 0.05, 5.0, ModelPriority.LOW)
```

**Rotation Pattern (30 iterations):**

- Premium model: ~15 iterations (50%)
- Standard model: ~10 iterations (33%)
- Economy model: ~5 iterations (17%)

## Economics

### Payment per Iteration

- Owner sets `payment_per_iteration` when registering model
- Typical range: 0.05 - 0.5 DESSIN per iteration
- Higher payment = Higher priority available
- Payment deducted after each successful iteration

### Initial Payment

- Owner provides initial payment balance
- Typical range: 5 - 50 DESSIN
- Lasts for 50-500 iterations (depending on payment/iter)
- Can be topped up at any time

### Cost Calculation

```
Total Cost = iterations × payment_per_iteration

Example:
- 100 iterations
- 0.1 DESSIN per iteration
- Total: 10 DESSIN
```

## Blockchain Integration

### Model Registration

When a model is created on the blockchain:

```python
model_id = node.nanochat_integration.create_model(
    name="my-model",
    owner_address=owner,
    training_payment_per_iteration=0.1,
    ...
)

# Automatically registered with scheduler
scheduler.register_model(
    model_id=model_id,
    owner=owner,
    payment_per_iteration=0.1,
    initial_payment=initial_storage_payment
)
```

### Training Loop

Mining nodes use the scheduler:

```python
while True:
    # Get next model to train
    model_id = scheduler.get_next_model_for_training()
    
    if not model_id:
        # No models ready (all paused or queue empty)
        continue
    
    # Train one iteration
    success, metrics = train_model_iteration(model_id)
    
    # Record with scheduler
    if success:
        scheduler.record_training_iteration(model_id)
```

## Benefits

### For Model Owners

1. **Fair Treatment** - No model monopolizes training time
2. **Predictable Cost** - Pay per iteration pricing
3. **Control** - Set priority and payment amount
4. **Transparency** - View training statistics anytime

### For Miners

1. **Simple** - Scheduler handles all complexity
2. **Revenue** - Earn from multiple models
3. **Efficient** - No idle time (always a model to train)
4. **Fair** - All miners use same rotation

### For Network

1. **Scalability** - Handle many models simultaneously
2. **Efficiency** - Maximize GPU/CPU utilization
3. **Economic** - Payment-based access control
4. **Democratic** - Fair rotation for all paying users

## Use Cases

### Research Lab

- Train 10 different experiments simultaneously
- Each experiment gets 10% of resources
- Pay only for iterations used

### Production Service

- Train chat, code, and reasoning models
- Set chat model to HIGH priority (customer-facing)
- Fair rotation ensures all models improve

### Community Network

- 100+ models from different owners
- All paying users get equal treatment
- No single model dominates

## Future Enhancements

- Dynamic priority adjustment based on demand
- Auction-based priority bidding
- Time-based scheduling (train at specific times)
- Resource-aware scheduling (GPU vs CPU models)
- Multi-node coordination for large models
