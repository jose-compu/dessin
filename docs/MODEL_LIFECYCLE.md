# Model Lifecycle Management

## Overview

Model owners have full control over their models' lifecycle on the DeSSIN blockchain. They can:

- **Pause** training temporarily
- **Resume** training when ready
- **Deprecate** old or useless models
- **Withdraw** remaining fee balance

## Lifecycle States

### Active
- Model is training normally
- Receives fair rotation slots
- Payment deducted per iteration

### Paused
- Training temporarily stopped
- No payment deductions
- Can be resumed by owner
- Balance preserved

### Deprecated
- Model marked as obsolete/useless
- Training permanently stopped
- Balance can be withdrawn
- Cannot be reactivated

## Operations

### Pause Model

Owner temporarily stops training:

```python
# Send PauseModelTransaction
pause_tx = PauseModelTransaction(
    sender=owner_address,
    model_id=model_id,
    pause=True,
    fee=0.01,
    timestamp=time.time()
)

# Process on blockchain
lifecycle_manager.process_pause_transaction(pause_tx)
```

**Result:**
```
⏸️  Paused model model_code_v1...
✓ Model model_code_v1... paused on blockchain
```

**Use Cases:**
- Temporary budget constraints
- Waiting for better training data
- Model performing poorly
- Testing other models first

### Resume Model

Owner restarts training:

```python
# Send ResumeModelTransaction (pause=False)
resume_tx = PauseModelTransaction(
    sender=owner_address,
    model_id=model_id,
    pause=False,  # Resume
    fee=0.01,
    timestamp=time.time()
)

# Process on blockchain
lifecycle_manager.process_pause_transaction(resume_tx)
```

**Result:**
```
▶️  Resumed model model_code_v1...
✓ Model model_code_v1... resumed on blockchain
```

### Deprecate Model

Owner marks model as obsolete:

```python
# Send DeprecateModelTransaction
deprecate_tx = DeprecateModelTransaction(
    sender=owner_address,
    model_id=model_id,
    reason="Model outdated, replaced by v2",
    fee=0.01,
    timestamp=time.time()
)

# Process on blockchain
lifecycle_manager.process_deprecate_transaction(deprecate_tx)
```

**Result:**
```
🗑️  Deprecated model model_legacy_v0...
  Reason: Model outdated, replaced by v2
  Remaining balance: 2.30 DESSIN (available for withdrawal)
✓ Model model_legacy_v0... deprecated on blockchain
```

**Use Cases:**
- Model replaced by newer version
- Model not performing well
- Model no longer needed
- Consolidating models

### Withdraw Balance

Owner retrieves remaining fees:

```python
# Send WithdrawModelBalanceTransaction
withdraw_tx = WithdrawModelBalanceTransaction(
    sender=owner_address,
    model_id=model_id,
    withdrawal_address=owner_address,  # Optional, defaults to sender
    fee=0.01,
    timestamp=time.time()
)

# Process on blockchain
lifecycle_manager.process_withdraw_transaction(withdraw_tx)
```

**Result:**
```
💰 Withdrawal successful
  Model: model_legacy_v0...
  Owner: owner_bob_0x456...
  Amount: 2.30 DESSIN
✓ Withdrew 2.30 DESSIN from model model_legacy_v0...
  To: owner_bob_0x456...
```

**Requirements:**
- Model must be paused or deprecated
- Only owner can withdraw
- Full remaining balance returned

## Example Workflow

### Scenario: Upgrading a Model

```python
# 1. Owner has model_v1 training
# Balance: 10.0 DESSIN, trained 50 iterations

# 2. Owner creates model_v2 (better architecture)
model_v2_id = create_model(...)

# 3. Owner deprecates model_v1
deprecate_model(model_v1_id, reason="Replaced by v2")

# 4. Owner withdraws remaining balance from v1
success, amount = withdraw_balance(model_v1_id)
# Returns: 7.0 DESSIN (10.0 - 3.0 spent on training)

# 5. Owner adds withdrawn funds to model_v2
add_model_payment(model_v2_id, amount=7.0)

# Result: Smooth transition, no funds wasted
```

## Demo

Run the lifecycle management demo:

```bash
./bin/python e2e/scripts/model_lifecycle_demo.py
```

**Demo Flow:**

1. **Create 3 models** - chat, code, legacy
2. **Train 20 iterations** - all models rotate fairly
3. **Pause code model** - owner decision
4. **Train 10 iterations** - only chat and legacy train
5. **Deprecate legacy** - model is obsolete
6. **Withdraw balance** - owner gets remaining 2.30 DESSIN
7. **Resume code model** - after adding more funds
8. **Final training** - chat and code models active

**Final Status:**
```
Model: model_chat_v1
  Status: ACTIVE
  Total iterations: 27
  Balance remaining: 7.30 DESSIN

Model: model_code_v1
  Status: ACTIVE (resumed)
  Total iterations: 6
  Balance remaining: 9.40 DESSIN

Model: model_legacy_v0
  Status: DEPRECATED
  Balance remaining: 0.00 DESSIN (withdrawn)
```

## Automatic Management

### Auto-Pause Unfunded Models

System automatically pauses models that run out of funds:

```python
paused_models = lifecycle_manager.auto_pause_unfunded_models()
# Returns: ['model_123', 'model_456']
```

**Result:**
```
⏸️  Auto-paused 2 unfunded models
```

### Auto-Deprecate Old Models

System can deprecate very old models:

```python
deprecated = lifecycle_manager.auto_deprecate_old_models(
    max_age_blocks=100000,
    current_block=150000
)
# Returns: ['model_ancient_001']
```

**Result:**
```
🗑️  Auto-deprecated 1 old models
```

## Economic Impact

### Payment Flow

```
Model Creation:
  Owner → Escrow: 10.0 DESSIN (initial payment)

Training (50 iterations @ 0.1 DESSIN):
  Escrow → Miners: 5.0 DESSIN (paid out)
  Remaining in Escrow: 5.0 DESSIN

Deprecation + Withdrawal:
  Escrow → Owner: 5.0 DESSIN (refund)
```

### Cost Savings

**Without Lifecycle Management:**
- Owner pays 10.0 DESSIN upfront
- Model trains 50 iterations (5.0 DESSIN used)
- Remaining 5.0 DESSIN locked forever
- **Loss: 5.0 DESSIN**

**With Lifecycle Management:**
- Owner pays 10.0 DESSIN upfront
- Model trains 50 iterations (5.0 DESSIN used)
- Owner withdraws 5.0 DESSIN
- **Loss: 0.0 DESSIN**

## Blockchain Transactions

### Transaction Types

| Transaction | Purpose | Fee | Owner Only |
|-------------|---------|-----|------------|
| `PauseModelTransaction` | Pause/resume training | 0.01 | Yes |
| `DeprecateModelTransaction` | Mark model obsolete | 0.01 | Yes |
| `WithdrawModelBalanceTransaction` | Retrieve funds | 0.01 | Yes |

### Transaction Validation

All lifecycle transactions verify:
1. **Ownership** - Only model owner can execute
2. **Model Exists** - Model must be registered
3. **Valid State** - Operation allowed in current state
4. **Signature** - Transaction properly signed

### Failed Transaction Examples

```python
# Non-owner tries to deprecate
deprecate_model(model_id, sender=attacker)
# Result: ❌ Not owner: attacker cannot deprecate model_id

# Try to withdraw from active model
withdraw_balance(active_model_id)
# Result: ⚠️  Model still active. Pause or deprecate before withdrawing.

# Try to resume deprecated model
resume_model(deprecated_model_id)
# Result: ❌ Cannot resume deprecated model
```

## Best Practices

### For Model Owners

1. **Monitor Performance** - Pause models that aren't improving
2. **Deprecate Promptly** - Don't waste funds on bad models
3. **Withdraw Regularly** - Retrieve unused funds
4. **Plan Transitions** - Pause old model before creating new one

### For Network Operators

1. **Auto-Pause Unfunded** - Prevent training without payment
2. **Auto-Deprecate Old** - Clean up ancient models
3. **Monitor Balances** - Alert owners of low balances
4. **Fair Rotation** - Exclude paused/deprecated from queue

## Status Tracking

### Check Model Status

```python
stats = scheduler.get_model_stats(model_id)

print(f"Status: {stats['is_active']}")
print(f"Deprecated: {stats['is_deprecated']}")
print(f"Balance: {stats['payment_balance']}")
```

### Rotation Summary

```
================================================================================
Training Rotation Summary
================================================================================
Total models: 3
Active models: 2
Paid models: 2

Model Status:
--------------------------------------------------------------------------------
✓ model_chat_v1        | Iters:   27 | Balance:   7.30 | Paid:   2.70
🗑 model_legacy_v0      | Iters:    7 | Balance:   0.00 | Paid:   0.70 [DEPRECATED]
⏸ model_code_v1        | Iters:    6 | Balance:   4.40 | Paid:   0.60
================================================================================
```

**Status Icons:**
- ✓ = Active and paid
- ⏸ = Paused
- 🗑 = Deprecated

## Benefits

### For Model Owners

1. **Cost Control** - Stop paying for unused models
2. **Flexibility** - Pause and resume as needed
3. **Fund Recovery** - Withdraw unused balance
4. **Clean Management** - Mark obsolete models

### For Network

1. **Efficiency** - Don't train deprecated models
2. **Fairness** - More slots for active models
3. **Economics** - Proper fund management
4. **Scalability** - Remove old models

### For Miners

1. **Clear Queue** - Only train active models
2. **Guaranteed Payment** - No unfunded models
3. **Higher Utilization** - Focus on paying customers
4. **Simple Logic** - Scheduler handles everything

## Future Enhancements

- **Scheduled Deprecation** - Auto-deprecate after date
- **Conditional Pause** - Pause if loss > threshold
- **Partial Withdrawal** - Withdraw some, keep some
- **Transfer Ownership** - Sell model to another owner
- **Model Archival** - Store deprecated models off-chain
