#!/usr/bin/env python3
"""
Model Lifecycle Management Demo
================================

Demonstrates:
1. Training multiple models
2. Pausing a model
3. Deprecating an old/useless model
4. Withdrawing remaining balance
5. Resuming training
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.models.model_training_scheduler import ModelTrainingScheduler, ModelPriority
from dessin.models.model_lifecycle_transactions import (
    DeprecateModelTransaction,
    PauseModelTransaction,
    WithdrawModelBalanceTransaction,
    ModelLifecycleManager
)
import time


class MockModelManager:
    """Mock model manager for demo"""
    def __init__(self):
        self.models = {}
    
    def add_model(self, model_id, owner):
        self.models[model_id] = type('obj', (object,), {
            'model_id': model_id,
            'owner': owner,
            'upload_block': 0
        })()
    
    def get_model_info(self, model_id):
        return self.models.get(model_id)


class MockEconomicSystem:
    """Mock economic system for demo"""
    def __init__(self):
        self.balances = {}
    
    def get_balance(self, address):
        return self.balances.get(address, 0.0)


def main():
    print("=" * 80)
    print("Model Lifecycle Management Demo")
    print("=" * 80)
    print()
    
    # Setup
    scheduler = ModelTrainingScheduler()
    model_manager = MockModelManager()
    economic_system = MockEconomicSystem()
    lifecycle_manager = ModelLifecycleManager(model_manager, scheduler, economic_system)
    
    print()
    
    # Create 3 models
    print("Step 1: Creating Models")
    print("-" * 80)
    
    owner_alice = "owner_alice_0x123"
    owner_bob = "owner_bob_0x456"
    
    models = [
        ("model_chat_v1", owner_alice, 10.0),
        ("model_code_v1", owner_alice, 5.0),
        ("model_legacy_v0", owner_bob, 3.0)  # Old model, low budget
    ]
    
    economic_system.balances[owner_alice] = 100.0
    economic_system.balances[owner_bob] = 50.0
    
    for model_id, owner, payment in models:
        model_manager.add_model(model_id, owner)
        scheduler.register_model(
            model_id=model_id,
            owner=owner,
            payment_per_iteration=0.1,
            initial_payment=payment,
            priority=ModelPriority.NORMAL
        )
    
    print()
    scheduler.print_rotation_summary()
    
    # Train for a while
    print("Step 2: Training All Models")
    print("-" * 80)
    print("Training 20 iterations...")
    print()
    
    for i in range(20):
        model_id = scheduler.get_next_model_for_training()
        if model_id:
            scheduler.record_training_iteration(model_id)
            print(f"  Iter {i+1:2d}: {model_id}")
        else:
            print(f"  Iter {i+1:2d}: No models available")
    
    print()
    scheduler.print_rotation_summary()
    
    # Pause a model
    print("Step 3: Owner Alice Pauses code_v1 Model")
    print("-" * 80)
    print("Alice decides to pause model_code_v1 temporarily...")
    print()
    
    # Simulate pause transaction (in production would be signed and mined)
    print("Sending PauseModelTransaction to blockchain...")
    
    success = lifecycle_manager.process_pause_transaction(
        type('obj', (object,), {
            'sender': owner_alice,
            'model_id': 'model_code_v1',
            'pause': True
        })()
    )
    
    print()
    
    # Train more iterations (code model won't train)
    print("Training 10 more iterations (code model paused)...")
    print()
    
    for i in range(10):
        model_id = scheduler.get_next_model_for_training()
        if model_id:
            scheduler.record_training_iteration(model_id)
            print(f"  Iter {i+1:2d}: {model_id}")
    
    print()
    scheduler.print_rotation_summary()
    
    # Deprecate legacy model
    print("Step 4: Owner Bob Deprecates Legacy Model")
    print("-" * 80)
    print("Bob decides model_legacy_v0 is outdated and useless...")
    print()
    
    # Simulate deprecate transaction (in production would be signed and mined)
    print("Sending DeprecateModelTransaction to blockchain...")
    
    success = lifecycle_manager.process_deprecate_transaction(
        type('obj', (object,), {
            'sender': owner_bob,
            'model_id': 'model_legacy_v0',
            'reason': 'Model outdated, replaced by v2'
        })()
    )
    
    print()
    scheduler.print_rotation_summary()
    
    # Withdraw balance
    print("Step 5: Owner Bob Withdraws Remaining Balance")
    print("-" * 80)
    print(f"Bob's balance before: {economic_system.get_balance(owner_bob):.2f} DESSIN")
    print()
    
    # Simulate withdraw transaction (in production would be signed and mined)
    print("Sending WithdrawModelBalanceTransaction to blockchain...")
    
    success = lifecycle_manager.process_withdraw_transaction(
        type('obj', (object,), {
            'sender': owner_bob,
            'model_id': 'model_legacy_v0',
            'withdrawal_address': owner_bob
        })()
    )
    
    print()
    print(f"Bob's balance after: {economic_system.get_balance(owner_bob):.2f} DESSIN")
    print()
    
    # Resume paused model
    print("Step 6: Owner Alice Resumes code_v1 Model")
    print("-" * 80)
    print("Alice decides to resume model_code_v1 training...")
    print()
    
    # Add more funds first
    scheduler.add_model_payment("model_code_v1", 5.0)
    print()
    
    # Simulate resume transaction (in production would be signed and mined)
    print("Sending ResumeModelTransaction to blockchain...")
    
    success = lifecycle_manager.process_pause_transaction(
        type('obj', (object,), {
            'sender': owner_alice,
            'model_id': 'model_code_v1',
            'pause': False  # Resume
        })()
    )
    
    print()
    
    # Final training
    print("Step 7: Final Training Round")
    print("-" * 80)
    print("Training 10 iterations with resumed model...")
    print()
    
    for i in range(10):
        model_id = scheduler.get_next_model_for_training()
        if model_id:
            scheduler.record_training_iteration(model_id)
            print(f"  Iter {i+1:2d}: {model_id}")
    
    print()
    scheduler.print_rotation_summary()
    
    # Final summary
    print("=" * 80)
    print("Lifecycle Management Summary")
    print("=" * 80)
    print()
    
    all_stats = scheduler.get_all_model_stats()
    
    for stats in all_stats:
        print(f"Model: {stats['model_id']}")
        print(f"  Owner: {stats['owner'][:20]}...")
        print(f"  Status: ", end="")
        
        if stats['is_deprecated']:
            print(f"DEPRECATED - {stats['deprecation_reason']}")
        elif stats['is_active']:
            print("ACTIVE")
        else:
            print("PAUSED")
        
        print(f"  Total iterations: {stats['total_iterations']}")
        print(f"  Balance remaining: {stats['payment_balance']:.2f} DESSIN")
        print(f"  Total paid: {stats['total_payments']:.2f} DESSIN")
        print()
    
    print("=" * 80)
    print("Demonstrated Lifecycle Operations:")
    print("=" * 80)
    print("  ✓ Created multiple models with different owners")
    print("  ✓ Trained all models with fair rotation")
    print("  ✓ Paused a model (owner decision)")
    print("  ✓ Deprecated an old/useless model")
    print("  ✓ Withdrew remaining balance to owner")
    print("  ✓ Resumed training after adding funds")
    print()
    print("Model owners have full control over their models' lifecycle!")
    print()


if __name__ == "__main__":
    main()
