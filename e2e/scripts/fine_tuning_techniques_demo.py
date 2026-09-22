#!/usr/bin/env python3
"""
Fine-Tuning Techniques Demo
============================

Demonstrates the modern fine-tuning techniques available in DeSSIN,
allowing model owners to configure and experiment with different methods.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.nanochat.nanochat_integration import TrainingDevice
from dessin.fine_tuning.fine_tuning_techniques import (
    FineTuningConfigurationManager,
    FineTuningMethod,
    OptimizerType,
    LRSchedulerType,
    TrainingConfig,
    LoRAConfig,
    QuantizationType
)
from dessin.fine_tuning.fine_tuning_transactions import (
    CreateFineTuningConfigTransaction,
    UpdateFineTuningConfigTransaction,
    SetModelFineTuningPresetTransaction,
    FineTuningConfigTransactionProcessor
)
import time


def main():
    print("=" * 80)
    print("Fine-Tuning Techniques Demo")
    print("=" * 80)
    print()
    print("This demo shows how model owners can configure fine-tuning techniques")
    print("using modern methods like LoRA, QLoRA, Adapters, and more.")
    print()
    
    # Initialize node
    print("Step 1: Initialize Node")
    print("-" * 80)
    config = DessinConfig.default()
    config.network.port = 9001
    config.network.bootstrap = True
    
    node = DessinNode(config)
    node.start()
    print(f"✓ Node started: {node.address[:20]}...")
    print()
    
    # Create user and model
    user_address = "model_owner_experimenter"
    if hasattr(node.consensus, 'economic_system'):
        node.consensus.economic_system.balances[user_address] = 1000.0
    
    print("Step 2: Create Base Model")
    print("-" * 80)
    model_id = node.nanochat_integration.create_model(
        name="experimental-model",
        owner_address=user_address,
        depth=10,
        device_batch_size=4,
        dataset_name="BaseData",
        storage_blocks=100,
        storage_payment=1.0,
        training_payment_per_iteration=0.1,
        training_device=TrainingDevice.CPU
    )
    
    model_info = node.model_manager.models.get(model_id)
    print(f"✓ Created model: {model_id[:30]}...")
    print(f"  Name: {model_info.name}")
    print(f"  Parameters: {model_info.parameters:,}")
    print()
    
    # Initialize fine-tuning configuration manager
    config_manager = FineTuningConfigurationManager()
    tx_processor = FineTuningConfigTransactionProcessor(config_manager, node.model_manager)
    
    # Demonstrate different fine-tuning techniques
    print("Step 3: Configure Different Fine-Tuning Techniques")
    print("-" * 80)
    print()
    
    # 1. LoRA Configuration
    print("1. LoRA (Low-Rank Adaptation)")
    print("   - Efficient: Only 0.1-2% parameters trained")
    print("   - Fast training and inference")
    print("   - Great for most use cases")
    print()
    
    lora_config_id = config_manager.create_configuration(
        model_id=model_id,
        owner=user_address,
        method=FineTuningMethod.LORA,
        lora_config=LoRAConfig(
            rank=16,
            alpha=32,
            dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        training_config=TrainingConfig(
            learning_rate=3e-4,
            optimizer=OptimizerType.ADAMW,
            lr_scheduler=LRSchedulerType.COSINE,
            num_epochs=3,
            batch_size=4
        ),
        description="Standard LoRA for efficient fine-tuning",
        experiment_name="lora_baseline"
    )
    
    lora_config = config_manager.get_configuration(lora_config_id)
    params_info = lora_config.get_trainable_params_estimate(model_info.parameters)
    print(f"   Trainable params: {params_info['trainable_params']:,} ({params_info['trainable_percentage']:.2f}%)")
    print()
    
    # 2. QLoRA Configuration
    print("2. QLoRA (Quantized LoRA)")
    print("   - Memory efficient: 4-bit quantization")
    print("   - Can train larger models on limited hardware")
    print("   - Minimal quality loss")
    print()
    
    qlora_config_id = config_manager.create_configuration(
        model_id=model_id,
        owner=user_address,
        method=FineTuningMethod.QLORA,
        lora_config=LoRAConfig(
            rank=32,
            alpha=64,
            dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            use_quantization=True,
            quantization_type=QuantizationType.INT4,
            use_double_quant=True
        ),
        training_config=TrainingConfig(
            learning_rate=2e-4,
            optimizer=OptimizerType.ADAMW,
            lr_scheduler=LRSchedulerType.COSINE,
            num_epochs=5,
            batch_size=8,
            bf16=True
        ),
        description="4-bit QLoRA for memory-efficient training",
        experiment_name="qlora_4bit"
    )
    
    print(f"   Using 4-bit quantization")
    print(f"   Can use larger batch sizes")
    print()
    
    # 3. Full Fine-Tuning
    print("3. Full Fine-Tuning")
    print("   - All parameters trained")
    print("   - Maximum quality")
    print("   - Requires more compute and memory")
    print()
    
    full_config_id = config_manager.create_configuration(
        model_id=model_id,
        owner=user_address,
        method=FineTuningMethod.FULL,
        training_config=TrainingConfig(
            learning_rate=5e-5,
            optimizer=OptimizerType.ADAMW,
            lr_scheduler=LRSchedulerType.COSINE_WITH_RESTARTS,
            num_epochs=3,
            batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            bf16=True
        ),
        description="Full parameter fine-tuning for maximum quality",
        experiment_name="full_finetuning"
    )
    
    full_config = config_manager.get_configuration(full_config_id)
    full_params_info = full_config.get_trainable_params_estimate(model_info.parameters)
    print(f"   Trainable params: {full_params_info['trainable_params']:,} (100%)")
    print()
    
    # 4. Using Presets
    print("4. Using Presets (Quick Setup)")
    print("   - Pre-configured settings for common use cases")
    print()
    
    print("   Available presets:")
    presets = [
        "lora_efficient",
        "lora_quality",
        "qlora_4bit",
        "full_finetuning",
        "adapter_lightweight",
        "prompt_tuning"
    ]
    
    for preset in presets:
        print(f"     • {preset}")
    print()
    
    # Use a preset via transaction
    preset_tx = SetModelFineTuningPresetTransaction(
        sender=user_address,
        model_id=model_id,
        preset_name="lora_quality",
        fee=0.01,
        timestamp=time.time(),
        public_key="pk_owner",
        signature="sig_preset",
        tx_id="tx_preset"
    )
    
    tx_processor.process_transaction(preset_tx)
    print()
    
    # Step 4: Demonstrate Configuration Updates
    print("Step 4: Update Configuration")
    print("-" * 80)
    print()
    print("Owner can update configurations to experiment with different settings")
    print()
    
    update_tx = UpdateFineTuningConfigTransaction(
        sender=user_address,
        config_id=lora_config_id,
        updates={
            'description': 'Updated LoRA with higher rank for better quality',
        },
        reason="Experimenting with higher rank",
        fee=0.01,
        timestamp=time.time(),
        public_key="pk_owner",
        signature="sig_update",
        tx_id="tx_update"
    )
    
    tx_processor.process_transaction(update_tx)
    print()
    
    # Step 5: Show All Configurations
    print("Step 5: All Configurations for Model")
    print("-" * 80)
    print()
    
    configs = config_manager.list_configurations(model_id=model_id)
    print(f"Total configurations: {len(configs)}")
    print()
    
    for config in configs:
        print(f"Config: {config.config_id}")
        print(f"  Method: {config.method.value}")
        print(f"  Description: {config.description}")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Learning rate: {config.training_config.learning_rate}")
        print(f"  Optimizer: {config.training_config.optimizer.value}")
        print(f"  LR Scheduler: {config.training_config.lr_scheduler.value}")
        print(f"  Version: {config.version}")
        
        params_info = config.get_trainable_params_estimate(model_info.parameters)
        print(f"  Trainable: {params_info['trainable_percentage']:.2f}% of parameters")
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("✓ Model owners can configure fine-tuning per model")
    print("✓ Multiple modern techniques supported:")
    print("   • LoRA (Low-Rank Adaptation)")
    print("   • QLoRA (Quantized LoRA)")
    print("   • Full Fine-Tuning")
    print("   • Adapter Layers")
    print("   • Prompt Tuning")
    print("   • Prefix Tuning")
    print("   • P-Tuning v2")
    print("   • IA3")
    print()
    print("✓ Configurable options:")
    print("   • Optimizers: AdamW, Adam, SGD, AdaFactor, Lion, Sophia")
    print("   • LR Schedulers: Constant, Linear, Cosine, Polynomial, etc.")
    print("   • Quantization: None, INT8, INT4, FP16, BF16")
    print("   • Gradient accumulation, checkpointing, mixed precision")
    print()
    print("✓ Blockchain transactions for configuration changes")
    print("✓ Preset configurations for quick setup")
    print("✓ Experimentation tracking with tags and versions")
    print()
    print("The community can experiment and find optimal configurations!")
    print()
    
    # Cleanup
    print("Stopping node...")
    node.stop()
    print("✓ Node stopped")


if __name__ == "__main__":
    main()
