"""
Nanochat Integration Demo
=========================

This script demonstrates how to use nanochat models with the DeSSIN blockchain.

Features demonstrated:
- Creating nanochat models with different configurations
- Storage rental payment with dynamic pricing
- Training iteration payments
- Model storage expiration and renewal
- Web interface for chatting with models
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig


def main():
    print("=" * 70)
    print("DeSSIN Nanochat Integration Demo")
    print("=" * 70)
    print()
    
    # Create a DeSSIN node
    print("1. Initializing DeSSIN node...")
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 0.06  # ~3.6s for demo
    config.network.port = 8001
    
    node = DessinNode(config)
    print(f"   Node address: {node.address}")
    print(f"   Initial balance: {node.get_balance()} DESSIN")
    print()
    
    # Create nanochat models
    print("2. Creating nanochat models...")
    print()
    
    # Create a small model (d20)
    print("   Creating small model (d20)...")
    model_id_small = node.create_nanochat_model(
        name="Small Chat Model",
        depth=20,  # ~220M parameters
        device_batch_size=32,
        dataset_name="fineweb",
        storage_blocks=1000,
        storage_payment=10.0,
        training_payment_per_iteration=0.001
    )
    print(f"   ✓ Created: {model_id_small}")
    print()
    
    # Create a medium model (d26)
    print("   Creating medium model (d26)...")
    model_id_medium = node.create_nanochat_model(
        name="Medium Chat Model",
        depth=26,
        device_batch_size=16,  # Reduced batch size for larger model
        dataset_name="fineweb",
        storage_blocks=1000,
        storage_payment=15.0,
        training_payment_per_iteration=0.002
    )
    print(f"   ✓ Created: {model_id_medium}")
    print()
    
    # Create a large model (d32)
    print("   Creating large model (d32)...")
    model_id_large = node.create_nanochat_model(
        name="Large Chat Model",
        depth=32,
        device_batch_size=8,  # Further reduced for even larger model
        dataset_name="fineweb",
        storage_blocks=1000,
        storage_payment=20.0,
        training_payment_per_iteration=0.003
    )
    print(f"   ✓ Created: {model_id_large}")
    print()
    
    # Show model information
    print("3. Model Information:")
    print()
    
    for model_id in [model_id_small, model_id_medium, model_id_large]:
        info = node.nanochat_integration.get_model_info(model_id)
        print(f"   Model: {info['name']}")
        print(f"   - ID: {model_id}")
        print(f"   - Depth: d{info['depth']}")
        print(f"   - Parameters: {info['parameters']:,}")
        print(f"   - Storage blocks: {info['training']['storage_blocks']}")
        print(f"   - Storage payment: {info['training']['storage_payment']} DESSIN")
        print(f"   - Training payment per iteration: {info['training']['training_payment_per_iteration']} DESSIN")
        print()
    
    # Demonstrate training
    print("4. Training Models...")
    print()
    
    for model_id in [model_id_small]:
        info = node.nanochat_integration.get_model_info(model_id)
        print(f"   Training: {info['name']}")
        
        # Start training
        node.nanochat_integration.start_training(
            model_id=model_id,
            miner_address=node.address,
            block_index=1
        )
        
        # Perform a few training iterations
        for i in range(5):
            success, metrics = node.nanochat_integration.train_iteration(model_id)
            if success:
                print(f"     Iteration {metrics['iteration']}: "
                      f"Loss {metrics['loss_before']:.4f} → {metrics['loss_after']:.4f} "
                      f"(improvement: {metrics['loss_improvement']:.4f})")
        print()
    
    # Demonstrate storage rental and dynamic pricing
    print("5. Storage Rental & Dynamic Pricing:")
    print()
    
    # Show storage costs
    storage_cost_1gb = node.model_manager.get_storage_cost(
        model_size_gb=1.0,
        storage_blocks=1000
    )
    print(f"   Storage cost for 1 GB, 1000 blocks: {storage_cost_1gb} DESSIN")
    
    storage_cost_10gb = node.model_manager.get_storage_cost(
        model_size_gb=10.0,
        storage_blocks=1000
    )
    print(f"   Storage cost for 10 GB, 1000 blocks: {storage_cost_10gb} DESSIN")
    print()
    
    # Demonstrate storage extension
    print("   Extending storage for small model by 500 blocks...")
    model_info = node.model_manager.get_model_info(model_id_small)
    extension_payment = node.model_manager.get_storage_cost(
        model_size_gb=model_info.size_gb,
        storage_blocks=500
    )
    
    success = node.model_manager.extend_storage(
        model_id=model_id_small,
        additional_blocks=500,
        payment=extension_payment
    )
    print(f"   {'✓ Success' if success else '✗ Failed'}")
    print()
    
    # Demonstrate dynamic block time scaling
    print("6. Dynamic Block Time Scaling:")
    print()
    
    # Show how block time scales with model size
    sizes = [
        (10 * 1024 * 1024, "10 MB"),
        (100 * 1024 * 1024, "100 MB"),
        (1024 * 1024 * 1024, "1 GB"),
        (10 * 1024 * 1024 * 1024, "10 GB"),
        (100 * 1024 * 1024 * 1024, "100 GB")
    ]
    
    from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager
    btm = DynamicBlockTimeManager(config.consensus)
    
    print("   Model size scaling factors:")
    for size_bytes, size_str in sizes:
        factor = btm.calculate_model_size_factor(size_bytes)
        print(f"   - {size_str:>8}: {factor:.1f}x")
    print()
    
    # List all models
    print("7. All Models:")
    print()
    models = node.nanochat_integration.list_models()
    print(f"   Total models: {len(models)}")
    for model in models:
        print(f"   - {model['name']} (d{model['depth']}, {model['parameters'] / 1e6:.1f}M params)")
    print()
    
    # Start web interface
    print("8. Starting Web Interface...")
    print()
    url = node.start_web_interface(port=8000, host="0.0.0.0")
    if url:
        print(f"   ✓ Web interface started at: {url}")
        print(f"   Open your browser and visit: {url}")
        print()
        print("   Press Ctrl+C to stop the server...")
        print()
        
        try:
            # Keep the server running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()
            print("   Stopping web interface...")
            node.stop_web_interface()
    else:
        print("   ✗ Failed to start web interface")
    
    # Cleanup
    print()
    print("9. Cleanup:")
    node.stop()
    print("   ✓ Node stopped")
    print()
    
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
