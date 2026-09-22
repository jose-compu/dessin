"""
Dynamic Economics Demo
======================

Demonstrates the new dynamic economic features:
1. Block leader can adjust block reward by ±0.1% per block
2. Block leader can adjust storage rental price by ±0.1% per block
3. Anyone can top-up storage for any model (not just owner)
4. Only owner can manage model training parameters
5. Nodes can configure CPU/GPU for training
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.economics.dynamic_parameters import ParameterType
from dessin.nanochat.nanochat_integration import TrainingDevice


def main():
    print("=" * 80)
    print("DeSSIN Dynamic Economics Demo")
    print("=" * 80)
    print()
    
    # Create nodes
    print("1. Creating Nodes")
    print("-" * 80)
    
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 0.06
    config.network.port = 8001
    
    # Node 1: Block leader
    node1 = DessinNode(config)
    print(f"Node 1 (Block Leader): {node1.address[:20]}...")
    
    # Node 2: Model owner  
    config2 = DessinConfig.default()
    config2.network.port = 8002
    node2 = DessinNode(config2)
    print(f"Node 2 (Model Owner): {node2.address[:20]}...")
    
    # Node 3: Storage contributor
    config3 = DessinConfig.default()
    config3.network.port = 8003
    node3 = DessinNode(config3)
    print(f"Node 3 (Contributor): {node3.address[:20]}...")
    print()
    
    # Show initial parameters
    print("2. Initial Network Parameters")
    print("-" * 80)
    params = node1.dynamic_params.get_current_parameters()
    print(f"Block Reward: {params['block_reward']} DESSIN")
    print(f"Storage Price: {params['storage_price']} DESSIN/GB/block")
    print(f"Training Payment: {params['training_payment']} DESSIN/iteration")
    print()
    
    # Block leader adjusts parameters
    print("3. Block Leader Adjusts Network Parameters")
    print("-" * 80)
    
    # Increase block reward (attract more miners)
    print("Block Leader: Increasing block reward to attract miners...")
    node1.adjust_network_parameter(
        ParameterType.BLOCK_REWARD,
        increase=True,
        reason="Low mining rate, need to attract more miners"
    )
    print()
    
    # Decrease storage price (encourage more storage usage)
    print("Block Leader: Decreasing storage price to encourage usage...")
    node1.adjust_network_parameter(
        ParameterType.STORAGE_PRICE,
        increase=False,
        reason="Low storage utilization, reducing price"
    )
    print()
    
    # Show updated parameters
    params = node1.dynamic_params.get_current_parameters()
    print(f"Updated Block Reward: {params['block_reward']:.6f} DESSIN")
    print(f"Updated Storage Price: {params['storage_price']:.6f} DESSIN/GB/block")
    print()
    
    # Model owner creates model
    print("4. Model Owner Creates Nanochat Model")
    print("-" * 80)
    print("Note: Only the owner can set training parameters and dataset")
    print()
    
    model_id = node2.create_nanochat_model(
        name="Community Chat Model",
        depth=20,
        dataset_name="fineweb",  # Only owner can choose dataset
        device_batch_size=32,  # Only owner can configure
        storage_blocks=150,  # Paying for minimum required blocks
        storage_payment=2.0,
        training_device=TrainingDevice.GPU  # Owner chooses GPU training
    )
    print()
    
    # Show model info
    model_info = node2.model_manager.get_model_info(model_id)
    print(f"Model created: {model_id}")
    print(f"  Owner: {model_info.owner[:20]}...")
    print(f"  Size: {model_info.size_gb:.2f} GB")
    print(f"  Expires at block: {model_info.storage_expires}")
    print(f"  Training device: GPU")
    print()
    
    # Anyone can top-up storage
    print("5. Community Members Top-Up Storage (Anyone Can Do This!)")
    print("-" * 80)
    
    # Contributor 1 tops up
    print(f"Contributor {node3.address[:10]}... supporting the model:")
    success = node3.topup_model_storage(
        model_id=model_id,
        additional_blocks=50,
        payment=0.1
    )
    print()
    
    # Owner also tops up
    print(f"Owner {node2.address[:10]}... also topping up:")
    success = node2.topup_model_storage(
        model_id=model_id,
        additional_blocks=100,
        payment=0.2
    )
    print()
    
    # Show storage contributors
    print("Storage Contributors:")
    contributors = node2.storage_topup.get_contributors(model_id)
    for address, amount in contributors.items():
        is_owner = "(OWNER)" if address == node2.address else ""
        print(f"  {address[:20]}... : {amount:.4f} DESSIN {is_owner}")
    print()
    
    # Show storage statistics
    stats = node2.storage_topup.get_storage_statistics(model_id)
    print(f"Total top-ups: {stats['total_topups']}")
    print(f"Total amount: {stats['total_amount']:.4f} DESSIN")
    print(f"Total blocks added: {stats['total_blocks_added']}")
    print(f"Unique contributors: {stats['unique_contributors']}")
    print()
    
    # Demonstrate parameter adjustments over time
    print("6. Simulating Market Adaptation (Multiple Blocks)")
    print("-" * 80)
    print("Block leaders continuously adjust parameters based on network conditions:")
    print()
    
    # Simulate 20 blocks of adjustments
    for block in range(1, 21):
        if block % 3 == 0:
            # Every 3rd block, adjust storage price
            node1.dynamic_params.adjust_storage_price(
                block_index=block,
                block_leader=node1.address,
                increase=block % 6 == 0,  # Oscillate
                reason=f"Block {block} market adjustment"
            )
        
        if block % 5 == 0:
            # Every 5th block, adjust block reward
            node1.dynamic_params.adjust_block_reward(
                block_index=block,
                block_leader=node1.address,
                increase=block % 10 == 0,  # Oscillate
                reason=f"Block {block} mining incentive adjustment"
            )
    
    print()
    
    # Show parameter trends
    print("7. Parameter Trends Analysis")
    print("-" * 80)
    
    trend = node1.dynamic_params.get_parameter_trend(
        ParameterType.STORAGE_PRICE,
        blocks=20
    )
    print(f"Storage Price Trend (last {trend['blocks_analyzed']} adjustments):")
    print(f"  Increases: {trend['increases']}")
    print(f"  Decreases: {trend['decreases']}")
    print(f"  Net change: {trend['net_change_percent']:+.4f}%")
    print(f"  Start value: {trend['start_value']:.6f}")
    print(f"  End value: {trend['end_value']:.6f}")
    print()
    
    # Show current parameters
    print("8. Final Network Parameters")
    print("-" * 80)
    params = node1.dynamic_params.get_current_parameters()
    print(f"Block Reward: {params['block_reward']:.6f} DESSIN")
    print(f"Storage Price: {params['storage_price']:.6f} DESSIN/GB/block")
    print(f"Training Payment: {params['training_payment']:.6f} DESSIN/iteration")
    print()
    
    # Demonstrate owner-only operations
    print("9. Owner-Only Operations")
    print("-" * 80)
    print("✓ Only owner can upload model")
    print("✓ Only owner can choose training dataset")
    print("✓ Only owner can set training parameters")
    print("✓ Only owner can configure device (CPU/GPU)")
    print()
    print("✗ Contributors CANNOT change these settings")
    print("✓ Contributors CAN top-up storage for any model")
    print()
    
    # Show training device configuration
    print("10. Training Device Configuration")
    print("-" * 80)
    
    device_info = node2.nanochat_integration.get_training_device()
    has_gpu = node2.nanochat_integration.has_gpu
    
    print(f"Node GPU Available: {has_gpu}")
    print(f"Selected Device: {device_info}")
    print()
    print("Models can be configured to use:")
    print("  - CPU: Force CPU training")
    print("  - GPU: Force GPU training (if available)")
    print("  - AUTO: Automatically select best device")
    print()
    
    # Summary
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Block leaders can adjust rewards/prices by ±0.1% per block")
    print("  ✓ Parameters adapt to market conditions")
    print("  ✓ Anyone can top-up storage for models")
    print("  ✓ Only owners control model parameters")
    print("  ✓ Minimum storage blocks enforced")
    print("  ✓ CPU/GPU training configuration")
    print("  ✓ Complete economic tracking")
    print()
    
    # Cleanup
    node1.stop()
    node2.stop()
    node3.stop()


if __name__ == "__main__":
    main()
