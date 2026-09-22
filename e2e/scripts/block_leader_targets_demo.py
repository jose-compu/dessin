"""
Block Leader Pricing Targets Demo
=================================

Demonstrates how node operators can set pricing targets that will be applied
when they are selected as block leader via VRF randomness.

Since operators can only influence network-wide prices when selected as leader,
they configure targets/strategies that automatically apply when they get their turn.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.networking.node_operator_api import PricingParameter, PricingStrategy


def main():
    print("=" * 80)
    print("Block Leader Pricing Targets Demo")
    print("=" * 80)
    print()
    
    print("Concept:")
    print("  Node operators can only influence network prices when selected as")
    print("  block leader via VRF randomness. They configure pricing targets that")
    print("  automatically apply when they become leader.")
    print()
    
    # Create node
    config = DessinConfig.default()
    config.network.port = 9001
    node = DessinNode(config)
    
    print()
    print("─" * 80)
    print("Scenario 1: Set Absolute Target Price")
    print("─" * 80)
    print()
    
    print("Operator wants storage price at 0.0015 DESSIN/GB/block")
    print("Strategy: Move towards target gradually when selected as leader")
    print()
    
    node.set_storage_price_target(target=0.0015)
    
    print()
    print("Target configured. Will apply when node selected as block leader.")
    print()
    
    # Simulate being selected as leader
    print("Simulating: Node selected as block leader...")
    node.on_selected_as_block_leader()
    
    print()
    print("─" * 80)
    print("Scenario 2: Increase Price by Percentage")
    print("─" * 80)
    print()
    
    print("Operator wants to increase block rewards by 50%")
    print("Strategy: Always increase when selected as leader")
    print()
    
    node.set_block_reward_target(increase_percent=0.5)
    
    print()
    print("Target configured. Will apply when node selected as block leader.")
    print()
    
    # Simulate being selected as leader
    print("Simulating: Node selected as block leader...")
    node.on_selected_as_block_leader()
    
    print()
    print("─" * 80)
    print("Scenario 3: Temporary Adjustment")
    print("─" * 80)
    print()
    
    print("Operator wants to decrease storage price by 20% for 100 blocks")
    print("Strategy: Promotional pricing for limited time")
    print()
    
    node.set_storage_price_target(decrease_percent=0.2, duration_blocks=100)
    
    print()
    print("Temporary target configured.")
    print()
    
    # Simulate being selected as leader multiple times
    print("Simulating: Node selected as leader 5 times over 100 blocks...")
    for i in range(5):
        print(f"\nBlock leader selection {i+1}/5:")
        node.on_selected_as_block_leader()
        
        # Show remaining blocks
        targets = node.get_leader_targets()
        for param_name, target_info in targets["targets"].items():
            if target_info.get("blocks_remaining"):
                print(f"  Blocks remaining: {target_info['blocks_remaining']}")
    
    print()
    print("─" * 80)
    print("Scenario 4: Multiple Targets")
    print("─" * 80)
    print()
    
    print("Operator configures multiple pricing targets:")
    print("  - Storage price: Target 0.002")
    print("  - Block reward: Increase by 30%")
    print()
    
    node.set_storage_price_target(target=0.002)
    node.set_block_reward_target(increase_percent=0.3)
    
    print()
    print("When selected as leader, all targets will be applied:")
    print()
    
    # Simulate being selected
    print("Simulating: Node selected as block leader...")
    node.on_selected_as_block_leader()
    
    print()
    print("─" * 80)
    print("Scenario 5: Remove Target")
    print("─" * 80)
    print()
    
    print("Operator decides to stop influencing storage price")
    print()
    
    node.set_storage_price_target()  # No arguments = remove target
    
    print()
    print("Storage price target removed. Block reward target still active.")
    print()
    
    # Show current targets
    targets = node.get_leader_targets()
    print("Active targets:")
    for param_name, target_info in targets["targets"].items():
        print(f"  - {param_name}: {target_info['strategy']}")
    
    print()
    print("─" * 80)
    print("Scenario 6: View Statistics")
    print("─" * 80)
    print()
    
    targets_info = node.get_leader_targets()
    stats = targets_info["statistics"]
    
    print("Block Leader Statistics:")
    print(f"  Times selected as leader: {stats['times_selected']}")
    print(f"  Adjustments made: {stats['adjustments_made']}")
    print(f"  Active targets: {stats['active_targets']}")
    print(f"  Expired targets: {stats['expired_targets']}")
    
    print()
    print("─" * 80)
    print("Real-World Example: Multi-Node Network")
    print("─" * 80)
    print()
    
    print("Network with 10 nodes, each with different strategies:")
    print()
    
    # Create multiple nodes with different strategies
    strategies = [
        ("Node 1", "Target storage at 0.0015", lambda n: n.set_storage_price_target(target=0.0015)),
        ("Node 2", "Increase rewards 20%", lambda n: n.set_block_reward_target(increase_percent=0.2)),
        ("Node 3", "Decrease storage 10%", lambda n: n.set_storage_price_target(decrease_percent=0.1)),
        ("Node 4", "Target rewards at 12.0", lambda n: n.set_block_reward_target(target=12.0)),
        ("Node 5", "Temporary +50% rewards", lambda n: n.set_block_reward_target(increase_percent=0.5, duration_blocks=50)),
    ]
    
    for node_name, strategy_desc, strategy_fn in strategies:
        print(f"{node_name}: {strategy_desc}")
    
    print()
    print("Network Dynamics:")
    print("  • Each node is selected as leader randomly (VRF)")
    print("  • When selected, node applies its pricing targets")
    print("  • Network prices evolve based on leader strategies")
    print("  • Market finds equilibrium through decentralized decision-making")
    print()
    
    print("─" * 80)
    print("Summary")
    print("─" * 80)
    print()
    
    print("Node Operator Strategies:")
    print()
    print("1. Absolute Target")
    print("   node.set_storage_price_target(target=0.0015)")
    print("   → Move towards specific price when leader")
    print()
    
    print("2. Relative Increase")
    print("   node.set_block_reward_target(increase_percent=0.5)")
    print("   → Always increase by 50% when leader")
    print()
    
    print("3. Relative Decrease")
    print("   node.set_storage_price_target(decrease_percent=0.2)")
    print("   → Always decrease by 20% when leader")
    print()
    
    print("4. Temporary Adjustment")
    print("   node.set_storage_price_target(")
    print("       decrease_percent=0.2,")
    print("       duration_blocks=100")
    print("   )")
    print("   → Temporary pricing for 100 blocks")
    print()
    
    print("5. No Target")
    print("   node.set_storage_price_target()")
    print("   → Let network determine price")
    print()
    
    print("Key Points:")
    print("  ✓ Targets only apply when selected as block leader")
    print("  ✓ Selection is random via VRF (can't predict)")
    print("  ✓ Each operator chooses their own strategy")
    print("  ✓ Network prices evolve through decentralized consensus")
    print("  ✓ Market-driven price discovery")
    print()
    
    # Cleanup
    node.stop()


if __name__ == "__main__":
    main()
