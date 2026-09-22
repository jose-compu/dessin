"""
Node Operator API Demo
======================

Demonstrates the node operator API for on-the-fly pricing configuration.
Shows how operators can adjust prices within safe bounds.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.networking.node_operator_api import PricingParameter


def main():
    print("=" * 80)
    print("Node Operator API Demo")
    print("=" * 80)
    print()
    
    # Create nodes with different configurations
    print("1. Creating Three Nodes with Different Strategies")
    print("-" * 80)
    
    config = DessinConfig.default()
    config.network.port = 8001
    
    node_competitive = DessinNode(config)
    print(f"Node 1 (Competitive): {node_competitive.address[:20]}...")
    
    config2 = DessinConfig.default()
    config2.network.port = 8002
    node_premium = DessinNode(config2)
    print(f"Node 2 (Premium): {node_premium.address[:20]}...")
    
    config3 = DessinConfig.default()
    config3.network.port = 8003
    node_budget = DessinNode(config3)
    print(f"Node 3 (Budget): {node_budget.address[:20]}...")
    print()
    
    # Show initial pricing
    print("2. Initial Pricing (All Nodes Start with Defaults)")
    print("-" * 80)
    pricing = node_competitive.get_operator_pricing()
    print(f"Storage price: {pricing['current_prices']['storage']:.6f} DESSIN/GB/block")
    print(f"Block reward target: {pricing['current_prices']['block_reward_target']:.4f} DESSIN")
    print(f"Training price: {pricing['current_prices']['training']:.6f} DESSIN/iteration")
    print(f"Query price: {pricing['current_prices']['query']:.6f} DESSIN/token")
    print()
    
    # Each node adjusts pricing to their strategy
    print("3. Node 1: Competitive Strategy (Lower Prices)")
    print("-" * 80)
    print("Operator: Setting competitive prices to attract more customers")
    print()
    
    node_competitive.set_storage_price(
        0.0008,
        reason="20% discount to attract storage customers"
    )
    
    node_competitive.set_training_price(
        0.0008,
        reason="Competitive training rates"
    )
    
    node_competitive.set_query_price(
        0.0008,
        reason="Affordable inference"
    )
    print()
    
    # Premium node
    print("4. Node 2: Premium Strategy (Higher Prices, Better Service)")
    print("-" * 80)
    print("Operator: Setting premium prices with guaranteed uptime")
    print()
    
    node_premium.set_storage_price(
        0.0015,
        reason="Premium storage with 99.9% uptime guarantee"
    )
    
    node_premium.set_training_price(
        0.0015,
        reason="High-performance GPU training"
    )
    
    node_premium.set_query_price(
        0.0015,
        reason="Premium inference with low latency"
    )
    
    node_premium.set_block_reward_target(
        15.0,
        reason="Higher expected rewards for premium service"
    )
    print()
    
    # Budget node
    print("5. Node 3: Budget Strategy (Minimal Prices)")
    print("-" * 80)
    print("Operator: Setting budget prices for cost-conscious users")
    print()
    
    node_budget.set_storage_price(
        0.0005,
        reason="Budget storage, best-effort service"
    )
    
    node_budget.set_training_price(
        0.0005,
        reason="CPU training, slower but cheap"
    )
    
    node_budget.set_query_price(
        0.0005,
        reason="Budget inference"
    )
    print()
    
    # Compare all nodes
    print("6. Price Comparison Across Nodes")
    print("-" * 80)
    
    nodes = [
        ("Competitive", node_competitive),
        ("Premium", node_premium),
        ("Budget", node_budget)
    ]
    
    print(f"{'Parameter':<20} {'Competitive':>12} {'Premium':>12} {'Budget':>12}")
    print("-" * 80)
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        if node == nodes[0][1]:
            print(f"{'Storage (GB/block)':<20} "
                  f"{pricing['current_prices']['storage']:>12.6f} ", end="")
        else:
            print(f"{pricing['current_prices']['storage']:>12.6f} ", end="")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        if node == nodes[0][1]:
            print(f"{'Training (iter)':<20} "
                  f"{pricing['current_prices']['training']:>12.6f} ", end="")
        else:
            print(f"{pricing['current_prices']['training']:>12.6f} ", end="")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        if node == nodes[0][1]:
            print(f"{'Query (token)':<20} "
                  f"{pricing['current_prices']['query']:>12.6f} ", end="")
        else:
            print(f"{pricing['current_prices']['query']:>12.6f} ", end="")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        if node == nodes[0][1]:
            print(f"{'Block Reward Target':<20} "
                  f"{pricing['current_prices']['block_reward_target']:>12.4f} ", end="")
        else:
            print(f"{pricing['current_prices']['block_reward_target']:>12.4f} ", end="")
    print()
    print()
    
    # Demonstrate safety bounds
    print("7. Safety Bounds Protection")
    print("-" * 80)
    print("Operator tries to set malicious/extreme prices:")
    print()
    
    print("Attempting to set storage price to 1000 DESSIN/GB/block (way too high)...")
    success = node_competitive.set_storage_price(1000.0)
    print(f"Result: {'✗ Rejected (out of bounds)' if not success else '✓ Accepted'}")
    print()
    
    print("Attempting to set storage price to 0.0000001 DESSIN/GB/block (way too low)...")
    success = node_competitive.set_storage_price(0.0000001)
    print(f"Result: {'✗ Rejected (out of bounds)' if not success else '✓ Accepted'}")
    print()
    
    print("Attempting to set block reward target to 1000 DESSIN...")
    success = node_competitive.set_block_reward_target(1000.0)
    print(f"Result: {'✗ Rejected (out of bounds)' if not success else '✓ Accepted'}")
    print()
    
    # Show bounds
    pricing = node_competitive.get_operator_pricing()
    print("Safety Bounds (Cannot Be Exceeded):")
    print(f"  Storage: [{pricing['bounds']['storage'][0]:.6f}, {pricing['bounds']['storage'][1]:.6f}]")
    print(f"  Block reward: [{pricing['bounds']['block_reward_target'][0]:.4f}, {pricing['bounds']['block_reward_target'][1]:.4f}]")
    print(f"  Training: [{pricing['bounds']['training'][0]:.6f}, {pricing['bounds']['training'][1]:.6f}]")
    print(f"  Query: [{pricing['bounds']['query'][0]:.6f}, {pricing['bounds']['query'][1]:.6f}]")
    print()
    
    # Demonstrate real-time updates
    print("8. Real-Time Price Updates (No Restart Required)")
    print("-" * 80)
    print("Competitive node adjusts prices based on market conditions:")
    print()
    
    print("Market condition: High demand detected")
    node_competitive.set_storage_price(
        0.0012,
        reason="High demand - increasing price to manage load"
    )
    print("✓ Price updated immediately (no restart needed)")
    print()
    
    print("Market condition: Excess capacity detected")
    node_competitive.set_storage_price(
        0.0007,
        reason="Low utilization - lowering price to attract customers"
    )
    print("✓ Price updated immediately (no restart needed)")
    print()
    
    # Show pricing history
    print("9. Pricing History & Transparency")
    print("-" * 80)
    
    history = node_competitive.operator_api.get_pricing_history(
        parameter=PricingParameter.STORAGE_PRICE,
        limit=5
    )
    
    print(f"Recent storage price changes (Node: Competitive):")
    for i, record in enumerate(history, 1):
        print(f"  {i}. {record.old_value:.6f} → {record.new_value:.6f} ({record.change_percent:+.1f}%)")
        if record.reason:
            print(f"     Reason: {record.reason}")
    print()
    
    # Demonstrate bulk updates
    print("10. Bulk Price Updates")
    print("-" * 80)
    print("Premium node adjusts all prices at once:")
    print()
    
    results = node_premium.bulk_update_pricing(
        {
            "storage": 0.002,
            "training": 0.0025,
            "query": 0.002,
            "block_reward": 20.0
        },
        reason="Market-wide price adjustment for premium tier"
    )
    
    print(f"Results:")
    for param, success in results.items():
        print(f"  {param}: {'✓ Updated' if success else '✗ Failed'}")
    print()
    
    # Economic impact analysis
    print("11. Economic Impact Analysis")
    print("-" * 80)
    print()
    
    print("Cost Comparison for 1GB Model (1000 blocks storage):")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        storage_cost = pricing['current_prices']['storage'] * 1.0 * 1000
        print(f"  {node_name:>12}: {storage_cost:7.4f} DESSIN")
    print()
    
    print("Cost Comparison for 1000 Training Iterations:")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        training_cost = pricing['current_prices']['training'] * 1000
        print(f"  {node_name:>12}: {training_cost:7.4f} DESSIN")
    print()
    
    print("Cost Comparison for 10,000 Token Query:")
    print()
    
    for node_name, node in nodes:
        pricing = node.get_operator_pricing()
        query_cost = pricing['current_prices']['query'] * 10000
        print(f"  {node_name:>12}: {query_cost:7.4f} DESSIN")
    print()
    
    # Summary
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ On-the-fly pricing updates (no restart)")
    print("  ✓ Per-node pricing configuration")
    print("  ✓ Safety bounds prevent malicious prices")
    print("  ✓ Complete pricing history tracking")
    print("  ✓ Bulk updates for convenience")
    print("  ✓ Market-responsive pricing")
    print("  ✓ Transparency and auditability")
    print()
    
    print("Safety Features:")
    print("  ✓ Minimum/maximum bounds enforced")
    print("  ✓ Malicious prices rejected")
    print("  ✓ Configuration validation")
    print("  ✓ Audit trail maintained")
    print()
    
    # Cleanup
    node_competitive.stop()
    node_premium.stop()
    node_budget.stop()


if __name__ == "__main__":
    main()
