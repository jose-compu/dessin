#!/usr/bin/env python3
"""
Script to run a 4-node DeSSIN network for testing.
This script starts 4 nodes with different ports and lets them discover each other.
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dessin import DessinNode
from dessin.runtime.config import DessinConfig


def start_node(node_id, port, mining=True):
    """Start a single node"""
    print(f"Starting Node {node_id} on port {port}...")
    
    # Create configuration
    config = DessinConfig.default()
    config.network.port = port
    config.network.max_peers = 20  # Increased for better connectivity
    config.consensus.block_time_minutes = (15 / 60.0) if mining else 0  # 15 seconds for testing
    
    # Create and start node
    node = DessinNode(config)
    success = node.start()
    
    if success:
        print(f"✓ Node {node_id} started successfully")
        print(f"  Address: {node.address}")
        print(f"  Port: {port}")
        print(f"  Mining: {'Yes' if mining else 'No'}")
        
        # Create a test model on the first node
        if node_id == 1:
            try:
                node._create_dummy_model()
                models = node.list_models()
                if models:
                    print(f"  Created test model: {models[0]['model_id']}")
            except Exception as e:
                print(f"  Warning: Could not create test model: {e}")
        
        return node
    else:
        print(f"✗ Failed to start Node {node_id}")
        return None


def main():
    """Main function to run the 4-node network"""
    print("DeSSIN 4-Node Network Setup")
    print("=" * 40)
    
    # Node configurations
    nodes_config = [
        {"id": 1, "port": 8001, "mining": True, "name": "Bootstrap Node"},
        {"id": 2, "port": 8002, "mining": True, "name": "Miner Node 2"},
        {"id": 3, "port": 8003, "mining": True, "name": "Miner Node 3"},
        {"id": 4, "port": 8004, "mining": False, "name": "Validator Node 4"}
    ]
    
    nodes = []
    threads = []
    
    try:
        # Start all nodes
        for config in nodes_config:
            node = start_node(config["id"], config["port"], config["mining"])
            if node:
                nodes.append(node)
                print()
            else:
                print(f"Failed to start {config['name']}")
                return 1
        
        if not nodes:
            print("No nodes started successfully")
            return 1
        
        # Connect nodes to each other
        print("Connecting nodes to each other...")
        import time
        
        # Give nodes time to fully start
        time.sleep(2)
        
        # Connect each node to all other nodes
        for i, node in enumerate(nodes):
            print(f"  Connecting Node {i+1} to other nodes...")
            for j, other_node in enumerate(nodes):
                if i != j:  # Don't connect to self
                    try:
                        # Connect to other nodes
                        result = node.chaincraft_node.connect_to_peer('127.0.0.1', other_node.chaincraft_node.port)
                        # connect_to_peer returns None on success, so we check for None
                        if result is None:
                            print(f"    ✓ Node {i+1} -> Node {j+1}: Connected")
                        else:
                            print(f"    ✗ Node {i+1} -> Node {j+1}: Failed")
                        time.sleep(0.5)  # Small delay between connections
                    except Exception as e:
                        print(f"    ✗ Node {i+1} -> Node {j+1}: Error - {e}")
        
        # Give connections time to establish
        print("Waiting for connections to establish...")
        time.sleep(3)
        
        print(f"✓ Network running with {len(nodes)} nodes")
        print("\nNetwork Status:")
        print("-" * 40)
        
        for i, config in enumerate(nodes_config[:len(nodes)]):
            node = nodes[i]
            print(f"Node {config['id']} ({config['name']}):")
            print(f"  Address: {node.address}")
            print(f"  Port: {config['port']}")
            print(f"  Mining: {'Yes' if config['mining'] else 'No'}")
            print()
        
        print("Network is running...")
        print("Nodes will automatically discover each other via local discovery")
        print("Press Ctrl+C to stop all nodes")
        print("-" * 40)
        
        # Monitor the network
        start_time = time.time()
        while True:
            time.sleep(10)
            
            # Print periodic status
            elapsed = time.time() - start_time
            print(f"\n[Network Status - {elapsed:.0f}s elapsed]")
            
            for i, node in enumerate(nodes):
                try:
                    # Get basic node info
                    info = node.get_node_info()
                    chain_info = node.get_chain_info()
                    
                    print(f"Node {i+1}: Chain height {chain_info.get('height', 0)}, "
                          f"Peers: {len(node.chaincraft_node.peers) if hasattr(node.chaincraft_node, 'peers') else 'Unknown'}")
                    
                except Exception as e:
                    print(f"Node {i+1}: Error getting status - {e}")
            
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nShutting down network...")
        
        # Stop all nodes
        for i, node in enumerate(nodes):
            try:
                print(f"Stopping Node {i+1}...")
                node.stop()
            except Exception as e:
                print(f"Error stopping Node {i+1}: {e}")
        
        print("Network stopped successfully")
        return 0
    
    except Exception as e:
        print(f"Error running network: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
