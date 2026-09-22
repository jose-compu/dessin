#!/usr/bin/env python3
"""
Start Node 2 (Miner Node) on port 8002
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dessin import DessinNode
from dessin.runtime.config import DessinConfig


def main():
    print("Starting DeSSIN Node 2 (Miner Node)")
    print("=" * 40)
    
    # Create configuration for Node 2
    config = DessinConfig.default()
    config.network.port = 8002
    config.network.max_peers = 20  # Increased for better connectivity
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds for testing
    
    # Create and start node
    node = DessinNode(config)
    
    try:
        success = node.start()
        if not success:
            print("Failed to start Node 2")
            return 1
        
        print("✓ Node 2 started successfully")
        print(f"Address: {node.address}")
        print(f"Port: 8002")
        print("Role: Miner Node")
        print()
        
        print("Node 2 is running...")
        print("Connecting to other nodes...")
        
        # Connect to other nodes
        peers = [8001, 8003, 8004]  # Connect to all other nodes
        for port in peers:
            try:
                result = node.chaincraft_node.connect_to_peer('127.0.0.1', port)
                # connect_to_peer returns None on success
                if result is None:
                    print(f"✓ Connected to Node on port {port}")
                else:
                    print(f"⚠ Failed to connect to Node on port {port}")
                time.sleep(0.5)  # Small delay between connections
            except Exception as e:
                print(f"⚠ Connection error to port {port}: {e}")
        
        print("Press Ctrl+C to stop")
        print("-" * 40)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down Node 2...")
        node.stop()
        print("Node 2 stopped")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
