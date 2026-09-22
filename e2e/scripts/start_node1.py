#!/usr/bin/env python3
"""
Start Node 1 (Bootstrap Node) on port 8001
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dessin import DessinNode
from dessin.runtime.config import DessinConfig


def main():
    print("Starting DeSSIN Node 1 (Bootstrap Node)")
    print("=" * 40)
    
    # Create configuration for Node 1
    config = DessinConfig.default()
    config.network.port = 8001
    config.network.max_peers = 20  # Increased for better connectivity
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds for testing
    
    # Create and start node
    node = DessinNode(config)
    
    try:
        success = node.start()
        if not success:
            print("Failed to start Node 1")
            return 1
        
        print("✓ Node 1 started successfully")
        print(f"Address: {node.address}")
        print(f"Port: 8001")
        print("Role: Bootstrap Node (Mining)")
        print()
        
        # Create a test model
        try:
            node._create_dummy_model()
            models = node.list_models()
            if models:
                print(f"Created test model: {models[0]['model_id']}")
        except Exception as e:
            print(f"Warning: Could not create test model: {e}")
        
        print("\nNode 1 is running...")
        print("Other nodes will connect to this bootstrap node")
        print("Press Ctrl+C to stop")
        print("-" * 40)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down Node 1...")
        node.stop()
        print("Node 1 stopped")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
