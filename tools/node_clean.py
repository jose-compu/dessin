#!/usr/bin/env python3
"""
Clean version of the mining method for testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
import time

def test_clean_mining():
    print("Testing clean mining method...")
    
    # Create a node
    config = DessinConfig.default()
    node = DessinNode(config)
    
    # Start the node
    success = node.start()
    if not success:
        print("Failed to start node")
        return
    
    print("Node started successfully")
    
    # Wait for initialization
    time.sleep(2)
    
    # Create dummy model
    node._create_dummy_model()
    
    # Get available models
    available_models = [
        model_id for model_id, model_info in node.model_manager.models.items()
        if node.model_manager.is_model_available(model_id, node.consensus.get_chain_length())
    ]
    
    if available_models:
        model_id = available_models[0]
        print(f"Creating block with model: {model_id}")
        
        # Create training block
        block = node.consensus.create_training_block(model_id)
        if not block:
            print("Failed to create training block")
            return
        
        print(f"Block created successfully")
        print(f"VRF proof type: {type(block.vrf_proof)}")
        
        # Convert to dict
        block_dict = block.to_dict()
        print(f"Block dict vrf_proof type: {type(block_dict.get('vrf_proof'))}")
        
        # Test JSON serialization
        import json
        try:
            json_str = json.dumps(block_dict)
            print("✓ Block dict JSON serialization successful")
        except Exception as e:
            print(f"✗ Block dict JSON serialization failed: {e}")
            return
        
        # Create message data - Tendermint style
        message_data = {
            "message_type": "POGO_BLOCK",
            "height": block.index,
            "block_data": block_dict
        }
        
        # Test JSON serialization of message data
        try:
            json_str = json.dumps(message_data)
            print("✓ Message data JSON serialization successful")
        except Exception as e:
            print(f"✗ Message data JSON serialization failed: {e}")
            return
        
        # Test chaincraft
        try:
            block_hash, _ = node.chaincraft_node.create_shared_message(message_data)
            print(f"✓ Chaincraft create_shared_message successful: {block_hash}")
        except Exception as e:
            print(f"✗ Chaincraft create_shared_message failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Stop the node
    node.stop()

if __name__ == "__main__":
    test_clean_mining()
