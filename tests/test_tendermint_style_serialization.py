#!/usr/bin/env python3
"""
Test VRF serialization using Tendermint-style approach
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig, ConsensusConfig, ModelConfig
from chaincraft.node import ChaincraftNode
from chaincraft.crypto_primitives.vrf import ECDSAVRFPrimitive


def test_tendermint_style_vrf():
    """Test VRF serialization using Tendermint approach (hex encoding)"""
    print("\n" + "="*60)
    print("TESTING TENDERMINT-STYLE VRF SERIALIZATION")
    print("="*60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create configs
        model_config = ModelConfig()
        model_config.model_cache_dir = str(temp_dir)
        
        consensus_config = ConsensusConfig()
        
        # Create model manager
        model_manager = ModelManager(model_config)
        
        # Register a test model
        model_manager.register_model(
            model_id="test_tendermint_model",
            name="Test Tendermint Model",
            size_gb=0.1,
            format="pytorch",
            quantization="32bit",
            parameters=1000000,
            ipfs_hash="QmTendermintTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=0,
            storage_expires=1000,
            model_hash="tendermint_test_hash"
        )
        
        # Create consensus instance
        consensus = PogoConsensus(
            consensus_config,
            model_manager,
            "test_miner_address"
        )
        
        print("✓ Created consensus instance")
        
        # Create a training block
        block = consensus.create_training_block("test_tendermint_model")
        assert block is not None
        print(f"✓ Created block with VRF proof type: {type(block.vrf_proof)}")
        
        # Convert to dict (Tendermint style)
        block_dict = block.to_dict()
        print(f"✓ Block dict VRF proof type: {type(block_dict['vrf_proof'])}")
        
        # Verify it's a hex string
        assert isinstance(block_dict['vrf_proof'], str)
        assert all(c in '0123456789abcdef' for c in block_dict['vrf_proof'].lower())
        print("✓ VRF proof is valid hex string")
        
        # Test JSON serialization (like Tendermint does)
        json_str = json.dumps(block_dict)
        print("✓ Block dict JSON serialization successful")
        
        # Test deserialization
        deserialized_dict = json.loads(json_str)
        deserialized_block = PogoBlock.from_dict(deserialized_dict)
        print("✓ Block deserialization successful")
        
        # Verify roundtrip
        assert isinstance(deserialized_block.vrf_proof, bytes)
        assert deserialized_block.vrf_proof == block.vrf_proof
        print("✓ VRF proof roundtrip successful")
        
        # Test with chaincraft (the critical test)
        print("\n--- Testing with Chaincraft ---")
        
        # Create chaincraft node
        chaincraft_node = ChaincraftNode(max_peers=10, port=0)
        chaincraft_node.start()
        print("✓ Chaincraft node started")
        
        # Create message data (Tendermint style)
        message_data = {
            "message_type": "POGO_BLOCK",
            "height": block.index,
            "block_data": block_dict  # This should contain only simple types
        }
        
        # Verify no bytes in message data
        def check_for_bytes(obj, path=""):
            if isinstance(obj, bytes):
                print(f"🔴 BYTES FOUND at {path}: {type(obj)}")
                return True
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if check_for_bytes(v, f"{path}.{k}" if path else k):
                        return True
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if check_for_bytes(v, f"{path}[{i}]" if path else f"[{i}]"):
                        return True
            return False
        
        has_bytes = check_for_bytes(message_data, "message_data")
        assert not has_bytes, "Message data should not contain bytes"
        print("✓ Message data contains no bytes objects")
        
        # Test JSON serialization of message data
        json.dumps(message_data)
        print("✓ Message data JSON serialization successful")
        
        # The critical test - pass to chaincraft
        try:
            result = chaincraft_node.create_shared_message(message_data)
            print("✓ Chaincraft create_shared_message SUCCESSFUL!")
            print(f"Result: {result[0][:8]}...")
        except Exception as e:
            print(f"✗ Chaincraft create_shared_message FAILED: {e}")
            raise
        
        print("\n" + "="*60)
        print("🎉 TENDERMINT-STYLE APPROACH SUCCESSFUL!")
        print("="*60)
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_full_mining_simulation_tendermint():
    """Test full mining simulation with Tendermint approach"""
    print("\n" + "="*60)
    print("TESTING FULL MINING SIMULATION - TENDERMINT STYLE")
    print("="*60)
    
    # Create DessinNode
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds
    
    node = DessinNode(config)
    success = node.start()
    assert success
    print("✓ DessinNode started")
    
    try:
        # Wait for initialization
        time.sleep(0.05)
        
        # Create dummy model
        node._create_dummy_model()
        print("✓ Dummy model created")

        # Affordability pre-check (added in the four-node battery work) requires
        # the model owner to be able to pay the miner. In this single-node demo
        # the owner is the node itself; pre-fund it from genesis-equivalent so
        # the gate lets the round through.
        node.consensus.economic_system.balances[node.address] = 100.0

        # Test mining
        block_hash = node.mine_block()
        assert block_hash is not None
        print(f"✓ Mining successful! Block hash: {block_hash[:8]}...")
        
        print("\n" + "="*60)
        print("🎉 FULL MINING SIMULATION SUCCESSFUL!")
        print("="*60)
        
    finally:
        node.stop()


if __name__ == "__main__":
    print("🔍 TESTING TENDERMINT-STYLE VRF SERIALIZATION")
    print("=" * 80)
    
    try:
        test_tendermint_style_vrf()
        test_full_mining_simulation_tendermint()
        
        print("\n" + "="*80)
        print("🎉 ALL TENDERMINT-STYLE TESTS PASSED!")
        print("The bytes serialization issue has been RESOLVED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
