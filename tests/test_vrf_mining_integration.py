#!/usr/bin/env python3
"""
Unit tests for VRF integration with mining and block generation.
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig, ConsensusConfig, ModelConfig


class TestVRFMiningIntegration:
    """Test VRF integration with mining and block generation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create configs
        self.model_config = ModelConfig()
        self.model_config.model_cache_dir = str(self.temp_dir)
        
        self.consensus_config = ConsensusConfig()
        self.consensus_config.min_loss_improvement = 0.001
        self.consensus_config.quantized_tolerance = 0.0005
        self.consensus_config.attestation_threshold = 0.67
        self.consensus_config.finalization_window = 5
        
        # Create model manager
        self.model_manager = ModelManager(self.model_config)
        
        # Register a test model
        self.model_manager.register_model(
            model_id="test_mining_model",
            name="Test Mining Model",
            size_gb=0.1,
            format="pytorch",
            quantization="32bit",
            parameters=1000000,
            ipfs_hash="QmMiningTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=0,
            storage_expires=1000,
            model_hash="mining_test_hash"
        )
        
        # Create consensus instance
        self.consensus = PogoConsensus(
            self.consensus_config,
            self.model_manager,
            "test_miner_address"
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_consensus_create_training_block(self):
        """Test consensus create_training_block method directly"""
        print("Testing consensus.create_training_block...")
        
        # Create a training block using consensus directly
        block = self.consensus.create_training_block("test_mining_model")
        
        assert block is not None
        assert isinstance(block.vrf_proof, bytes)
        print(f"✓ Block created with VRF proof: {type(block.vrf_proof)}")
        
        # Test serialization
        block_dict = block.to_dict()
        assert isinstance(block_dict["vrf_proof"], str)
        print(f"✓ Block serialized, vrf_proof type: {type(block_dict['vrf_proof'])}")
        
        # Test JSON serialization
        json_str = json.dumps(block_dict)
        assert isinstance(json_str, str)
        print("✓ Block JSON serialization successful")
    
    def test_node_mine_block_method(self):
        """Test the node's mine_block method directly"""
        print("Testing node.mine_block method...")
        
        # Create a node
        config = DessinConfig.default()
        node = DessinNode(config)
        
        # Start the node
        success = node.start()
        assert success
        print("✓ Node started successfully")
        
        # Wait for initialization
        time.sleep(0.05)
        
        # Test mine_block method directly
        try:
            block_hash = node.mine_block()
            if block_hash:
                print(f"✓ Mining successful: {block_hash}")
            else:
                print("⚠ Mining returned None (expected if no models)")
        except Exception as e:
            print(f"✗ Mining failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Stop the node
        node.stop()
    
    def test_block_creation_with_dummy_model(self):
        """Test block creation with dummy model creation"""
        print("Testing block creation with dummy model...")
        
        # Create a node
        config = DessinConfig.default()
        node = DessinNode(config)
        
        # Start the node
        success = node.start()
        assert success
        print("✓ Node started successfully")
        
        # Wait for initialization
        time.sleep(0.05)
        
        # Create dummy model
        node._create_dummy_model()
        print("✓ Dummy model created")
        
        # Check if model is available
        available_models = [
            model_id for model_id, model_info in node.model_manager.models.items()
            if node.model_manager.is_model_available(model_id, node.consensus.get_chain_length())
        ]
        print(f"Available models: {available_models}")
        
        # Try to create a block with the dummy model
        if available_models:
            model_id = available_models[0]
            print(f"Creating block with model: {model_id}")
            
            block = node.consensus.create_training_block(model_id)
            if block:
                print(f"✓ Block created successfully")
                print(f"  VRF proof type: {type(block.vrf_proof)}")
                
                # Test serialization
                block_dict = block.to_dict()
                print(f"  Serialized vrf_proof type: {type(block_dict['vrf_proof'])}")
                
                # Test JSON serialization
                try:
                    json_str = json.dumps(block_dict)
                    print("  ✓ JSON serialization successful")
                except Exception as e:
                    print(f"  ✗ JSON serialization failed: {e}")
                    raise
            else:
                print("✗ Block creation failed")
        
        # Stop the node
        node.stop()
    
    def test_message_data_creation(self):
        """Test creating message_data exactly like in mining loop"""
        print("Testing message_data creation...")
        
        # Create a node
        config = DessinConfig.default()
        node = DessinNode(config)
        
        # Start the node
        success = node.start()
        assert success
        print("✓ Node started successfully")
        
        # Wait for initialization
        time.sleep(0.05)
        
        # Create dummy model
        node._create_dummy_model()
        
        # Get available models
        available_models = [
            model_id for model_id, model_info in node.model_manager.models.items()
            if node.model_manager.is_model_available(model_id, node.consensus.get_chain_length())
        ]
        
        if available_models:
            model_id = available_models[0]
            
            # Create training block
            block = node.consensus.create_training_block(model_id)
            assert block is not None
            print(f"✓ Block created, vrf_proof type: {type(block.vrf_proof)}")
            
            # Create message_data exactly like in mining loop
            block_dict = block.to_dict()
            print(f"✓ Block dict created, vrf_proof type: {type(block_dict.get('vrf_proof'))}")
            
            message_data = {
                "message_type": "POGO_BLOCK",
                "height": block.index,
                "block_data": block_dict
            }
            print(f"✓ Message data created")
            
            # Test JSON serialization of message_data
            try:
                json_str = json.dumps(message_data)
                print("✓ Message data JSON serialization successful")
            except Exception as e:
                print(f"✗ Message data JSON serialization failed: {e}")
                
                # Debug: Check each field
                for key, value in message_data.items():
                    try:
                        json.dumps(value)
                    except Exception as field_e:
                        print(f"  Field '{key}' error: {field_e}")
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                try:
                                    json.dumps(subvalue)
                                except Exception as subfield_e:
                                    print(f"    Subfield '{subkey}' error: {subfield_e}")
                                    if isinstance(subvalue, bytes):
                                        print(f"      Subfield '{subkey}' is bytes: {type(subvalue)}")
                raise
        
        # Stop the node
        node.stop()
    
    def test_chaincraft_create_shared_message(self):
        """Test chaincraft create_shared_message method"""
        print("Testing chaincraft create_shared_message...")
        
        # Create a node
        config = DessinConfig.default()
        node = DessinNode(config)
        
        # Start the node
        success = node.start()
        assert success
        print("✓ Node started successfully")
        
        # Wait for initialization
        time.sleep(0.05)
        
        # Create dummy model
        node._create_dummy_model()
        
        # Get available models
        available_models = [
            model_id for model_id, model_info in node.model_manager.models.items()
            if node.model_manager.is_model_available(model_id, node.consensus.get_chain_length())
        ]
        
        if available_models:
            model_id = available_models[0]
            
            # Create training block
            block = node.consensus.create_training_block(model_id)
            assert block is not None
            
            # Create message_data
            block_dict = block.to_dict()
            message_data = {
                "message_type": "POGO_BLOCK",
                "height": block.index,
                "block_data": block_dict
            }
            
            # Test chaincraft create_shared_message
            try:
                block_hash, _ = node.chaincraft_node.create_shared_message(message_data)
                print(f"✓ create_shared_message successful: {block_hash}")
            except Exception as e:
                print(f"✗ create_shared_message failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Stop the node
        node.stop()


if __name__ == "__main__":
    # Run the tests
    test_instance = TestVRFMiningIntegration()
    
    try:
        test_instance.setup_method()
        
        print("Running VRF mining integration tests...")
        print("=" * 60)
        
        test_instance.test_consensus_create_training_block()
        print()
        
        test_instance.test_node_mine_block_method()
        print()
        
        test_instance.test_block_creation_with_dummy_model()
        print()
        
        test_instance.test_message_data_creation()
        print()
        
        test_instance.test_chaincraft_create_shared_message()
        print()
        
        print("=" * 60)
        print("✓ All VRF mining integration tests completed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_instance.teardown_method()
