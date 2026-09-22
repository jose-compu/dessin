#!/usr/bin/env python3
"""
Comprehensive unit test to trace the origin of the bytes vrf_proof issue.
Using chaincraft to build a complex test that mimics our exact mining flow.
"""

import sys
import os
import json
import time
import tempfile
import base64
from pathlib import Path

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig, ConsensusConfig, ModelConfig
from chaincraft.node import ChaincraftNode
from chaincraft.crypto_primitives.vrf import ECDSAVRFPrimitive


class TestVRFBytesOrigin:
    """Test to trace the exact origin of the bytes vrf_proof issue"""
    
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
            model_id="test_bytes_model",
            name="Test Bytes Model",
            size_gb=0.1,
            format="pytorch",
            quantization="32bit",
            parameters=1000000,
            ipfs_hash="QmBytesTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=0,
            storage_expires=1000,
            model_hash="bytes_test_hash"
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
    
    def _trace_bytes(self, obj, path="", depth=0):
        """Recursively trace all bytes objects in a data structure"""
        indent = "  " * depth
        if isinstance(obj, bytes):
            print(f"{indent}🔴 BYTES FOUND at {path}: {type(obj)} - {len(obj)} bytes")
            print(f"{indent}   Content: {obj[:20]}...")
            return True
        elif isinstance(obj, dict):
            found_bytes = False
            print(f"{indent}📁 Dict at {path}: {len(obj)} keys")
            for k, v in obj.items():
                if self._trace_bytes(v, f"{path}.{k}" if path else k, depth + 1):
                    found_bytes = True
            return found_bytes
        elif isinstance(obj, list):
            found_bytes = False
            print(f"{indent}📋 List at {path}: {len(obj)} items")
            for i, v in enumerate(obj):
                if self._trace_bytes(v, f"{path}[{i}]" if path else f"[{i}]", depth + 1):
                    found_bytes = True
            return found_bytes
        else:
            print(f"{indent}📄 {type(obj).__name__} at {path}: {str(obj)[:50]}...")
            return False
    
    def test_step1_vrf_creation(self):
        """Step 1: Test VRF creation in isolation"""
        print("\n" + "="*60)
        print("STEP 1: Testing VRF creation in isolation")
        print("="*60)
        
        # Create VRF primitive
        vrf_primitive = ECDSAVRFPrimitive()
        vrf_primitive.generate_key()
        
        # Create VRF proof
        message = b'test_message'
        proof = vrf_primitive.sign(message)
        
        print(f"VRF proof type: {type(proof)}")
        print(f"VRF proof length: {len(proof)}")
        print(f"VRF proof content: {proof[:20]}...")
        
        # Test encoding
        encoded = base64.b64encode(proof).decode('utf-8')
        print(f"Encoded type: {type(encoded)}")
        print(f"Encoded length: {len(encoded)}")
        
        # Test JSON serialization
        test_data = {'vrf_proof': encoded}
        try:
            json_str = json.dumps(test_data)
            print("✓ VRF proof JSON serialization successful")
        except Exception as e:
            print(f"✗ VRF proof JSON serialization failed: {e}")
            
        assert isinstance(proof, bytes)
        assert isinstance(encoded, str)
    
    def test_step2_consensus_block_creation(self):
        """Step 2: Test consensus block creation"""
        print("\n" + "="*60)
        print("STEP 2: Testing consensus block creation")
        print("="*60)
        
        # Create training block
        block = self.consensus.create_training_block("test_bytes_model")
        
        assert block is not None
        print(f"Block created successfully")
        print(f"Block VRF proof type: {type(block.vrf_proof)}")
        
        # Trace the block object
        print("\nTracing block object:")
        self._trace_bytes(block.__dict__, "block")
        
        assert isinstance(block.vrf_proof, bytes)
    
    def test_step3_block_to_dict_conversion(self):
        """Step 3: Test block to_dict conversion"""
        print("\n" + "="*60)
        print("STEP 3: Testing block to_dict conversion")
        print("="*60)
        
        # Create training block
        block = self.consensus.create_training_block("test_bytes_model")
        
        print(f"Original block VRF proof type: {type(block.vrf_proof)}")
        
        # Convert to dict
        block_dict = block.to_dict()
        
        print(f"Dict VRF proof type: {type(block_dict.get('vrf_proof'))}")
        
        # Trace the dict
        print("\nTracing block dict:")
        has_bytes = self._trace_bytes(block_dict, "block_dict")
        
        if has_bytes:
            print("🚨 PROBLEM: Block dict still contains bytes!")
        else:
            print("✓ Block dict is clean - no bytes found")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(block_dict)
            print("✓ Block dict JSON serialization successful")
        except Exception as e:
            print(f"✗ Block dict JSON serialization failed: {e}")
            raise
        
        assert not has_bytes, "Block dict should not contain bytes"
    
    def test_step4_message_data_creation(self):
        """Step 4: Test message data creation"""
        print("\n" + "="*60)
        print("STEP 4: Testing message data creation")
        print("="*60)
        
        # Create training block
        block = self.consensus.create_training_block("test_bytes_model")
        block_dict = block.to_dict()
        
        # Create message data exactly like in mining - Tendermint style
        message_data = {
            "message_type": "POGO_BLOCK",
            "height": block.index,
            "block_data": block_dict
        }
        
        print(f"Message data type: {type(message_data)}")
        print(f"Block data VRF proof type: {type(message_data['block_data']['vrf_proof'])}")
        
        # Trace the message data
        print("\nTracing message data:")
        has_bytes = self._trace_bytes(message_data, "message_data")
        
        if has_bytes:
            print("🚨 PROBLEM: Message data contains bytes!")
        else:
            print("✓ Message data is clean - no bytes found")
        
        # Test JSON serialization
        try:
            json_str = json.dumps(message_data)
            print("✓ Message data JSON serialization successful")
        except Exception as e:
            print(f"✗ Message data JSON serialization failed: {e}")
            raise
        
        assert not has_bytes, "Message data should not contain bytes"
    
    def test_step5_chaincraft_node_creation(self):
        """Step 5: Test chaincraft node creation and message passing"""
        print("\n" + "="*60)
        print("STEP 5: Testing chaincraft node creation and message passing")
        print("="*60)
        
        # Create chaincraft node
        node = ChaincraftNode(max_peers=10, port=8001)
        node.start()
        print("✓ Chaincraft node started")
        
        # Create clean message data
        block = self.consensus.create_training_block("test_bytes_model")
        block_dict = block.to_dict()
        message_data = {
            "message_type": "POGO_BLOCK",
            "height": block.index,
            "block_data": block_dict
        }
        
        # Verify it's clean before passing to chaincraft
        has_bytes_before = self._trace_bytes(message_data, "message_data_before_chaincraft")
        
        print(f"Message data has bytes before chaincraft: {has_bytes_before}")
        
        # Test with chaincraft
        try:
            result = node.create_shared_message(message_data)
            print("✓ Chaincraft create_shared_message successful")
            print(f"Result: {result}")
        except Exception as e:
            print(f"✗ Chaincraft create_shared_message failed: {e}")
            
            # Trace again after the error
            print("\nTracing message data after chaincraft error:")
            has_bytes_after = self._trace_bytes(message_data, "message_data_after_error")
            print(f"Message data has bytes after error: {has_bytes_after}")
            
            # Check if chaincraft modified our data
            if has_bytes_after and not has_bytes_before:
                print("🚨 CHAINCRAFT MODIFIED OUR DATA!")
            elif has_bytes_after:
                print("🚨 BYTES WERE PRESENT BEFORE CHAINCRAFT")
            
            raise
        
        assert not has_bytes_before, "Message data should not contain bytes before chaincraft"
    
    def test_step6_full_mining_simulation(self):
        """Step 6: Full mining simulation using DessinNode"""
        print("\n" + "="*60)
        print("STEP 6: Full mining simulation using DessinNode")
        print("="*60)
        
        # Create DessinNode
        config = DessinConfig.default()
        node = DessinNode(config)
        
        # Start the node
        success = node.start()
        assert success
        print("✓ DessinNode started")
        
        # Wait for initialization
        time.sleep(0.05)
        
        # Create dummy model
        node._create_dummy_model()
        
        # Get available models
        available_models = [
            model_id for model_id, model_info in node.model_manager.models.items()
            if node.model_manager.is_model_available(model_id, node.consensus.get_chain_length())
        ]
        
        assert available_models, "Should have available models"
        model_id = available_models[0]
        print(f"Using model: {model_id}")
        
        # Step-by-step mining simulation
        print("\n--- Creating training block ---")
        block = node.consensus.create_training_block(model_id)
        assert block is not None
        print(f"Block VRF proof type: {type(block.vrf_proof)}")
        
        print("\n--- Converting block to dict ---")
        block_dict = block.to_dict()
        print(f"Dict VRF proof type: {type(block_dict.get('vrf_proof'))}")
        
        # Trace the block dict
        print("\nTracing block dict in full simulation:")
        has_bytes_dict = self._trace_bytes(block_dict, "simulation_block_dict")
        
        print("\n--- Creating message data ---")
        message_data = {
            "message_type": "POGO_BLOCK",
            "height": block.index,
            "block_data": block_dict
        }
        
        # Trace the message data
        print("\nTracing message data in full simulation:")
        has_bytes_msg = self._trace_bytes(message_data, "simulation_message_data")
        
        print("\n--- Testing chaincraft ---")
        try:
            result = node.chaincraft_node.create_shared_message(message_data)
            print("✓ Full mining simulation successful")
        except Exception as e:
            print(f"✗ Full mining simulation failed: {e}")
            
            # Final trace to see what went wrong
            print("\nFinal trace after error:")
            self._trace_bytes(message_data, "final_message_data")
            
            raise
        
        # Stop the node
        node.stop()
        
        assert not has_bytes_dict, "Block dict should not contain bytes"
        assert not has_bytes_msg, "Message data should not contain bytes"
    
    def test_step7_deep_copy_investigation(self):
        """Step 7: Investigate if deep copy affects the data"""
        print("\n" + "="*60)
        print("STEP 7: Investigating deep copy effects")
        print("="*60)
        
        import copy
        
        # Create training block
        block = self.consensus.create_training_block("test_bytes_model")
        block_dict = block.to_dict()
        
        print("Original block dict:")
        has_bytes_orig = self._trace_bytes(block_dict, "original")
        
        # Test shallow copy
        shallow_copy = block_dict.copy()
        print("\nShallow copy:")
        has_bytes_shallow = self._trace_bytes(shallow_copy, "shallow_copy")
        
        # Test deep copy
        deep_copy = copy.deepcopy(block_dict)
        print("\nDeep copy:")
        has_bytes_deep = self._trace_bytes(deep_copy, "deep_copy")
        
        # Test JSON round-trip
        json_str = json.dumps(block_dict)
        json_copy = json.loads(json_str)
        print("\nJSON round-trip:")
        has_bytes_json = self._trace_bytes(json_copy, "json_copy")
        
        assert not has_bytes_orig, "Original should not have bytes"
        assert not has_bytes_shallow, "Shallow copy should not have bytes"
        assert not has_bytes_deep, "Deep copy should not have bytes"
        assert not has_bytes_json, "JSON copy should not have bytes"


if __name__ == "__main__":
    # Run the tests manually
    test_instance = TestVRFBytesOrigin()
    
    try:
        test_instance.setup_method()
        
        print("🔍 COMPREHENSIVE VRF BYTES ORIGIN INVESTIGATION")
        print("=" * 80)
        
        test_instance.test_step1_vrf_creation()
        test_instance.test_step2_consensus_block_creation()
        test_instance.test_step3_block_to_dict_conversion()
        test_instance.test_step4_message_data_creation()
        test_instance.test_step5_chaincraft_node_creation()
        test_instance.test_step6_full_mining_simulation()
        test_instance.test_step7_deep_copy_investigation()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - NO BYTES ORIGIN FOUND!")
        print("="*80)
        
    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_instance.teardown_method()
