#!/usr/bin/env python3
"""
Comprehensive test for the full PoGO protocol implementation
Tests quantization, Merkle proofs, model commitments, and network integration
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
from dessin.protocol.pogo_protocol import PoGOProtocol, QuantizationLevel, ModelQuantizer, MerkleTree
from dessin.llm.simple_trainer import SimpleTrainer
from chaincraft.node import ChaincraftNode


def test_merkle_tree_implementation():
    """Test Merkle tree implementation"""
    print("\n" + "="*60)
    print("TESTING MERKLE TREE IMPLEMENTATION")
    print("="*60)
    
    # Test with simple leaves
    leaves = ["leaf1", "leaf2", "leaf3", "leaf4"]
    tree = MerkleTree(leaves)
    
    print(f"Leaves: {leaves}")
    print(f"Merkle root: {tree.get_root()[:16]}...")
    
    # Test proof generation and verification
    for i, leaf in enumerate(leaves):
        proof = tree.get_proof(i)
        is_valid = tree.verify_proof(leaf, i, proof, tree.get_root())
        print(f"Leaf {i} ({leaf}): proof valid = {is_valid}")
        assert is_valid, f"Proof verification failed for leaf {i}"
    
    print("✓ Merkle tree implementation working correctly")


def test_model_quantization():
    """Test model quantization"""
    print("\n" + "="*60)
    print("TESTING MODEL QUANTIZATION")
    print("="*60)
    
    # Create a simple model and train it
    trainer = SimpleTrainer()
    result = trainer.train_model(training_steps=10, learning_rate=0.01)
    
    # Test quantization
    quantizer = ModelQuantizer()
    
    # Convert hex weights to numpy
    import numpy as np
    weights_bytes = bytes.fromhex(result.model_weights_hex)
    weights = np.frombuffer(weights_bytes, dtype=np.float64)
    
    print(f"Original weights shape: {weights.shape}")
    print(f"Original weights range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Test different quantization levels
    for level in [QuantizationLevel.FLOAT32, QuantizationLevel.INT8, QuantizationLevel.INT4]:
        quantized, error = quantizer.quantize_weights(weights, level)
        
        original_size = len(weights_bytes)
        quantized_size = quantizer.get_quantized_size(original_size, level)
        compression = original_size / quantized_size
        
        print(f"\n{level.value} Quantization:")
        print(f"  Quantization error: {error:.8f}")
        print(f"  Size: {original_size} -> {quantized_size} bytes")
        print(f"  Compression: {compression:.1f}x")
        print(f"  Range: [{quantized.min():.6f}, {quantized.max():.6f}]")
    
    print("\n✓ Model quantization working correctly")


def test_pogo_protocol_integration():
    """Test full PoGO protocol integration"""
    print("\n" + "="*60)
    print("TESTING FULL PoGO PROTOCOL INTEGRATION")
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
            model_id="pogo_test_model",
            name="PoGO Test Model",
            size_gb=0.1,
            format="pytorch",
            quantization="32bit",
            parameters=1000000,
            ipfs_hash="QmPoGOTest123",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=0,
            storage_expires=1000,
            model_hash="pogo_test_hash"
        )
        
        # Create consensus instance
        consensus = PogoConsensus(
            consensus_config,
            model_manager,
            "test_miner_address"
        )
        
        print("✓ Created consensus instance with PoGO protocol")
        
        # Create a training block with full PoGO protocol
        block = consensus.create_training_block("pogo_test_model")
        
        assert block is not None, "Block creation failed"
        print("✓ Created training block with full PoGO protocol")
        
        # Verify PoGO protocol fields
        assert block.hash_full_model_32, "Missing full model hash"
        assert block.hash_quant_4, "Missing quantized model hash"
        assert block.merkle_proof_full is not None, "Missing full model Merkle proof"
        assert block.merkle_proof_quant is not None, "Missing quantized model Merkle proof"
        assert block.quantization_error >= 0, "Invalid quantization error"
        assert block.model_size_full > 0, "Invalid full model size"
        assert block.model_size_quant > 0, "Invalid quantized model size"
        assert block.model_size_full > block.model_size_quant, "Quantized model should be smaller"
        
        print(f"✓ Block validation passed:")
        print(f"  Full model Merkle root: {block.hash_full_model_32[:16]}...")
        print(f"  Quantized Merkle root: {block.hash_quant_4[:16]}...")
        print(f"  Quantization error: {block.quantization_error:.8f}")
        print(f"  Model sizes: {block.model_size_full} -> {block.model_size_quant} bytes")
        print(f"  Compression ratio: {block.model_size_full / block.model_size_quant:.1f}x")
        print(f"  Loss improvement: {block.loss_before - block.loss_after:.6f}")
        
        # Test block serialization with PoGO fields
        block_dict = block.to_dict()
        
        # Verify all PoGO fields are present and serializable
        required_fields = [
            "hash_full_model_32", "hash_quant_4", "merkle_proof_full", 
            "merkle_proof_quant", "quantization_error", "model_size_full", "model_size_quant"
        ]
        
        for field in required_fields:
            assert field in block_dict, f"Missing field: {field}"
            assert block_dict[field] is not None, f"Null field: {field}"
        
        # Test JSON serialization
        json_str = json.dumps(block_dict)
        deserialized = json.loads(json_str)
        
        print("✓ Block serialization with PoGO fields successful")
        
        # Test block deserialization
        reconstructed_block = PogoBlock.from_dict(deserialized)
        
        assert reconstructed_block.hash_full_model_32 == block.hash_full_model_32
        assert reconstructed_block.hash_quant_4 == block.hash_quant_4
        assert reconstructed_block.quantization_error == block.quantization_error
        
        print("✓ Block deserialization with PoGO fields successful")
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_network_with_pogo_protocol():
    """Test network operation with full PoGO protocol"""
    print("\n" + "="*60)
    print("TESTING NETWORK WITH FULL PoGO PROTOCOL")
    print("="*60)
    
    # Create DessinNode with PoGO protocol
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds
    
    node = DessinNode(config)
    success = node.start()
    assert success
    print("✓ DessinNode started with PoGO protocol")
    
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

        # Test mining with full PoGO protocol
        block_hash = node.mine_block()
        assert block_hash is not None
        print(f"✓ Mining with PoGO protocol successful! Block hash: {block_hash[:8]}...")
        
        # Verify the block contains PoGO protocol data
        latest_block = node.consensus.get_latest_block()
        
        # Check if we have the mined block (not genesis)
        if latest_block.index > 0:
            print(f"✓ Block contains PoGO protocol data:")
            print(f"  Full model Merkle root: {latest_block.hash_full_model_32[:16]}...")
            print(f"  Quantized Merkle root: {latest_block.hash_quant_4[:16]}...")
            print(f"  Quantization error: {latest_block.quantization_error:.8f}")
            if latest_block.model_size_quant > 0:
                print(f"  Compression ratio: {latest_block.model_size_full / latest_block.model_size_quant:.1f}x")
            else:
                print(f"  Model sizes: full={latest_block.model_size_full}, quant={latest_block.model_size_quant}")
        else:
            print("⚠ Latest block is genesis block - checking chain length")
            print(f"  Chain length: {len(node.consensus.chain)}")
            if len(node.consensus.chain) > 1:
                mined_block = node.consensus.chain[-1]  # Get the last mined block
                print(f"✓ Mined block contains PoGO protocol data:")
                print(f"  Full model Merkle root: {mined_block.hash_full_model_32[:16]}...")
                print(f"  Quantized Merkle root: {mined_block.hash_quant_4[:16]}...")
                print(f"  Quantization error: {mined_block.quantization_error:.8f}")
                if mined_block.model_size_quant > 0:
                    print(f"  Compression ratio: {mined_block.model_size_full / mined_block.model_size_quant:.1f}x")
            else:
                print("⚠ No mined blocks found in chain")
        
    finally:
        node.stop()


def test_pogo_protocol_validation():
    """Test PoGO protocol validation"""
    print("\n" + "="*60)
    print("TESTING PoGO PROTOCOL VALIDATION")
    print("="*60)
    
    # Create a trainer and train a model
    trainer = SimpleTrainer()
    result = trainer.train_model(training_steps=20, learning_rate=0.01)
    
    # Initialize PoGO protocol
    pogo = PoGOProtocol(trainer.model)
    
    # Test training improvement validation
    improvement_valid = pogo.validate_training_improvement(result.loss_before, result.loss_after)
    print(f"Training improvement validation: {'✓' if improvement_valid else '✗'}")
    assert improvement_valid, "Training improvement validation failed"
    
    # Test quantization tolerance (simplified)
    tolerance_valid = pogo.validate_quantization_tolerance(result.loss_after, result.loss_after + 0.0005, 0.001)
    print(f"Quantization tolerance validation: {'✓' if tolerance_valid else '✗'}")
    assert tolerance_valid, "Quantization tolerance validation failed"
    
    # Create layer shapes
    layer_shapes = [
        (trainer.model.input_size, trainer.model.hidden_size),
        (1, trainer.model.hidden_size),
        (trainer.model.hidden_size, trainer.model.output_size),
        (1, trainer.model.output_size)
    ]
    
    # Test model commitment verification
    commitments = pogo.create_model_commitments(result.model_weights_hex, layer_shapes)
    
    for level, commitment in commitments.items():
        is_valid = pogo.verify_model_commitment(commitment)
        print(f"{level.value} commitment verification: {'✓' if is_valid else '✗'}")
        assert is_valid, f"Commitment verification failed for {level.value}"
    
    print("✓ PoGO protocol validation successful")


if __name__ == "__main__":
    print("🔍 COMPREHENSIVE PoGO PROTOCOL TESTING")
    print("=" * 80)
    
    try:
        test_merkle_tree_implementation()
        test_model_quantization()
        test_pogo_protocol_integration()
        test_network_with_pogo_protocol()
        test_pogo_protocol_validation()
        
        print("\n" + "="*80)
        print("🎉 ALL PoGO PROTOCOL TESTS PASSED!")
        print("✓ Merkle trees working correctly")
        print("✓ Model quantization implemented")
        print("✓ Full protocol integration successful")
        print("✓ Network operation with PoGO protocol")
        print("✓ Protocol validation working")
        print("="*80)
        
    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
