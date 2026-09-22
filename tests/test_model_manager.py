"""
Unit tests for DeSSIN model manager.
"""

import sys
import os
import tempfile
import hashlib
from pathlib import Path

# Add chaincraft to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chaincraft"))

import pytest
from dessin.models.model_manager import ModelManager, ModelInfo, ModelQueryResult
from dessin.runtime.config import ModelConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def model_manager(temp_dir):
    """Create model manager for testing"""
    config = ModelConfig()
    config.model_cache_dir = str(temp_dir)
    return ModelManager(config)


@pytest.fixture
def sample_model_file(temp_dir):
    """Create a sample model file for testing"""
    model_path = temp_dir / "test_model.gguf"
    
    # Create a small test file
    test_data = b"GGUF test model data" * 1000  # ~20KB
    with open(model_path, "wb") as f:
        f.write(test_data)
    
    return model_path


def test_model_registration(model_manager):
    """Test model registration"""
    success = model_manager.register_model(
        model_id="test_model_1",
        name="Test Model 1",
        size_gb=1.5,
        format="gguf",
        quantization="4bit",
        parameters=124000000,
        ipfs_hash="QmTest123",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=100,
        storage_expires=1100,
        model_hash="hash123"
    )
    
    assert success
    assert "test_model_1" in model_manager.models
    
    model_info = model_manager.get_model_info("test_model_1")
    assert model_info is not None
    assert model_info.name == "Test Model 1"
    assert model_info.size_gb == 1.5
    assert model_info.format == "gguf"
    assert model_info.quantization == "4bit"
    assert model_info.parameters == 124000000


def test_model_hash_calculation(model_manager, sample_model_file):
    """Test model hash calculation"""
    calculated_hash = model_manager.get_model_hash(str(sample_model_file))
    
    # Verify it's a valid SHA-256 hash
    assert len(calculated_hash) == 64
    assert all(c in "0123456789abcdef" for c in calculated_hash)
    
    # Calculate expected hash
    with open(sample_model_file, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    
    assert calculated_hash == expected_hash


def test_model_size_estimation(model_manager, sample_model_file):
    """Test model size estimation"""
    size_gb = model_manager.estimate_model_size(str(sample_model_file))
    
    # File is ~20KB, so should be much less than 1GB
    assert 0 < size_gb < 0.001  # Less than 1MB


def test_merkle_tree_creation(model_manager, sample_model_file):
    """Test Merkle tree creation and verification"""
    merkle_root, leaves = model_manager.create_merkle_tree(str(sample_model_file), leaf_size_mb=0.01)  # 10KB leaves
    
    assert merkle_root is not None
    assert len(merkle_root) == 64  # SHA-256 hash
    assert len(leaves) >= 1  # Should have at least one leaf
    
    # For a ~20KB file with 10KB leaves, should have 2 leaves
    assert len(leaves) == 2


def test_merkle_proof_verification(model_manager):
    """Test Merkle proof verification"""
    # Simple test with known values
    leaf_hash = "abc123"
    proof = ["def456"]
    root = hashlib.sha256((leaf_hash + proof[0]).encode()).hexdigest()
    
    # This should pass
    assert model_manager.verify_merkle_proof(leaf_hash, 0, proof, root)
    
    # This should fail with wrong root
    assert not model_manager.verify_merkle_proof(leaf_hash, 0, proof, "wrong_root")


def test_model_availability(model_manager):
    """Test model availability checking"""
    # Register a model
    model_manager.register_model(
        model_id="test_model_2",
        name="Test Model 2",
        size_gb=1.0,
        format="gguf",
        quantization="4bit",
        parameters=100000000,
        ipfs_hash="QmTest456",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=100,
        storage_expires=200,
        model_hash="hash456"
    )
    
    # Should be available before expiry
    assert model_manager.is_model_available("test_model_2", 150)
    
    # Should not be available after expiry
    assert not model_manager.is_model_available("test_model_2", 250)
    
    # Non-existent model should not be available
    assert not model_manager.is_model_available("non_existent", 150)


def test_model_cleanup(model_manager, temp_dir):
    """Test cleanup of expired models"""
    # Register some models
    model_manager.register_model(
        model_id="model_1",
        name="Model 1",
        size_gb=1.0,
        format="gguf",
        quantization="4bit",
        parameters=100000000,
        ipfs_hash="QmTest1",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=100,
        storage_expires=150,  # Expires early
        model_hash="hash1"
    )
    
    model_manager.register_model(
        model_id="model_2", 
        name="Model 2",
        size_gb=1.0,
        format="gguf",
        quantization="4bit",
        parameters=100000000,
        ipfs_hash="QmTest2",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=100,
        storage_expires=250,  # Expires later
        model_hash="hash2"
    )
    
    # Create dummy cache files
    (temp_dir / "model_1.gguf").touch()
    (temp_dir / "model_2.gguf").touch()
    
    # Both should exist initially
    assert "model_1" in model_manager.models
    assert "model_2" in model_manager.models
    
    # Clean up at block 200 (model_1 should be removed)
    removed = model_manager.cleanup_expired_models(200)
    
    assert "model_1" in removed
    assert "model_2" not in removed
    assert "model_1" not in model_manager.models
    assert "model_2" in model_manager.models


def test_model_query_simulation(model_manager):
    """Test model query simulation"""
    # Register a test model
    model_manager.register_model(
        model_id="query_test_model",
        name="Query Test Model",
        size_gb=0.5,
        format="gguf",
        quantization="4bit",
        parameters=50000000,
        ipfs_hash="QmQueryTest",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=100,
        storage_expires=1000,
        model_hash="queryhash"
    )
    
    # Create a dummy model file for the test
    model_path = Path(model_manager.config.model_cache_dir) / "query_test_model.gguf"
    model_path.touch()
    
    # Query the model (should simulate response since model isn't actually loaded)
    result = model_manager.query_model("query_test_model", "Hello, world!", max_tokens=50)
    
    # Should get a simulated response
    assert isinstance(result, ModelQueryResult)
    assert result.query == "Hello, world!"
    assert result.tokens_used > 0
    assert result.compute_cost >= 0
    
    # Query non-existent model should fail
    result_fail = model_manager.query_model("non_existent", "test")
    assert not result_fail.success
    assert result_fail.error_message is not None


def test_model_list_operations(model_manager):
    """Test model listing operations"""
    # Initially empty
    assert len(model_manager.list_models()) == 0
    
    # Add some models
    for i in range(3):
        model_manager.register_model(
            model_id=f"list_test_model_{i}",
            name=f"List Test Model {i}",
            size_gb=float(i + 1),
            format="gguf",
            quantization="4bit",
            parameters=(i + 1) * 1000000,
            ipfs_hash=f"QmListTest{i}",
            owner="0x1234567890123456789012345678901234567890",
            upload_block=100,
            storage_expires=1000,
            model_hash=f"listhash{i}"
        )
    
    models = model_manager.list_models()
    assert len(models) == 3
    
    # Check model info retrieval
    model_0 = model_manager.get_model_info("list_test_model_0")
    assert model_0 is not None
    assert model_0.name == "List Test Model 0"
    assert model_0.size_gb == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
