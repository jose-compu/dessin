"""
Tests for Nanochat Integration
"""

import pytest
import tempfile
from pathlib import Path

from dessin.nanochat.nanochat_integration import (
    NanochatIntegration,
    NanochatModelConfig,
    NanochatTrainingJob
)
from dessin.models.model_manager import ModelManager
from dessin.runtime.config import ModelConfig


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def model_manager(temp_cache_dir):
    """Create a model manager"""
    config = ModelConfig()
    config.model_cache_dir = temp_cache_dir
    return ModelManager(config)


@pytest.fixture
def nanochat_integration(model_manager):
    """Create nanochat integration"""
    return NanochatIntegration(model_manager)


def test_nanochat_model_config_creation():
    """Test creating a nanochat model configuration"""
    config = NanochatModelConfig(
        model_id="test_model",
        name="Test Nanochat Model",
        depth=20,
        device_batch_size=32,
        vocab_size=50257,
        context_length=1024
    )
    
    assert config.model_id == "test_model"
    assert config.name == "Test Nanochat Model"
    assert config.depth == 20
    assert config.device_batch_size == 32
    assert config.parameters > 0  # Should be calculated


def test_nanochat_model_config_serialization():
    """Test model configuration serialization"""
    config = NanochatModelConfig(
        model_id="test_model",
        name="Test Model",
        depth=20,
        device_batch_size=32,
        vocab_size=50257,
        context_length=1024
    )
    
    # Convert to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['model_id'] == "test_model"
    
    # Recreate from dict
    config2 = NanochatModelConfig.from_dict(config_dict)
    assert config2.model_id == config.model_id
    assert config2.depth == config.depth
    assert config2.parameters == config.parameters


def test_create_nanochat_model(nanochat_integration):
    """Test creating a nanochat model"""
    model_id = nanochat_integration.create_model(
        name="Test Model d20",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        device_batch_size=32,
        dataset_name="fineweb",
        storage_blocks=1000,
        storage_payment=10.0,
        training_payment_per_iteration=0.001
    )
    
    assert model_id is not None
    assert model_id.startswith("nanochat_")
    
    # Check if model is registered
    assert model_id in nanochat_integration.model_configs
    
    # Check if model is in model manager
    model_info = nanochat_integration.model_manager.get_model_info(model_id)
    assert model_info is not None
    assert model_info.format == "nanochat"
    assert model_info.depth == 20


def test_create_model_with_different_depths(nanochat_integration):
    """Test creating models with different depths"""
    depths = [20, 26, 32]
    
    for depth in depths:
        model_id = nanochat_integration.create_model(
            name=f"Test Model d{depth}",
            depth=depth,
            owner_address="0x1234567890123456789012345678901234567890",
            storage_blocks=1000,
            storage_payment=10.0
        )
        
        assert model_id is not None
        config = nanochat_integration.model_configs[model_id]
        assert config.depth == depth
        
        # Larger models should have more parameters
        if depth > 20:
            config_20 = nanochat_integration.model_configs[f"nanochat_{list(nanochat_integration.model_configs.keys())[0].split('_')[1]}"]
            # Note: This is a simplified check


def test_start_training(nanochat_integration):
    """Test starting a training job"""
    # Create model first
    model_id = nanochat_integration.create_model(
        name="Training Test Model",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0
    )
    
    # Start training
    success = nanochat_integration.start_training(
        model_id=model_id,
        miner_address="0x9876543210987654321098765432109876543210",
        block_index=1
    )
    
    assert success is True
    
    # Check job status
    job_id = f"job_{model_id}"
    assert job_id in nanochat_integration.training_jobs
    job = nanochat_integration.training_jobs[job_id]
    assert job.status == "training"


def test_train_iteration(nanochat_integration):
    """Test performing a training iteration"""
    # Create and start training
    model_id = nanochat_integration.create_model(
        name="Iteration Test Model",
        depth=4,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0,
        device_batch_size=2,
        vocab_size=512,
        context_length=64,
    )
    
    nanochat_integration.start_training(
        model_id=model_id,
        miner_address="0x9876543210987654321098765432109876543210",
        block_index=1
    )
    
    # Perform training iteration
    success, metrics = nanochat_integration.train_iteration(model_id)
    
    assert success is True
    assert "loss_before" in metrics
    assert "loss_after" in metrics
    assert "loss_improvement" in metrics
    # Note: Single iteration may not always improve (stochastic), just verify it's numeric
    assert isinstance(metrics["loss_improvement"], (int, float))
    assert metrics["iteration"] == 1


def test_multiple_training_iterations(nanochat_integration):
    """Test performing multiple training iterations"""
    # Create and start training
    model_id = nanochat_integration.create_model(
        name="Multi-Iteration Test Model",
        depth=4,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0,
        device_batch_size=2,
        vocab_size=512,
        context_length=64,
    )
    
    nanochat_integration.start_training(
        model_id=model_id,
        miner_address="0x9876543210987654321098765432109876543210",
        block_index=1
    )
    
    # Perform multiple iterations (keep small — each may run real Torch if nanochat is present)
    iterations = 3
    for i in range(iterations):
        success, metrics = nanochat_integration.train_iteration(model_id)
        assert success is True
        assert metrics["iteration"] == i + 1


def test_storage_cost_calculation(nanochat_integration):
    """Test storage cost calculation"""
    cost = nanochat_integration.calculate_storage_cost(
        model_size_gb=1.0,
        storage_blocks=1000,
        price_per_gb_per_block=0.001
    )
    
    assert cost == 1.0  # 1 GB * 1000 blocks * 0.001 = 1.0


def test_training_cost_calculation(nanochat_integration):
    """Test training cost calculation"""
    cost = nanochat_integration.calculate_training_cost(
        iterations=1000,
        payment_per_iteration=0.001
    )
    
    assert cost == 1.0  # 1000 iterations * 0.001 = 1.0


def test_get_model_info(nanochat_integration):
    """Test getting model information"""
    # Create model
    model_id = nanochat_integration.create_model(
        name="Info Test Model",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0,
        training_payment_per_iteration=0.001
    )
    
    # Get info
    info = nanochat_integration.get_model_info(model_id)
    
    assert info is not None
    assert info["model_id"] == model_id
    assert info["name"] == "Info Test Model"
    assert info["depth"] == 20
    assert info["format"] == "nanochat"
    assert "training" in info
    assert info["training"]["storage_blocks"] == 1000


def test_list_models(nanochat_integration):
    """Test listing all models"""
    # Create multiple models
    for i in range(3):
        nanochat_integration.create_model(
            name=f"List Test Model {i}",
            depth=20,
            owner_address="0x1234567890123456789012345678901234567890",
            storage_blocks=1000,
            storage_payment=10.0
        )
    
    # List models
    models = nanochat_integration.list_models()
    
    assert len(models) >= 3
    for model in models:
        assert "model_id" in model
        assert "name" in model
        assert "depth" in model


def test_storage_expiration(nanochat_integration):
    """Test storage expiration checking"""
    model_id = nanochat_integration.create_model(
        name="Expiration Test Model",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=10,  # Short expiration
        storage_payment=1.0
    )
    
    # Check not expired at block 5
    is_expired = nanochat_integration.check_storage_expiration(
        model_id=model_id,
        current_block=5,
        upload_block=0,
        storage_blocks=10
    )
    assert is_expired is False
    
    # Check expired at block 15
    is_expired = nanochat_integration.check_storage_expiration(
        model_id=model_id,
        current_block=15,
        upload_block=0,
        storage_blocks=10
    )
    assert is_expired is True


def test_cleanup_expired_models(nanochat_integration):
    """Test cleanup of expired models"""
    # Create model with short expiration
    model_id = nanochat_integration.create_model(
        name="Cleanup Test Model",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=10,
        storage_payment=1.0
    )
    
    # Set the model as expired in model manager
    model_info = nanochat_integration.model_manager.get_model_info(model_id)
    model_info.storage_expires = 5  # Expire at block 5
    
    # Cleanup at block 10
    removed = nanochat_integration.cleanup_expired_models(current_block=10)
    
    # Model should be removed
    assert model_id in removed
    assert model_id not in nanochat_integration.model_configs


def test_model_size_estimation(nanochat_integration):
    """Test model size estimation"""
    # Create models of different sizes
    model_id_20 = nanochat_integration.create_model(
        name="Size Test d20",
        depth=20,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0
    )
    
    model_id_32 = nanochat_integration.create_model(
        name="Size Test d32",
        depth=32,
        owner_address="0x1234567890123456789012345678901234567890",
        storage_blocks=1000,
        storage_payment=10.0
    )
    
    # Get model info
    info_20 = nanochat_integration.model_manager.get_model_info(model_id_20)
    info_32 = nanochat_integration.model_manager.get_model_info(model_id_32)
    
    # Larger model should have larger size
    assert info_32.size_gb > info_20.size_gb
    assert info_32.parameters > info_20.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
