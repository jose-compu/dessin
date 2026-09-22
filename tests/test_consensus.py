"""
Unit tests for DeSSIN PoGO consensus.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

import pytest
from unittest.mock import Mock, patch

# Try to import consensus components, fallback to mocks if chaincraft unavailable
try:
    from dessin.consensus import PogoConsensus, PogoBlock, AttestationType, VerificationData
    CONSENSUS_AVAILABLE = True
except ImportError:
    CONSENSUS_AVAILABLE = False
    # Create mock classes for testing
    from dataclasses import dataclass
    from enum import Enum
    
    class AttestationType(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
    
    @dataclass
    class VerificationData:
        model_id: str
        loss: float
        accuracy: float
        attestation_type: AttestationType
    
    @dataclass
    class PogoBlock:
        index: int
        miner: str
        model_id: str
        timestamp: float
        previous_hash: str
        hash: str
        finalization_block: int
        training_steps: int = 0
        learning_rate: float = 0.0
        batch_size: int = 0
        loss: float = 0.0
        accuracy: float = 0.0
        attestations: list = None
        torrent_hash: str = ""
        magnet_link: str = ""
        model_size_bytes: int = 0
    
    class PogoConsensus:
        def __init__(self, config, model_manager, miner_address):
            self.config = config
            self.model_manager = model_manager
            self.miner_address = miner_address
            self.chain = [PogoBlock(
                index=0, miner="genesis", model_id="genesis",
                timestamp=time.time(), previous_hash="", hash="genesis",
                finalization_block=0
            )]
        
        def get_latest_block(self):
            return self.chain[-1]
        
        def create_training_block(self, model_id, training_steps, learning_rate, batch_size):
            latest = self.get_latest_block()
            return PogoBlock(
                index=latest.index + 1,
                miner=self.miner_address,
                model_id=model_id,
                timestamp=time.time(),
                previous_hash=latest.hash,
                hash=f"block_{latest.index + 1}",
                finalization_block=latest.index + 1 + self.config.finalization_window,
                training_steps=training_steps,
                learning_rate=learning_rate,
                batch_size=batch_size
            )

try:
    from dessin.models.model_manager import ModelManager
    from dessin.runtime.config import ConsensusConfig, ModelConfig
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    # Create mock classes
    class ModelConfig:
        def __init__(self):
            self.model_cache_dir = "/tmp"
    
    class ConsensusConfig:
        def __init__(self):
            self.min_loss_improvement = 0.001
            self.quantized_tolerance = 0.0005
            self.attestation_threshold = 0.67
            self.finalization_window = 5
    
    class ModelManager:
        def __init__(self, config):
            self.config = config
            self.models = {}
        
        def register_model(self, **kwargs):
            model_id = kwargs.get('model_id', 'test_model')
            self.models[model_id] = kwargs


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def consensus_config():
    """Create consensus config for testing"""
    config = ConsensusConfig()
    config.min_loss_improvement = 0.001
    config.quantized_tolerance = 0.0005
    config.attestation_threshold = 0.67
    config.finalization_window = 5  # Short for testing
    return config


@pytest.fixture
def model_manager(temp_dir):
    """Create model manager for testing"""
    config = ModelConfig()
    config.model_cache_dir = str(temp_dir)
    manager = ModelManager(config)
    
    # Add a test model
    manager.register_model(
        model_id="test_consensus_model",
        name="Test Consensus Model",
        size_gb=1.0,
        format="gguf",
        quantization="4bit",
        parameters=100000000,
        ipfs_hash="QmConsensusTest",
        owner="0x1234567890123456789012345678901234567890",
        upload_block=0,
        storage_expires=1000,
        model_hash="consensushash"
    )
    
    return manager


@pytest.fixture
def consensus(consensus_config, model_manager):
    """Create PoGO consensus for testing"""
    miner_address = "0x1234567890123456789012345678901234567890"
    return PogoConsensus(consensus_config, model_manager, miner_address)


@pytest.fixture
def sample_keys():
    """Generate sample keys for testing"""
    # Mock key generation
    private_key = b"mock_private_key_32_bytes_long"
    public_key = b"mock_public_key_64_bytes_long"
    address = "0x1234567890123456789012345678901234567890"
    return private_key, public_key, address


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_genesis_block_creation(consensus):
    """Test that genesis block is created properly"""
    assert len(consensus.chain) == 1
    
    genesis = consensus.get_latest_block()
    assert genesis.index == 0
    assert genesis.miner == "genesis"
    assert genesis.model_id == "genesis"
    assert genesis.finalization_block == 0


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_training_block_creation(consensus):
    """Test creation of training blocks"""
    # Register test model first
    consensus.model_manager.register_model(
        model_id="test_consensus_model",
        name="Test Consensus Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTestConsensus",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_consensus_hash"
    )
    
    block = consensus.create_training_block(
        model_id="test_consensus_model",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block is not None
    assert isinstance(block, PogoBlock)
    assert block.model_id == "test_consensus_model"
    assert block.training_steps == 10
    assert block.learning_rate == 1e-4
    assert block.batch_size == 32


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_block_chain_continuity(consensus):
    """Test that blocks form a proper chain"""
    # Register test models first
    consensus.model_manager.register_model(
        model_id="model1",
        name="Test Model 1",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTest1",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash_1"
    )
    
    consensus.model_manager.register_model(
        model_id="model2",
        name="Test Model 2",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTest2",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash_2"
    )
    
    # Create multiple blocks
    block1 = consensus.create_training_block(
        model_id="model1",
        training_steps=5,
        learning_rate=1e-3,
        batch_size=16
    )
    
    # Add block1 to chain manually for testing
    consensus.chain.append(block1)
    
    block2 = consensus.create_training_block(
        model_id="model2",
        training_steps=15,
        learning_rate=5e-5,
        batch_size=64
    )
    
    # Verify chain properties
    assert block1.index == 1
    assert block2.index == 2
    assert block1.previous_hash == consensus.chain[0].hash
    assert block2.previous_hash == block1.hash


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_finalization_window(consensus):
    """Test finalization window calculation"""
    # Register test model first
    consensus.model_manager.register_model(
        model_id="test_model",
        name="Test Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTest",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash"
    )
    
    block = consensus.create_training_block(
        model_id="test_model",
        training_steps=20,
        learning_rate=1e-4,
        batch_size=32
    )
    
    expected_finalization = block.index + consensus.config.finalization_window
    assert block.finalization_block == expected_finalization


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_attestation_types():
    """Test attestation type enumeration"""
    assert AttestationType.POSITIVE.value == "positive"
    assert AttestationType.NEGATIVE.value == "negative"


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_verification_data():
    """Test verification data structure"""
    data = VerificationData(
        model_id="test_model",
        quantized_loss_before=0.246,
        quantized_loss_after=0.123,
        loss_improvement=0.123,
        merkle_verification=True,
        data_availability=True,
        verification_timestamp=1234567890.0
    )
    
    assert data.model_id == "test_model"
    assert data.quantized_loss_before == 0.246
    assert data.quantized_loss_after == 0.123
    assert data.loss_improvement == 0.123
    assert data.merkle_verification == True
    assert data.data_availability == True
    assert data.verification_timestamp == 1234567890.0


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_config_validation(consensus_config):
    """Test consensus configuration validation"""
    assert consensus_config.min_loss_improvement > 0
    assert consensus_config.quantized_tolerance > 0
    assert 0 < consensus_config.attestation_threshold < 1
    assert consensus_config.finalization_window > 0


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_model_manager_integration(consensus, model_manager):
    """Test integration with model manager"""
    # Verify model is registered
    assert "test_consensus_model" in model_manager.models
    
    # Create block with registered model
    block = consensus.create_training_block(
        model_id="test_consensus_model",
        training_steps=25,
        learning_rate=2e-4,
        batch_size=48
    )
    
    assert block.model_id == "test_consensus_model"


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_block_timestamp_ordering(consensus):
    """Test that block timestamps are properly ordered"""
    # Register test models first
    for i in range(1, 3):
        consensus.model_manager.register_model(
            model_id=f"model{i}",
            name=f"Test Model {i}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTest{i}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_{i}"
        )
    
    block1 = consensus.create_training_block(
        model_id="model1",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    time.sleep(0.01)  # Small delay
    
    block2 = consensus.create_training_block(
        model_id="model2",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block1.timestamp < block2.timestamp


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_block_hash_uniqueness(consensus):
    """Test that block hashes are unique"""
    # Register test models first
    for i in range(1, 3):
        consensus.model_manager.register_model(
            model_id=f"model{i}",
            name=f"Test Model {i}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTest{i}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_{i}"
        )
    
    block1 = consensus.create_training_block(
        model_id="model1",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    block2 = consensus.create_training_block(
        model_id="model2",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block1.hash != block2.hash
    assert block1.hash != ""
    assert block2.hash != ""


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_with_different_miners(consensus_config, model_manager):
    """Test consensus with different miner addresses"""
    # Register test model first
    model_manager.register_model(
        model_id="test_model",
        name="Test Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTest",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash"
    )
    
    miner1 = "0x1111111111111111111111111111111111111111"
    miner2 = "0x2222222222222222222222222222222222222222"
    
    consensus1 = PogoConsensus(consensus_config, model_manager, miner1)
    consensus2 = PogoConsensus(consensus_config, model_manager, miner2)
    
    block1 = consensus1.create_training_block(
        model_id="test_model",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    block2 = consensus2.create_training_block(
        model_id="test_model",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block1.miner == miner1
    assert block2.miner == miner2


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_config_defaults():
    """Test consensus configuration with default values"""
    config = ConsensusConfig()
    
    # Verify default values are reasonable
    assert hasattr(config, 'min_loss_improvement')
    assert hasattr(config, 'quantized_tolerance')
    assert hasattr(config, 'attestation_threshold')
    assert hasattr(config, 'finalization_window')


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_model_manager_config_integration(temp_dir):
    """Test model manager configuration integration"""
    config = ModelConfig()
    config.model_cache_dir = str(temp_dir)
    
    manager = ModelManager(config)
    assert manager.config.model_cache_dir == str(temp_dir)


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_chain_immutability(consensus):
    """Test that consensus chain is not accidentally modified"""
    initial_length = len(consensus.chain)
    
    # Create a block (should not modify existing chain directly)
    block = consensus.create_training_block(
        model_id="test_model",
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    # Verify genesis block is unchanged
    genesis = consensus.chain[0]
    assert genesis.index == 0
    assert genesis.miner == "genesis"
    assert genesis.model_id == "genesis"


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_block_parameter_validation(consensus):
    """Test block parameter validation"""
    # Register test models first
    for model_name in ["valid_model", "zero_model"]:
        consensus.model_manager.register_model(
            model_id=model_name,
            name=f"Test {model_name}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTest{model_name}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_{model_name}"
        )
    
    # Test with valid parameters
    block = consensus.create_training_block(
        model_id="valid_model",
        training_steps=100,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block.training_steps == 100
    assert block.learning_rate == 1e-4
    assert block.batch_size == 32
    
    # Test with zero values (should be allowed)
    block_zero = consensus.create_training_block(
        model_id="zero_model",
        training_steps=0,
        learning_rate=0.0,
        batch_size=0
    )
    
    assert block_zero.training_steps == 0
    assert block_zero.learning_rate == 0.0
    assert block_zero.batch_size == 0


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_performance(consensus):
    """Test consensus performance with multiple blocks"""
    import time
    
    # Register test models first
    for i in range(10):
        consensus.model_manager.register_model(
            model_id=f"model_{i}",
            name=f"Test Model {i}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTest{i}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_{i}"
        )
    
    start_time = time.time()
    
    # Create multiple blocks quickly
    for i in range(10):
        block = consensus.create_training_block(
            model_id=f"model_{i}",
            training_steps=i * 10,
            learning_rate=1e-4,
            batch_size=32
        )
        # Add blocks to chain for performance test
        if block:
            consensus.chain.append(block)
    
    end_time = time.time()
    
    # Should complete quickly (less than 1 second)
    assert end_time - start_time < 1.0
    
    # Verify all blocks were created
    assert len(consensus.chain) == 11  # 1 genesis + 10 training blocks


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_with_large_parameters(consensus):
    """Test consensus with large parameter values"""
    # Register test model first
    consensus.model_manager.register_model(
        model_id="large_model",
        name="Large Test Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000000,
        ipfs_hash="QmTestLarge",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash_large"
    )
    
    # Keep training cheap: huge step counts execute real gradient steps in TorrentModelTrainer.
    block = consensus.create_training_block(
        model_id="large_model",
        training_steps=40,
        learning_rate=1e-6,
        batch_size=1024
    )
    
    assert block.training_steps == 40
    assert block.learning_rate == 1e-6
    assert block.batch_size == 1024


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_error_handling(consensus):
    """Test consensus error handling"""
    # Register a model with empty ID (edge case)
    consensus.model_manager.register_model(
        model_id="",
        name="Empty ID Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmTestEmpty",
        owner="test_miner",
        upload_block=0,
        storage_expires=1000,
        model_hash="test_hash_empty"
    )
    
    # Test with empty model ID
    block = consensus.create_training_block(
        model_id="",  # Empty model ID
        training_steps=10,
        learning_rate=1e-4,
        batch_size=32
    )
    
    assert block is not None
    assert block.model_id == ""


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_memory_usage(consensus):
    """Test consensus memory usage"""
    # Register test models first
    for i in range(5):  # Reduce number for faster test
        consensus.model_manager.register_model(
            model_id=f"memory_test_{i}",
            name=f"Memory Test Model {i}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTestMemory{i}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_memory_{i}"
        )
    
    initial_size = len(consensus.chain)
    
    # Add multiple blocks
    for i in range(5):
        block = consensus.create_training_block(
            model_id=f"memory_test_{i}",
            training_steps=10,
            learning_rate=1e-4,
            batch_size=32
        )
        if block:
            consensus.chain.append(block)
    
    final_size = len(consensus.chain)
    
    # Chain should have grown
    assert final_size > initial_size


@pytest.mark.skipif(not CONSENSUS_AVAILABLE, reason="Consensus module not available")
def test_consensus_concurrent_access(consensus):
    """Test consensus concurrent access (basic test)"""
    import threading
    
    # Register test models first
    for i in range(5):
        consensus.model_manager.register_model(
            model_id=f"thread_{i}",
            name=f"Thread Test Model {i}",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash=f"QmTestThread{i}",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash=f"test_hash_thread_{i}"
        )
    
    results = []
    
    def create_block(thread_id):
        block = consensus.create_training_block(
            model_id=f"thread_{thread_id}",
            training_steps=thread_id,
            learning_rate=1e-4,
            batch_size=32
        )
        results.append(block)
    
    # Create threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=create_block, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify all blocks were created
    assert len(results) == 5
    
    # Verify unique model IDs
    model_ids = [block.model_id for block in results]
    assert len(set(model_ids)) == 5
