#!/usr/bin/env python3
"""
Comprehensive test for BitTorrent integration in PoGO protocol
Tests model file creation, torrent generation, and network sharing
"""

import unittest
import tempfile
import time
import json
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.llm.torrent_model_trainer import TorrentModelTrainer, TorrentTrainingResult
from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager
from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig


class TestTorrentIntegration(unittest.TestCase):
    """Test BitTorrent integration with PoGO protocol"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.torrent_dir = self.temp_dir / "torrents"
        self.torrent_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_torrent_trainer_basic_functionality(self):
        """Test basic torrent trainer functionality"""
        print("\n" + "="*60)
        print("TESTING TORRENT TRAINER BASIC FUNCTIONALITY")
        print("="*60)
        
        trainer = TorrentModelTrainer(str(self.torrent_dir))
        
        # Train and create torrent
        result = trainer.train_and_create_torrent(
            model_id="test_model",
            training_steps=10,
            learning_rate=0.01,
            data_seed=42
        )
        
        # Verify result type
        self.assertIsInstance(result, TorrentTrainingResult)
        
        # Verify training worked
        self.assertGreater(result.loss_before, result.loss_after)
        improvement = result.loss_before - result.loss_after
        self.assertGreater(improvement, 0.0001)
        
        # Verify torrent fields
        self.assertTrue(result.model_file_path)
        self.assertTrue(result.torrent_hash)
        self.assertTrue(result.magnet_link)
        self.assertGreater(result.model_file_size, 0)
        
        # Verify model file exists
        self.assertTrue(os.path.exists(result.model_file_path))
        
        print(f"✓ Training completed: {result.loss_before:.6f} -> {result.loss_after:.6f}")
        print(f"✓ Model file created: {result.model_file_path}")
        print(f"✓ File size: {result.model_file_size} bytes")
        print(f"✓ Torrent hash: {result.torrent_hash}")

    def test_model_file_structure(self):
        """Test the structure and content of created model files"""
        print("\n" + "="*60)
        print("TESTING MODEL FILE STRUCTURE")
        print("="*60)
        
        trainer = TorrentModelTrainer(str(self.torrent_dir))
        result = trainer.train_and_create_torrent(
            model_id="structure_test",
            training_steps=5,
            learning_rate=0.01
        )
        
        # Load and verify model file structure
        with open(result.model_file_path, 'r') as f:
            model_data = json.load(f)
        
        # Check required top-level keys
        required_keys = [
            "model_id", "timestamp", "training_metadata", 
            "model_architecture", "model_weights", "training_data"
        ]
        
        for key in required_keys:
            self.assertIn(key, model_data)
            print(f"✓ Found required key: {key}")
        
        # Check training metadata
        training_meta = model_data["training_metadata"]
        self.assertIn("loss_before", training_meta)
        self.assertIn("loss_after", training_meta)
        self.assertIn("improvement", training_meta)
        self.assertGreater(training_meta["improvement"], 0)
        
        # Check model architecture
        arch = model_data["model_architecture"]
        self.assertIn("input_size", arch)
        self.assertIn("hidden_size", arch)
        self.assertIn("output_size", arch)
        self.assertIn("layer_shapes", arch)
        
        # Check model weights
        weights = model_data["model_weights"]
        self.assertIn("weights_hex", weights)
        self.assertIn("checksum", weights)
        self.assertIn("format", weights)
        
        print(f"✓ Model structure validated")
        print(f"  Architecture: {arch['input_size']}x{arch['hidden_size']}x{arch['output_size']}")
        print(f"  Loss improvement: {training_meta['improvement']:.6f}")

    def test_model_file_integrity_verification(self):
        """Test model file integrity verification"""
        print("\n" + "="*60)
        print("TESTING MODEL FILE INTEGRITY VERIFICATION")
        print("="*60)
        
        trainer = TorrentModelTrainer(str(self.torrent_dir))
        result = trainer.train_and_create_torrent(
            model_id="integrity_test",
            training_steps=8,
            learning_rate=0.01
        )
        
        # Test valid file verification
        is_valid = trainer.verify_model_file_integrity(result.model_file_path)
        self.assertTrue(is_valid)
        print("✓ Valid model file verified successfully")
        
        # Test corrupted file
        corrupted_file = self.torrent_dir / "corrupted_model.json"
        with open(result.model_file_path, 'r') as f:
            model_data = json.load(f)
        
        # Corrupt the checksum
        model_data["model_weights"]["checksum"] = "invalid_checksum"
        
        with open(corrupted_file, 'w') as f:
            json.dump(model_data, f)
        
        is_valid = trainer.verify_model_file_integrity(str(corrupted_file))
        self.assertFalse(is_valid)
        print("✓ Corrupted model file correctly rejected")

    def test_model_loading_from_torrent_file(self):
        """Test loading model weights from torrent-shared file"""
        print("\n" + "="*60)
        print("TESTING MODEL LOADING FROM TORRENT FILE")
        print("="*60)
        
        # Create and train first model
        trainer1 = TorrentModelTrainer(str(self.torrent_dir))
        result1 = trainer1.train_and_create_torrent(
            model_id="source_model",
            training_steps=10,
            learning_rate=0.01,
            data_seed=42
        )
        
        # Create second trainer and load model from file
        trainer2 = TorrentModelTrainer(str(self.torrent_dir))
        success = trainer2.load_model_from_torrent_file(result1.model_file_path)
        self.assertTrue(success)
        
        # Verify models have same weights
        weights1_hex = trainer1.model.get_weights_bytes().hex()
        weights2_hex = trainer2.model.get_weights_bytes().hex()
        self.assertEqual(weights1_hex, weights2_hex)
        
        print("✓ Model loaded successfully from torrent file")
        print("✓ Model weights match original")

    def test_consensus_with_torrent_integration(self):
        """Test consensus integration with torrent model trainer"""
        print("\n" + "="*60)
        print("TESTING CONSENSUS WITH TORRENT INTEGRATION")
        print("="*60)
        
        # Create consensus with torrent support
        config = DessinConfig.default()
        config.model.model_cache_dir = str(self.temp_dir / "model_cache")
        
        model_manager = ModelManager(config.model)
        model_manager.register_model(
            model_id="consensus_test_model",
            name="Consensus Test Model",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash="QmConsensusTest",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash="consensus_hash"
        )
        
        consensus = PogoConsensus(
            config.consensus,
            model_manager,
            "test_miner"
        )
        
        # Create training block with torrent integration
        block = consensus.create_training_block("consensus_test_model")
        
        self.assertIsNotNone(block)
        self.assertIsInstance(block, PogoBlock)
        
        # Verify torrent fields in block
        self.assertTrue(hasattr(block, 'torrent_hash'))
        self.assertTrue(hasattr(block, 'magnet_link'))
        self.assertTrue(hasattr(block, 'model_size_bytes'))
        
        self.assertTrue(block.torrent_hash)
        self.assertTrue(block.magnet_link)
        self.assertGreater(block.model_size_bytes, 0)
        
        print(f"✓ Block created with torrent integration")
        print(f"  Torrent hash: {block.torrent_hash[:16]}...")
        print(f"  Magnet link: {block.magnet_link[:60]}...")
        print(f"  Model file size: {block.model_size_bytes} bytes")

    def test_block_serialization_with_torrent_data(self):
        """Test block serialization with torrent data"""
        print("\n" + "="*60)
        print("TESTING BLOCK SERIALIZATION WITH TORRENT DATA")
        print("="*60)
        
        # Create consensus and block
        config = DessinConfig.default()
        config.model.model_cache_dir = str(self.temp_dir / "model_cache")
        
        model_manager = ModelManager(config.model)
        model_manager.register_model(
            model_id="serialization_test",
            name="Serialization Test Model",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash="QmSerializationTest",
            owner="test_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash="serialization_hash"
        )
        
        consensus = PogoConsensus(
            config.consensus,
            model_manager,
            "test_miner"
        )
        
        block = consensus.create_training_block("serialization_test")
        self.assertIsNotNone(block)
        
        # Test serialization
        block_dict = block.to_dict()
        self.assertIsInstance(block_dict, dict)
        
        # Verify torrent fields are serializable
        self.assertIn("torrent_hash", block_dict)
        self.assertIn("magnet_link", block_dict)
        self.assertIn("model_size_bytes", block_dict)
        
        # Test JSON serialization
        json_str = json.dumps(block_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        deserialized_dict = json.loads(json_str)
        reconstructed_block = PogoBlock.from_dict(deserialized_dict)
        
        self.assertEqual(reconstructed_block.torrent_hash, block.torrent_hash)
        self.assertEqual(reconstructed_block.magnet_link, block.magnet_link)
        self.assertEqual(reconstructed_block.model_size_bytes, block.model_size_bytes)
        
        print("✓ Block serialization with torrent data successful")
        print("✓ JSON serialization/deserialization working")

    def test_multiple_model_files_creation(self):
        """Test creating multiple model files with different parameters"""
        print("\n" + "="*60)
        print("TESTING MULTIPLE MODEL FILES CREATION")
        print("="*60)
        
        trainer = TorrentModelTrainer(str(self.torrent_dir))
        
        # Create multiple models with different parameters
        models = []
        for i in range(3):
            result = trainer.train_and_create_torrent(
                model_id=f"multi_model_{i}",
                training_steps=5 + i * 2,
                learning_rate=0.01 + i * 0.005,
                data_seed=42 + i
            )
            models.append(result)
            time.sleep(0.02)  # Ensure different timestamps
        
        # Verify all models are different
        for i, model in enumerate(models):
            self.assertTrue(os.path.exists(model.model_file_path))
            self.assertTrue(model.torrent_hash)
            print(f"✓ Model {i}: {model.torrent_hash[:16]}... ({model.model_file_size} bytes)")
        
        # Verify torrent hashes are unique
        hashes = [model.torrent_hash for model in models]
        self.assertEqual(len(hashes), len(set(hashes)))
        print("✓ All torrent hashes are unique")
        
        # Verify model files have different content
        file_contents = []
        for model in models:
            with open(model.model_file_path, 'r') as f:
                content = f.read()
                file_contents.append(content)
        
        self.assertEqual(len(file_contents), len(set(file_contents)))
        print("✓ All model files have unique content")

    def test_torrent_directory_structure(self):
        """Test torrent directory structure and file organization"""
        print("\n" + "="*60)
        print("TESTING TORRENT DIRECTORY STRUCTURE")
        print("="*60)
        
        trainer = TorrentModelTrainer(str(self.torrent_dir))
        
        # Create a few models
        for i in range(2):
            trainer.train_and_create_torrent(
                model_id=f"dir_test_{i}",
                training_steps=5,
                learning_rate=0.01
            )
        
        # Check directory structure
        self.assertTrue(self.torrent_dir.exists())
        model_files = list(self.torrent_dir.glob("*.json"))
        self.assertGreaterEqual(len(model_files), 2)
        
        print(f"✓ Torrent directory created: {self.torrent_dir}")
        print(f"✓ Found {len(model_files)} model files")
        
        # Verify file naming convention
        for model_file in model_files:
            # Should be: modelid_timestamp_hash.json
            parts = model_file.stem.split('_')
            self.assertGreaterEqual(len(parts), 3)
            print(f"  - {model_file.name}")


def run_torrent_integration_tests():
    """Run all torrent integration tests"""
    print("🔗 COMPREHENSIVE TORRENT INTEGRATION TESTING")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTorrentIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL TORRENT INTEGRATION TESTS PASSED!")
        print(f"✓ {result.testsRun} tests completed successfully")
        print("✓ Model file creation working")
        print("✓ Torrent generation working")
        print("✓ File integrity verification working")
        print("✓ Model loading from torrents working")
        print("✓ Consensus integration working")
        print("✓ Block serialization with torrents working")
    else:
        print("💥 SOME TORRENT TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_torrent_integration_tests()
    sys.exit(0 if success else 1)
