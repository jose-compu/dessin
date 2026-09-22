"""
Simple unit tests for BitTorrent distributor without hybrid dependencies.
"""

import unittest
import tempfile
import time
import hashlib
from pathlib import Path
import shutil

from dessin.models.model_manager import ModelManager
from dessin.distribution.bittorrent_distributor import BitTorrentDistributor, TorrentInfo
from dessin.runtime.config import ModelConfig


class TestSimpleBitTorrent(unittest.TestCase):
    """Test basic BitTorrent functionality without complex dependencies"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create model config
        model_config = ModelConfig(
            model_cache_dir=str(self.temp_dir / "models"),
            max_model_size_gb=1,
            quantization_bits=4
        )
        
        self.model_manager = ModelManager(config=model_config)
        self.distributor = BitTorrentDistributor(
            model_manager=self.model_manager,
            download_dir=str(self.temp_dir / "torrents")
        )
        
        # Create test model file
        self.test_model_path = self.temp_dir / "micro_model.gguf"
        with open(self.test_model_path, "w") as f:
            f.write("Micro model content for testing\n" * 50)  # Small model
        
        # Register test model with required ModelInfo fields
        file_size = self.test_model_path.stat().st_size
        self.model_manager.register_model(
            model_id="micro_model",
            name="Micro Test Model",
            size_gb=file_size / 1024 / 1024 / 1024,  # Convert to GB
            format="gguf",
            quantization="4bit",
            parameters=1000000,  # 1M parameters
            ipfs_hash="QmTestHash123",
            owner="test_user",
            upload_block=1,
            storage_expires=999999,
            model_hash=hashlib.sha256(self.test_model_path.read_bytes()).hexdigest()
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_torrent_info_creation(self):
        """Test TorrentInfo data class"""
        torrent_info = TorrentInfo(
            model_id="test_model",
            torrent_hash="abcdef123456",
            magnet_link="magnet:?xt=urn:btih:abcdef123456",
            file_size=1024,
            piece_count=4,
            piece_length=256,
            created_at=time.time()
        )
        
        self.assertEqual(torrent_info.model_id, "test_model")
        self.assertEqual(torrent_info.file_size, 1024)
        
        # Test serialization
        data = torrent_info.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["model_id"], "test_model")
    
    def test_create_micro_torrent(self):
        """Test creating torrent for micro model"""
        torrent_info = self.distributor.create_model_torrent(
            model_id="micro_model",
            model_path=str(self.test_model_path)
        )
        
        self.assertIsNotNone(torrent_info)
        self.assertEqual(torrent_info.model_id, "micro_model")
        self.assertGreater(torrent_info.file_size, 0)
        self.assertGreater(torrent_info.piece_count, 0)
        self.assertIsNotNone(torrent_info.torrent_hash)
        self.assertIsNotNone(torrent_info.magnet_link)
        self.assertIn("magnet:", torrent_info.magnet_link)
    
    def test_seed_micro_model(self):
        """Test seeding a micro model"""
        success = self.distributor.seed_model("micro_model", str(self.test_model_path))
        self.assertTrue(success)
        
        # Verify torrent was created
        self.assertIn("micro_model", self.distributor.model_torrents)
        
        # Check that we can get progress
        progress = self.distributor.get_download_progress("micro_model")
        self.assertIsNotNone(progress)
        self.assertEqual(progress["model_id"], "micro_model")
    
    def test_download_mock_torrent(self):
        """Test downloading with mock implementation"""
        # First create a torrent
        torrent_info = self.distributor.create_model_torrent(
            model_id="micro_model",
            model_path=str(self.test_model_path)
        )
        
        # Test download via torrent info
        success = self.distributor.download_model_torrent(torrent_info)
        self.assertTrue(success)
    
    def test_download_by_magnet_link(self):
        """Test downloading by magnet link"""
        # Create torrent first
        torrent_info = self.distributor.create_model_torrent(
            model_id="micro_model",
            model_path=str(self.test_model_path)
        )
        
        # Download using magnet link
        success = self.distributor.download_model_by_magnet(
            magnet_link=torrent_info.magnet_link,
            model_id="micro_model_copy"
        )
        self.assertTrue(success)
    
    def test_network_statistics(self):
        """Test getting network statistics"""
        # Seed a model first
        self.distributor.seed_model("micro_model", str(self.test_model_path))
        
        stats = self.distributor.get_network_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_models", stats)
        self.assertIn("active_torrents", stats)
        self.assertGreaterEqual(stats["total_models"], 1)
    
    def test_list_available_models(self):
        """Test listing available models"""
        # Seed a model first
        self.distributor.seed_model("micro_model", str(self.test_model_path))
        
        models = self.distributor.list_available_models()
        self.assertIsInstance(models, list)
        
        if models:  # Mock implementation might return empty list
            model = models[0]
            self.assertIn("model_id", model)
            self.assertIn("magnet_link", model)
    
    def test_stop_torrent(self):
        """Test stopping a torrent"""
        # Start seeding
        self.distributor.seed_model("micro_model", str(self.test_model_path))
        
        # Stop the torrent
        success = self.distributor.stop_torrent("micro_model")
        self.assertTrue(success)
        
        # Verify it's stopped
        progress = self.distributor.get_download_progress("micro_model")
        self.assertIsNone(progress)
    
    def test_cleanup_downloads(self):
        """Test cleanup of completed downloads"""
        # Seed a model
        self.distributor.seed_model("micro_model", str(self.test_model_path))
        
        # Run cleanup
        completed = self.distributor.cleanup_completed_downloads()
        self.assertIsInstance(completed, list)
    
    def test_torrent_with_custom_trackers(self):
        """Test creating torrent with custom trackers"""
        custom_trackers = [
            "udp://custom.tracker.com:8080",
            "http://another.tracker.net:6969/announce"
        ]
        
        torrent_info = self.distributor.create_model_torrent(
            model_id="micro_model",
            model_path=str(self.test_model_path),
            trackers=custom_trackers,
            comment="Test torrent with custom trackers"
        )
        
        self.assertIsNotNone(torrent_info)
        self.assertEqual(torrent_info.model_id, "micro_model")


if __name__ == "__main__":
    unittest.main()
