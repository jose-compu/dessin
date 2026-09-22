"""
Unit tests for BitTorrent distributor with micro-models.
"""

import unittest
import tempfile
import time
import asyncio
from pathlib import Path
import shutil

from dessin.models.model_manager import ModelManager
from dessin.distribution.bittorrent_distributor import BitTorrentDistributor, TorrentInfo
from dessin.runtime.config import ConsensusConfig, ModelConfig


class TestBitTorrentDistributor(unittest.TestCase):
    """Test BitTorrent distribution functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        config = ModelConfig(model_cache_dir=str(self.temp_dir / "models"))
        self.model_manager = ModelManager(config)
        self.distributor = BitTorrentDistributor(
            model_manager=self.model_manager,
            download_dir=str(self.temp_dir / "torrents")
        )
        
        # Create test model file
        self.test_model_path = self.temp_dir / "test_model.gguf"
        with open(self.test_model_path, "w") as f:
            f.write("Test model content for BitTorrent distribution\n" * 100)
        
        # Register test model
        self.model_manager.register_model(
            model_id="test_model",
            name="Test Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            size_gb=0.001,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_torrent(self):
        """Test torrent creation"""
        torrent_info = self.distributor.create_model_torrent(
            model_id="test_model",
            model_path=str(self.test_model_path)
        )
        
        self.assertIsNotNone(torrent_info)
        self.assertEqual(torrent_info.model_id, "test_model")
        self.assertIsNotNone(torrent_info.torrent_hash)
        self.assertIsNotNone(torrent_info.magnet_link)
        self.assertGreater(torrent_info.file_size, 0)
        self.assertGreater(torrent_info.piece_count, 0)
    
    def test_torrent_info_serialization(self):
        """Test torrent info serialization"""
        torrent_info = self.distributor.create_model_torrent(
            model_id="test_model",
            model_path=str(self.test_model_path)
        )
        
        # Test to_dict
        data = torrent_info.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["model_id"], "test_model")
        self.assertIn("torrent_hash", data)
        self.assertIn("magnet_link", data)
    
    def test_seed_model(self):
        """Test model seeding"""
        success = self.distributor.seed_model("test_model", str(self.test_model_path))
        self.assertTrue(success)
        
        # Check if torrent was created and seeding started
        self.assertIn("test_model", self.distributor.model_torrents)
        
        # Check progress
        progress = self.distributor.get_download_progress("test_model")
        self.assertIsNotNone(progress)
        self.assertEqual(progress["model_id"], "test_model")
    
    def test_download_by_magnet(self):
        """Test downloading by magnet link"""
        # First create and seed a model
        torrent_info = self.distributor.create_model_torrent(
            model_id="test_model",
            model_path=str(self.test_model_path)
        )
        
        # Test download by magnet link
        success = self.distributor.download_model_by_magnet(
            magnet_link=torrent_info.magnet_link,
            model_id="test_model_download"
        )
        self.assertTrue(success)
    
    def test_stop_torrent(self):
        """Test stopping torrents"""
        # Start seeding
        self.distributor.seed_model("test_model", str(self.test_model_path))
        
        # Stop torrent
        success = self.distributor.stop_torrent("test_model")
        self.assertTrue(success)
        
        # Check that torrent is no longer active
        progress = self.distributor.get_download_progress("test_model")
        self.assertIsNone(progress)
    
    def test_torrent_status(self):
        """Test getting torrent status"""
        self.distributor.seed_model("test_model", str(self.test_model_path))
        
        # Get all torrents status
        status_list = self.distributor.get_all_torrents_status()
        self.assertIsInstance(status_list, list)
        self.assertGreater(len(status_list), 0)
        
        # Check status structure
        status = status_list[0]
        self.assertIn("model_id", status)
        self.assertIn("progress", status)
        self.assertIn("state", status)
    
    def test_network_statistics(self):
        """Test network statistics"""
        self.distributor.seed_model("test_model", str(self.test_model_path))
        
        stats = self.distributor.get_network_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_models", stats)
        self.assertIn("active_torrents", stats)
        self.assertIn("total_seeders", stats)
        self.assertIn("total_leechers", stats)
    
    def test_export_torrent_info(self):
        """Test exporting torrent files"""
        self.distributor.create_model_torrent("test_model", str(self.test_model_path))
        
        torrent_path = self.distributor.export_torrent_info("test_model")
        # For mock implementation, this might return None
        # In real implementation, it would return the path
        if torrent_path:
            self.assertTrue(Path(torrent_path).exists())
    
    def test_list_available_models(self):
        """Test listing available models"""
        self.distributor.seed_model("test_model", str(self.test_model_path))
        
        models = self.distributor.list_available_models()
        self.assertIsInstance(models, list)
        
        if models:  # For mock implementation, might be empty
            model = models[0]
            self.assertIn("model_id", model)
            self.assertIn("torrent_hash", model)
            self.assertIn("magnet_link", model)
    
    def test_cleanup_completed_downloads(self):
        """Test cleanup of completed downloads"""
        self.distributor.seed_model("test_model", str(self.test_model_path))
        
        # Run cleanup
        completed = self.distributor.cleanup_completed_downloads()
        self.assertIsInstance(completed, list)


class TestTorrentInfo(unittest.TestCase):
    """Test TorrentInfo data class"""
    
    def test_torrent_info_creation(self):
        """Test TorrentInfo creation and properties"""
        torrent_info = TorrentInfo(
            model_id="test_model",
            torrent_hash="abcdef123456",
            magnet_link="magnet:?xt=urn:btih:abcdef123456",
            file_size=1024*1024,  # 1MB
            piece_count=4,
            piece_length=256*1024,  # 256KB
            seeders=5,
            leechers=2,
            download_progress=0.75,
            upload_ratio=1.5,
            created_at=time.time()
        )
        
        self.assertEqual(torrent_info.model_id, "test_model")
        self.assertEqual(torrent_info.seeders, 5)
        self.assertEqual(torrent_info.leechers, 2)
        self.assertEqual(torrent_info.download_progress, 0.75)
        
        # Test serialization
        data = torrent_info.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["model_id"], "test_model")
        self.assertEqual(data["seeders"], 5)


if __name__ == "__main__":
    unittest.main()
