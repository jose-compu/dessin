"""
Integration tests for DeSSIN BitTorrent distribution system.
Tests the interaction between multiple components.
"""

import unittest
import tempfile
import time
from pathlib import Path
import shutil
import hashlib

from dessin.models.model_manager import ModelManager
from dessin.distribution.bittorrent_distributor import BitTorrentDistributor
from dessin.runtime.progress_tracker import get_progress_tracker, ProgressType
from dessin.distribution.huggingface_distributor import HuggingFaceDistributor
from dessin.runtime.config import ModelConfig


def create_test_model(name: str, size_kb: int = 10) -> str:
    """Create a test model file"""
    temp_dir = Path(tempfile.gettempdir()) / "dessin_integration_test"
    temp_dir.mkdir(exist_ok=True)
    
    model_path = temp_dir / f"{name}.gguf"
    content = f"Test model: {name}\n" + "x" * (size_kb * 1024 - len(f"Test model: {name}\n"))
    
    with open(model_path, "w") as f:
        f.write(content)
    
    return str(model_path)


class TestModelManagerIntegration(unittest.TestCase):
    """Test integration between model manager and distributors"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.model_config = ModelConfig(
            model_cache_dir=str(self.temp_dir / "models"),
            max_model_size_gb=1
        )
        
        self.model_manager = ModelManager(config=self.model_config)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Cleanup integration test dir
        integration_dir = Path(tempfile.gettempdir()) / "dessin_integration_test"
        if integration_dir.exists():
            shutil.rmtree(integration_dir)
    
    def test_model_manager_bittorrent_integration(self):
        """Test integration between model manager and BitTorrent distributor"""
        # Create BitTorrent distributor
        bt_distributor = BitTorrentDistributor(
            model_manager=self.model_manager,
            download_dir=str(self.temp_dir / "torrents")
        )
        
        # Create test model
        model_path = create_test_model("integration_test_model", 20)
        
        # Register model
        file_size = Path(model_path).stat().st_size
        success = self.model_manager.register_model(
            model_id="integration_test",
            name="Integration Test Model",
            size_gb=file_size / 1024 / 1024 / 1024,
            format="gguf",
            quantization="4bit",
            parameters=1000000,
            ipfs_hash="QmTestHash",
            owner="test_user",
            upload_block=1,
            storage_expires=999999,
            model_hash=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        )
        
        self.assertTrue(success)
        
        # Create torrent
        torrent_info = bt_distributor.create_model_torrent(
            model_id="integration_test",
            model_path=model_path
        )
        
        self.assertIsNotNone(torrent_info)
        self.assertEqual(torrent_info.model_id, "integration_test")
        self.assertGreater(torrent_info.file_size, 0)
        
        # Test seeding
        seed_success = bt_distributor.seed_model("integration_test", model_path)
        self.assertTrue(seed_success)
        
        # Check network statistics
        stats = bt_distributor.get_network_statistics()
        self.assertGreaterEqual(stats["total_models"], 1)


if __name__ == "__main__":
    unittest.main()