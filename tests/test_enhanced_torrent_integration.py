#!/usr/bin/env python3
"""
Test enhanced BitTorrent integration with real ports and file sharing verification
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

from dessin.distribution.enhanced_torrent_distributor import EnhancedTorrentDistributor, TorrentPeer
from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager
from dessin.runtime.config import DessinConfig


class TestEnhancedTorrentIntegration(unittest.TestCase):
    """Test enhanced BitTorrent integration with port management"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.base_port = 6881
        
    def tearDown(self):
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_enhanced_torrent_distributor_creation(self):
        """Test creating enhanced torrent distributors with different ports"""
        print("\n" + "="*60)
        print("TESTING ENHANCED TORRENT DISTRIBUTOR CREATION")
        print("="*60)
        
        distributors = []
        
        # Create 4 distributors with different ports
        for i in range(4):
            node_id = f"test_node_{i+1}"
            port = self.base_port + i
            torrent_dir = self.temp_dir / f"node_{i+1}"
            
            distributor = EnhancedTorrentDistributor(
                node_id=node_id,
                torrent_dir=str(torrent_dir),
                listen_port=port
            )
            
            distributors.append(distributor)
            
            self.assertEqual(distributor.node_id, node_id)
            self.assertEqual(distributor.listen_port, port)
            self.assertTrue(distributor.torrent_dir.exists())
            
            print(f"✓ Created distributor {node_id} on port {port}")
        
        print(f"✓ All {len(distributors)} distributors created successfully")

    def test_torrent_creation_with_tracker_ports(self):
        """Test creating torrents with tracker port information"""
        print("\n" + "="*60)
        print("TESTING TORRENT CREATION WITH TRACKER PORTS")
        print("="*60)
        
        # Create distributor
        distributor = EnhancedTorrentDistributor(
            node_id="creator_node",
            torrent_dir=str(self.temp_dir / "creator"),
            listen_port=6881
        )
        
        # Create test model file
        model_file = self.temp_dir / "test_model.json"
        with open(model_file, 'w') as f:
            json.dump({
                "model_id": "test_model",
                "weights": "0x" + "abc123" * 50,
                "loss_improvement": 0.15
            }, f)
        
        # Create torrent with tracker ports
        tracker_ports = [6882, 6883, 6884]
        torrent_info = distributor.create_torrent_with_ports(
            str(model_file),
            "test_model",
            tracker_ports
        )
        
        self.assertIsNotNone(torrent_info)
        self.assertEqual(torrent_info.model_id, "test_model")
        self.assertEqual(torrent_info.listen_port, 6881)
        self.assertEqual(torrent_info.tracker_ports, tracker_ports)
        self.assertTrue(torrent_info.torrent_hash)
        self.assertTrue(torrent_info.magnet_link)
        self.assertGreater(torrent_info.file_size, 0)
        
        # Verify tracker ports in magnet link
        for port in tracker_ports:
            tracker_url = f"tr=http://127.0.0.1:{port}/announce"
            self.assertIn(tracker_url, torrent_info.magnet_link)
        
        print(f"✓ Torrent created with tracker ports: {tracker_ports}")
        print(f"  Hash: {torrent_info.torrent_hash[:16]}...")
        print(f"  File size: {torrent_info.file_size} bytes")

    def test_peer_management(self):
        """Test adding and managing torrent peers"""
        print("\n" + "="*60)
        print("TESTING PEER MANAGEMENT")
        print("="*60)
        
        # Create distributor
        distributor = EnhancedTorrentDistributor(
            node_id="peer_manager",
            torrent_dir=str(self.temp_dir / "peer_test"),
            listen_port=6881
        )
        
        # Create test torrent
        model_file = self.temp_dir / "peer_test_model.json"
        with open(model_file, 'w') as f:
            json.dump({"model_id": "peer_test", "data": "test"}, f)
        
        torrent_info = distributor.create_torrent_with_ports(
            str(model_file),
            "peer_test_model",
            [6882, 6883]
        )
        
        # Add peers
        seeder_peer = TorrentPeer(
            ip="127.0.0.1",
            port=6882,
            node_id="seeder_node",
            last_seen=time.time(),
            is_seeder=True
        )
        
        leecher_peer = TorrentPeer(
            ip="127.0.0.1",
            port=6883,
            node_id="leecher_node",
            last_seen=time.time(),
            is_seeder=False
        )
        
        distributor.add_peer_to_torrent(torrent_info.torrent_hash, seeder_peer)
        distributor.add_peer_to_torrent(torrent_info.torrent_hash, leecher_peer)
        
        # Verify peer counts
        status = distributor.get_torrent_status(torrent_info.torrent_hash)
        self.assertIsNotNone(status)
        self.assertEqual(len(status["seeders"]), 1)
        self.assertEqual(len(status["leechers"]), 1)
        
        print(f"✓ Added peers to torrent {torrent_info.torrent_hash[:8]}...")
        print(f"  Seeders: {len(status['seeders'])}")
        print(f"  Leechers: {len(status['leechers'])}")

    def test_file_sharing_verification(self):
        """Test file sharing verification between peers"""
        print("\n" + "="*60)
        print("TESTING FILE SHARING VERIFICATION")
        print("="*60)
        
        # Create distributor
        distributor = EnhancedTorrentDistributor(
            node_id="verifier",
            torrent_dir=str(self.temp_dir / "verification"),
            listen_port=6881
        )
        
        # Create test torrent
        model_file = self.temp_dir / "verification_model.json"
        with open(model_file, 'w') as f:
            json.dump({"verification": True, "data": "test"}, f)
        
        torrent_info = distributor.create_torrent_with_ports(
            str(model_file),
            "verification_model",
            [6882, 6883, 6884]
        )
        
        # Test verification
        peer_ports = [6882, 6883, 6884]
        verification = distributor.verify_file_sharing(
            torrent_info.torrent_hash,
            peer_ports
        )
        
        self.assertTrue(verification["success"])
        self.assertEqual(verification["torrent_hash"], torrent_info.torrent_hash)
        self.assertTrue(verification["file_exists"])
        self.assertEqual(verification["listen_port"], 6881)
        self.assertEqual(len(verification["peer_connections"]), 3)
        
        print(f"✓ File sharing verification completed")
        print(f"  File exists: {verification['file_exists']}")
        print(f"  Peer connections tested: {len(verification['peer_connections'])}")
        
        # Show peer connection results
        for conn in verification["peer_connections"]:
            status = "✓" if conn["reachable"] else "✗"
            print(f"    {status} {conn['node_id']} (port {conn['port']})")

    def test_consensus_integration_with_enhanced_torrents(self):
        """Test consensus integration with enhanced torrent information"""
        print("\n" + "="*60)
        print("TESTING CONSENSUS INTEGRATION WITH ENHANCED TORRENTS")
        print("="*60)
        
        # Create consensus with enhanced torrent support
        config = DessinConfig.default()
        config.model.model_cache_dir = str(self.temp_dir / "consensus_test")
        
        model_manager = ModelManager(config.model)
        model_manager.register_model(
            model_id="enhanced_test_model",
            name="Enhanced Test Model",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash="QmEnhancedTest",
            owner="enhanced_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash="enhanced_hash"
        )
        
        consensus = PogoConsensus(
            config.consensus,
            model_manager,
            "enhanced_miner",
            torrent_port=6881
        )
        
        # Create training block
        block = consensus.create_training_block("enhanced_test_model")
        
        self.assertIsNotNone(block)
        self.assertIsInstance(block, PogoBlock)
        
        # Verify enhanced torrent fields
        self.assertTrue(hasattr(block, 'torrent_listen_port'))
        self.assertTrue(hasattr(block, 'torrent_tracker_ports'))
        self.assertTrue(hasattr(block, 'torrent_piece_count'))
        self.assertTrue(hasattr(block, 'torrent_seeders'))
        
        # Check that enhanced torrent fields are populated
        self.assertGreater(block.torrent_listen_port, 0)
        self.assertIsNotNone(block.torrent_tracker_ports)
        self.assertGreater(block.torrent_piece_count, 0)
        self.assertGreaterEqual(block.torrent_seeders, 0)
        
        print(f"✓ Enhanced block created with torrent integration")
        print(f"  Torrent hash: {block.torrent_hash[:16]}...")
        print(f"  Listen port: {block.torrent_listen_port}")
        print(f"  Tracker ports: {block.torrent_tracker_ports}")
        print(f"  Piece count: {block.torrent_piece_count}")
        print(f"  Seeders: {block.torrent_seeders}")

    def test_block_serialization_with_enhanced_torrent_data(self):
        """Test block serialization with enhanced torrent fields"""
        print("\n" + "="*60)
        print("TESTING BLOCK SERIALIZATION WITH ENHANCED TORRENT DATA")
        print("="*60)
        
        # Create consensus
        config = DessinConfig.default()
        config.model.model_cache_dir = str(self.temp_dir / "serialization_test")
        
        model_manager = ModelManager(config.model)
        model_manager.register_model(
            model_id="serialization_test_model",
            name="Serialization Test Model",
            size_gb=0.001,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash="QmSerializationTest",
            owner="serialization_miner",
            upload_block=0,
            storage_expires=1000,
            model_hash="serialization_hash"
        )
        
        consensus = PogoConsensus(
            config.consensus,
            model_manager,
            "serialization_miner",
            torrent_port=6882
        )
        
        block = consensus.create_training_block("serialization_test_model")
        self.assertIsNotNone(block)
        
        # Test serialization
        block_dict = block.to_dict()
        self.assertIsInstance(block_dict, dict)
        
        # Verify enhanced torrent fields are in serialized data
        enhanced_fields = [
            "torrent_listen_port",
            "torrent_tracker_ports", 
            "torrent_piece_count",
            "torrent_piece_length",
            "torrent_seeders",
            "torrent_created_at"
        ]
        
        for field in enhanced_fields:
            self.assertIn(field, block_dict)
            print(f"✓ Found enhanced field: {field} = {block_dict[field]}")
        
        # Test JSON serialization
        json_str = json.dumps(block_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        deserialized_dict = json.loads(json_str)
        reconstructed_block = PogoBlock.from_dict(deserialized_dict)
        
        # Verify enhanced fields are preserved
        self.assertEqual(reconstructed_block.torrent_listen_port, block.torrent_listen_port)
        self.assertEqual(reconstructed_block.torrent_tracker_ports, block.torrent_tracker_ports)
        self.assertEqual(reconstructed_block.torrent_piece_count, block.torrent_piece_count)
        self.assertEqual(reconstructed_block.torrent_seeders, block.torrent_seeders)
        
        print("✓ Enhanced torrent data serialization/deserialization successful")


def run_enhanced_torrent_tests():
    """Run all enhanced torrent integration tests"""
    print("🔗 ENHANCED TORRENT INTEGRATION TESTING")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnhancedTorrentIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL ENHANCED TORRENT TESTS PASSED!")
        print(f"✓ {result.testsRun} tests completed successfully")
        print("✓ Enhanced torrent distributors working")
        print("✓ Port management working")
        print("✓ Peer discovery and tracking working")
        print("✓ File sharing verification working")
        print("✓ Consensus integration working")
        print("✓ Enhanced block serialization working")
    else:
        print("💥 SOME ENHANCED TORRENT TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_enhanced_torrent_tests()
    sys.exit(0 if success else 1)
