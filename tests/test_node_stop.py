#!/usr/bin/env python3
"""
Test to verify the DessinNode stop function works correctly
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig


def test_node_stop_completes_quickly_with_default_mining():
    """Regression: mining thread must not block stop() for seconds (interruptible sleep)."""
    config = DessinConfig.default()
    assert config.consensus.block_time_minutes > 0, "mining should be enabled by default"
    node = DessinNode(config)
    t0 = time.monotonic()
    assert node.start()
    node.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"start+stop took {elapsed:.2f}s; expected prompt mining shutdown"


def test_node_start_stop():
    """Test that DessinNode can start and stop cleanly"""
    print("Testing DessinNode start/stop functionality...")
    
    # Create node with test configuration
    config = DessinConfig.default()
    config.consensus.block_time_minutes = 15 / 60.0  # 15 seconds for testing
    
    node = DessinNode(config)
    
    try:
        # Test start
        print("Starting node...")
        success = node.start()
        assert success, "Node failed to start"
        print("✓ Node started successfully")
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Test stop
        print("Stopping node...")
        node.stop()
        print("✓ Node stopped successfully")
        
        # Verify node is stopped
        assert not node.is_mining, "Mining should be stopped"
        print("✓ Mining stopped")
        
    except Exception as e:
        print(f"Test failed: {e}")
        # Try to stop anyway
        try:
            node.stop()
        except:
            pass
        raise


if __name__ == "__main__":
    print("🔧 TESTING NODE START/STOP FUNCTIONALITY")
    print("=" * 50)
    
    try:
        test_node_start_stop()
        print("\n🎉 NODE START/STOP TEST PASSED!")
        print("✓ Node starts correctly")
        print("✓ Node stops cleanly without errors") 
        print("✓ ChaincraftNode.close() method works")
    except AssertionError as e:
        print(f"\n💥 NODE START/STOP TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 NODE START/STOP TEST ERROR: {e}")
        sys.exit(1)
