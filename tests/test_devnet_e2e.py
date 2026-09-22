"""
End-to-end tests for DeSSIN devnet simulation.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Try to import components, fallback to mocks if unavailable
try:
    from dessin.distribution.bittorrent_distributor import BitTorrentDistributor
    from dessin.distribution.huggingface_distributor import HuggingFaceDistributor
    from dessin.distribution.hybrid_distributor import HybridDistributor
    from dessin.models.model_manager import ModelManager
    from dessin.runtime.progress_tracker import ProgressTracker
    from dessin.runtime.config import ModelConfig
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    # Create mock classes
    class BitTorrentDistributor:
        def __init__(self, model_manager):
            self.model_manager = model_manager
            self.active_downloads = {}
        
        async def create_model_torrent(self, model_id, model_path):
            return Mock()  # Return a mock object that can be awaited
        
        async def download_model_torrent(self, torrent_info, save_path):
            return True, f"{save_path}/mock_model.gguf", 1.0
    
    class HuggingFaceDistributor:
        def __init__(self, model_manager, force_mock=True):
            self.model_manager = model_manager
        
        async def download_model(self, model_id, save_path=None):
            return True, f"/tmp/{model_id}.gguf", 1.0
    
    class HybridDistributor:
        def __init__(self, model_manager):
            self.model_manager = model_manager
            self.distributors = {}
        
        async def download_model(self, model_id, **kwargs):
            return {"success": True, "method_used": "mock", "local_path": f"/tmp/{model_id}.gguf"}
    
    class ModelManager:
        def __init__(self, config):
            self.config = config
            self.models = {}
        
        def register_model(self, **kwargs):
            model_id = kwargs.get('model_id', 'test_model')
            self.models[model_id] = kwargs
        
        def get_model_info(self, model_id):
            return self.models.get(model_id)
    
    class ProgressTracker:
        def __init__(self):
            self.operations = {}
        
        def start_operation(self, operation_id, operation_type, description, total_size=0):
            self.operations[operation_id] = {"status": "running"}
        
        def complete_operation(self, operation_id):
            if operation_id in self.operations:
                self.operations[operation_id]["status"] = "completed"
    
    class ModelConfig:
        def __init__(self):
            self.model_cache_dir = "/tmp"


class MockNode:
    """Mock node for testing"""
    
    def __init__(self, node_id: str, port: int = 8000):
        self.node_id = node_id
        self.port = port
        self.peers = []
        self.models = {}
        self.is_running = False
    
    async def start(self):
        """Start the node"""
        self.is_running = True
        print(f"Node {self.node_id} started on port {self.port}")
    
    async def stop(self):
        """Stop the node"""
        self.is_running = False
        print(f"Node {self.node_id} stopped")
    
    async def connect_to_peer(self, peer):
        """Connect to another peer"""
        if peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)
            print(f"Node {self.node_id} connected to {peer.node_id}")
    
    async def broadcast_model(self, model_id: str, model_data: dict):
        """Broadcast model to peers"""
        # Store locally and propagate across the connected graph (BFS)
        self.models[model_id] = model_data
        visited = set()
        queue = [self]
        propagated_count = 0
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for peer in current.peers:
                if peer.is_running and model_id not in peer.models:
                    peer.models[model_id] = model_data
                    propagated_count += 1
                    queue.append(peer)
        print(f"Node {self.node_id} broadcasted model {model_id} to {propagated_count} peers")
    
    def get_model(self, model_id: str):
        """Get model from local storage"""
        return self.models.get(model_id)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def model_config(temp_dir):
    """Create model config for testing"""
    config = ModelConfig()
    config.model_cache_dir = str(temp_dir)
    return config


@pytest.fixture
def model_manager(model_config):
    """Create model manager for testing"""
    return ModelManager(model_config)


@pytest.fixture
def progress_tracker():
    """Create progress tracker for testing"""
    return ProgressTracker()


@pytest.fixture
def bittorrent_distributor(model_manager):
    """Create BitTorrent distributor for testing"""
    return BitTorrentDistributor(model_manager)


@pytest.fixture
def huggingface_distributor(model_manager):
    """Create HuggingFace distributor for testing"""
    return HuggingFaceDistributor(model_manager, force_mock=True)


@pytest.fixture
def hybrid_distributor(model_manager):
    """Create hybrid distributor for testing"""
    return HybridDistributor(model_manager)


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_single_node_operation(model_manager, progress_tracker):
    """Test single node operation"""
    # Create a single node
    node = MockNode("node_1", 8001)
    
    # Start the node
    await node.start()
    assert node.is_running is True
    
    # Register a model
    model_manager.register_model(
        model_id="test_model_1",
        name="Test Model 1",
        owner="test_owner",
        ipfs_hash="QmTestHash123456789",
        upload_block=1000,
        storage_expires=2000,
        size_gb=1.0,
        format="gguf",
        quantization="4bit",
        parameters=100000000,
        model_hash="hash1"
    )
    
    # Broadcast the model
    model_data = {
        "model_id": "test_model_1",
        "name": "Test Model 1",
        "size_gb": 1.0
    }
    await node.broadcast_model("test_model_1", model_data)
    
    # Verify model is stored locally
    stored_model = node.get_model("test_model_1")
    assert stored_model is not None
    assert stored_model["model_id"] == "test_model_1"
    
    # Stop the node
    await node.stop()
    assert node.is_running is False


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_two_node_network():
    """Test two-node network communication"""
    # Create two nodes
    node1 = MockNode("node_1", 8001)
    node2 = MockNode("node_2", 8002)
    
    # Start both nodes
    await node1.start()
    await node2.start()
    
    # Connect nodes
    await node1.connect_to_peer(node2)
    
    # Verify connection
    assert node2 in node1.peers
    assert node1 in node2.peers
    
    # Broadcast model from node1
    model_data = {
        "model_id": "shared_model",
        "name": "Shared Model",
        "size_gb": 2.0
    }
    await node1.broadcast_model("shared_model", model_data)
    
    # Verify model propagated to node2
    stored_model = node2.get_model("shared_model")
    assert stored_model is not None
    assert stored_model["model_id"] == "shared_model"
    
    # Stop nodes
    await node1.stop()
    await node2.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_five_node_network():
    """Test five-node network with model distribution"""
    # Create five nodes
    nodes = []
    for i in range(5):
        node = MockNode(f"node_{i+1}", 8001 + i)
        nodes.append(node)
    
    # Start all nodes
    for node in nodes:
        await node.start()
    
    # Connect nodes in a ring topology
    for i in range(5):
        await nodes[i].connect_to_peer(nodes[(i + 1) % 5])
    
    # Verify connections
    for i, node in enumerate(nodes):
        assert len(node.peers) == 2  # Each node has 2 peers in ring
        expected_peers = [nodes[(i - 1) % 5], nodes[(i + 1) % 5]]
        for peer in expected_peers:
            assert peer in node.peers
    
    # Broadcast model from node 0
    model_data = {
        "model_id": "distributed_model",
        "name": "Distributed Model",
        "size_gb": 3.0
    }
    await nodes[0].broadcast_model("distributed_model", model_data)
    
    # Verify model propagated to all nodes
    for node in nodes:
        stored_model = node.get_model("distributed_model")
        assert stored_model is not None
        assert stored_model["model_id"] == "distributed_model"
    
    # Stop all nodes
    for node in nodes:
        await node.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_model_distribution_with_bittorrent(bittorrent_distributor, model_manager):
    """Test model distribution using BitTorrent"""
    # Register a model
    model_manager.register_model(
        model_id="bittorrent_model",
        name="BitTorrent Model",
        owner="test_owner",
        ipfs_hash="QmTestHash123456789",
        upload_block=1000,
        storage_expires=2000,
        size_gb=5.0,
        format="gguf",
        quantization="4bit",
        parameters=500000000,
        model_hash="bt_hash"
    )
    
    # Create torrent for the model
    model_path = "/tmp/bittorrent_model.gguf"
    # Create a dummy model file for testing
    with open(model_path, "w") as f:
        f.write("Mock model content for testing")
    
    torrent_info = await bittorrent_distributor.create_model_torrent("bittorrent_model", model_path)
    
    assert torrent_info is not None
    assert hasattr(torrent_info, 'torrent_hash')
    assert hasattr(torrent_info, 'magnet_link')
    
    # Download the model via torrent
    save_path = "/tmp/downloads"
    success, local_path, download_time = await bittorrent_distributor.download_model_torrent(
        torrent_info, save_path
    )
    
    assert success is True
    assert local_path is not None
    assert download_time > 0


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_model_distribution_with_huggingface(huggingface_distributor, model_manager):
    """Test model distribution using HuggingFace"""
    # Download model from HuggingFace
    model_id = "microsoft/DialoGPT-small"
    success, local_path, download_time = await huggingface_distributor.download_model(model_id)
    
    assert success is True
    assert local_path is not None
    assert download_time > 0
    
    # Verify model was registered
    model_info = model_manager.get_model_info(model_id)
    assert model_info is not None


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_hybrid_distribution(hybrid_distributor, model_manager):
    """Test hybrid distribution system"""
    # Download model using hybrid distributor
    model_id = "hybrid_test_model"
    result = await hybrid_distributor.download_model(model_id)
    
    assert result.success is True
    assert result.method_used is not None
    assert result.local_path is not None


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_progress_tracking(progress_tracker):
    """Test progress tracking functionality"""
    # Start an operation
    operation_id = "test_operation"
    progress_tracker.start_operation(operation_id, "download", "Test download", 100)
    
    # Verify operation is tracked
    assert operation_id in progress_tracker.operations
    assert progress_tracker.operations[operation_id].status == "running"
    
    # Complete the operation
    progress_tracker.complete_operation(operation_id)
    assert progress_tracker.operations[operation_id].status == "completed"


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_network_partition():
    """Test network partition scenarios"""
    # Create nodes
    node1 = MockNode("node_1", 8001)
    node2 = MockNode("node_2", 8002)
    node3 = MockNode("node_3", 8003)
    
    # Start nodes
    await node1.start()
    await node2.start()
    await node3.start()
    
    # Connect nodes
    await node1.connect_to_peer(node2)
    await node2.connect_to_peer(node3)
    
    # Broadcast model from node1
    model_data = {"model_id": "partition_test", "name": "Partition Test"}
    await node1.broadcast_model("partition_test", model_data)
    
    # Verify model propagated to connected nodes
    assert node1.get_model("partition_test") is not None
    assert node2.get_model("partition_test") is not None
    assert node3.get_model("partition_test") is not None
    
    # Simulate partition by disconnecting node3
    node2.peers.remove(node3)
    node3.peers.remove(node2)
    
    # Broadcast new model from node1
    new_model_data = {"model_id": "after_partition", "name": "After Partition"}
    await node1.broadcast_model("after_partition", new_model_data)
    
    # Verify model only reached connected nodes
    assert node1.get_model("after_partition") is not None
    assert node2.get_model("after_partition") is not None
    assert node3.get_model("after_partition") is None  # Disconnected
    
    # Stop nodes
    await node1.stop()
    await node2.stop()
    await node3.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_concurrent_model_broadcasts():
    """Test concurrent model broadcasts"""
    # Create nodes
    nodes = []
    for i in range(3):
        node = MockNode(f"node_{i+1}", 8001 + i)
        nodes.append(node)
    
    # Start and connect nodes
    for node in nodes:
        await node.start()
    
    for i in range(3):
        await nodes[i].connect_to_peer(nodes[(i + 1) % 3])
    
    # Broadcast models concurrently
    async def broadcast_model(node_id, model_id):
        node = nodes[node_id]
        model_data = {"model_id": model_id, "name": f"Model {model_id}"}
        await node.broadcast_model(model_id, model_data)
    
    # Start concurrent broadcasts
    tasks = [
        broadcast_model(0, "concurrent_model_1"),
        broadcast_model(1, "concurrent_model_2"),
        broadcast_model(2, "concurrent_model_3")
    ]
    
    await asyncio.gather(*tasks)
    
    # Verify all models propagated
    for node in nodes:
        for model_id in ["concurrent_model_1", "concurrent_model_2", "concurrent_model_3"]:
            assert node.get_model(model_id) is not None
    
    # Stop nodes
    for node in nodes:
        await node.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_large_network_simulation():
    """Test large network simulation (10 nodes)"""
    # Create 10 nodes
    nodes = []
    for i in range(10):
        node = MockNode(f"node_{i+1}", 8001 + i)
        nodes.append(node)
    
    # Start all nodes
    for node in nodes:
        await node.start()
    
    # Connect nodes in a mesh topology (each node connects to 3 others)
    for i in range(10):
        for j in range(1, 4):  # Connect to next 3 nodes
            target_idx = (i + j) % 10
            if nodes[target_idx] not in nodes[i].peers:
                await nodes[i].connect_to_peer(nodes[target_idx])
    
    # Verify connections
    for node in nodes:
        assert len(node.peers) >= 3  # Each node should have at least 3 peers
    
    # Broadcast model from multiple nodes
    for i in range(5):
        model_data = {"model_id": f"large_network_model_{i}", "name": f"Large Network Model {i}"}
        await nodes[i].broadcast_model(f"large_network_model_{i}", model_data)
    
    # Verify all models propagated to all nodes
    for node in nodes:
        for i in range(5):
            model_id = f"large_network_model_{i}"
            assert node.get_model(model_id) is not None
    
    # Stop all nodes
    for node in nodes:
        await node.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
@pytest.mark.asyncio
async def test_network_recovery():
    """Test network recovery after node failures"""
    # Create nodes
    nodes = []
    for i in range(5):
        node = MockNode(f"node_{i+1}", 8001 + i)
        nodes.append(node)
    
    # Start and connect nodes
    for node in nodes:
        await node.start()
    
    for i in range(5):
        await nodes[i].connect_to_peer(nodes[(i + 1) % 5])
    
    # Broadcast initial model
    await nodes[0].broadcast_model("recovery_test", {"model_id": "recovery_test"})
    
    # Simulate node failure (stop node 2)
    await nodes[2].stop()
    
    # Broadcast new model (should not reach failed node)
    await nodes[1].broadcast_model("after_failure", {"model_id": "after_failure"})
    
    # Verify model reached connected nodes but not failed node
    assert nodes[0].get_model("after_failure") is not None
    assert nodes[1].get_model("after_failure") is not None
    assert nodes[3].get_model("after_failure") is not None
    assert nodes[4].get_model("after_failure") is not None
    assert nodes[2].get_model("after_failure") is None  # Failed node
    
    # Restart failed node
    await nodes[2].start()
    
    # Reconnect to network
    await nodes[1].connect_to_peer(nodes[2])
    await nodes[3].connect_to_peer(nodes[2])
    
    # Broadcast recovery model
    await nodes[0].broadcast_model("recovery_model", {"model_id": "recovery_model"})
    
    # Verify all nodes have the recovery model
    for node in nodes:
        assert node.get_model("recovery_model") is not None
    
    # Stop all nodes
    for node in nodes:
        await node.stop()


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
def test_performance_benchmarks():
    """Test performance benchmarks"""
    import time
    
    # Test node creation performance
    start_time = time.time()
    nodes = []
    for i in range(20):
        node = MockNode(f"perf_node_{i}", 9001 + i)
        nodes.append(node)
    creation_time = time.time() - start_time
    
    assert creation_time < 1.0  # Should create 20 nodes quickly
    
    # Test model storage performance
    start_time = time.time()
    for i, node in enumerate(nodes):
        for j in range(10):
            model_data = {"model_id": f"perf_model_{i}_{j}", "name": f"Performance Model {i}_{j}"}
            node.models[f"perf_model_{i}_{j}"] = model_data
    storage_time = time.time() - start_time
    
    assert storage_time < 1.0  # Should store 200 models quickly
    
    # Verify storage
    total_models = sum(len(node.models) for node in nodes)
    assert total_models == 200
