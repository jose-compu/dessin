"""
Enhanced networking capabilities using Chaincraft (UDP; large datagram payloads where supported).
Optimized for model distribution and PoGO consensus.
"""

import json
import time
import zlib
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from chaincraft import ChaincraftNode, SharedMessage, SharedObject
    CHAINCRAFT_AVAILABLE = True
except ImportError:
    CHAINCRAFT_AVAILABLE = False
    print("Warning: Chaincraft not available")

from ..models.model_manager import ModelManager


class MessageType(Enum):
    """Enhanced message types for DeSSIN"""
    # Basic blockchain messages
    BLOCK_ANNOUNCEMENT = "block_announcement"
    BLOCK_REQUEST = "block_request"
    BLOCK_RESPONSE = "block_response"
    
    # Transaction messages
    TRANSACTION = "transaction"
    TRANSACTION_POOL = "transaction_pool"
    
    # Model distribution messages (NEW - leveraging 16KB support)
    MODEL_CHUNK = "model_chunk"
    MODEL_METADATA = "model_metadata"
    MODEL_REQUEST = "model_request"
    TORRENT_ANNOUNCEMENT = "torrent_announcement"
    
    # PoGO consensus messages
    TRAINING_BLOCK = "training_block"
    ATTESTATION = "attestation"
    VERIFICATION_REQUEST = "verification_request"
    MERKLE_PROOF = "merkle_proof"
    
    # Network coordination
    PEER_DISCOVERY = "peer_discovery"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    SYNC_REQUEST = "sync_request"


@dataclass
class ModelChunk:
    """Represents a chunk of model data for transmission"""
    model_id: str
    chunk_index: int
    total_chunks: int
    chunk_hash: str
    data: bytes
    compression: str = "none"  # none, gzip, lz4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_hash": self.chunk_hash,
            "data": self.data.hex(),  # Convert bytes to hex for JSON
            "compression": self.compression
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelChunk":
        return cls(
            model_id=data["model_id"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            chunk_hash=data["chunk_hash"],
            data=bytes.fromhex(data["data"]),
            compression=data.get("compression", "none")
        )


@dataclass
class NodeCapabilities:
    """Announces node capabilities to the network"""
    node_id: str
    supports_model_distribution: bool = True
    supports_torrent_seeding: bool = True
    supports_pogo_consensus: bool = True
    max_model_size_gb: float = 50.0
    available_bandwidth_mbps: float = 100.0
    storage_available_gb: float = 1000.0
    gpu_available: bool = False
    preferred_model_formats: List[str] = None
    
    def __post_init__(self):
        if self.preferred_model_formats is None:
            self.preferred_model_formats = ["gguf"]


class EnhancedChaincraftNode:
    """Enhanced Chaincraft node with 16KB message support and model distribution"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        max_message_size: int = 16384,  # 16KB - new limit in v0.4.3r3
        enable_compression: bool = True,
        chunk_size: int = 12288,  # 12KB chunks (leaving room for metadata)
        **chaincraft_kwargs
    ):
        self.model_manager = model_manager
        self.max_message_size = max_message_size
        self.enable_compression = enable_compression
        self.chunk_size = chunk_size
        
        # Initialize Chaincraft node
        if CHAINCRAFT_AVAILABLE:
            self.chaincraft_node = ChaincraftNode(
                use_compression=enable_compression,
                **chaincraft_kwargs
            )
            # Check if the node supports larger messages
            if hasattr(self.chaincraft_node, 'max_msg_size'):
                actual_max = self.chaincraft_node.max_msg_size
                print(f"✓ Chaincraft max message size: {actual_max} bytes")
                if actual_max >= 16384:
                    print("✓ 16KB message support confirmed!")
                self.max_message_size = min(self.max_message_size, actual_max)
        else:
            self.chaincraft_node = None
        
        # Enhanced networking state
        self.node_capabilities = NodeCapabilities(
            node_id=self._generate_node_id(),
            supports_model_distribution=True,
            supports_torrent_seeding=True,
            max_model_size_gb=model_manager.config.max_model_size_gb
        )
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._register_default_handlers()
        
        # Model transfer state
        self.active_downloads: Dict[str, Dict] = {}  # model_id -> download state
        self.active_uploads: Dict[str, Dict] = {}    # model_id -> upload state
        
        # Peer tracking
        self.peer_capabilities: Dict[str, NodeCapabilities] = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "models_downloaded": 0,
            "models_uploaded": 0,
            "large_messages_sent": 0,  # Messages > 4KB
            "compression_ratio": 0.0
        }
        
        print(f"✓ Enhanced Chaincraft node initialized")
        print(f"  Max message size: {self.max_message_size} bytes")
        print(f"  Chunk size: {self.chunk_size} bytes")
        print(f"  Compression: {'enabled' if enable_compression else 'disabled'}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import socket
        import uuid
        hostname = socket.gethostname()
        unique_id = str(uuid.uuid4())[:8]
        return f"dessin_{hostname}_{unique_id}"
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers.update({
            MessageType.MODEL_CHUNK: self._handle_model_chunk,
            MessageType.MODEL_METADATA: self._handle_model_metadata,
            MessageType.MODEL_REQUEST: self._handle_model_request,
            MessageType.CAPABILITY_ANNOUNCEMENT: self._handle_capability_announcement,
            MessageType.TORRENT_ANNOUNCEMENT: self._handle_torrent_announcement
        })
    
    def start(self) -> bool:
        """Start the enhanced networking node"""
        if not self.chaincraft_node:
            print("Chaincraft node not available")
            return False
        
        success = self.chaincraft_node.start()
        if success:
            # Announce capabilities to network
            self.announce_capabilities()
            print(f"✓ Enhanced node started successfully")
            print(f"  Node ID: {self.node_capabilities.node_id}")
            print(f"  Listening on port: {getattr(self.chaincraft_node, 'port', 'unknown')}")
        
        return success
    
    def stop(self):
        """Stop the enhanced networking node"""
        if self.chaincraft_node:
            self.chaincraft_node.stop()
        print("Enhanced node stopped")
    
    def send_large_message(
        self, 
        message_type: MessageType, 
        data: Any,
        target_peer: Optional[str] = None,
        compress: bool = None
    ) -> bool:
        """Send a message, automatically handling large messages with chunking"""
        try:
            if compress is None:
                compress = self.enable_compression
            
            # Serialize data
            if isinstance(data, dict):
                serialized = json.dumps(data)
            elif isinstance(data, str):
                serialized = data
            else:
                serialized = str(data)
            
            # Compress if enabled and beneficial
            compressed_data = serialized.encode()
            compression_used = "none"
            
            if compress and len(compressed_data) > 1024:  # Only compress if > 1KB
                compressed = zlib.compress(compressed_data)
                if len(compressed) < len(compressed_data):
                    compressed_data = compressed
                    compression_used = "gzip"
                    
                    # Update compression stats
                    ratio = len(compressed) / len(compressed_data)
                    self.stats["compression_ratio"] = (
                        self.stats["compression_ratio"] * 0.9 + ratio * 0.1
                    )
            
            # Check if message fits in single packet
            metadata = {
                "type": message_type.value,
                "compression": compression_used,
                "timestamp": time.time(),
                "node_id": self.node_capabilities.node_id
            }
            
            metadata_size = len(json.dumps(metadata).encode())
            available_space = self.max_message_size - metadata_size - 100  # Safety margin
            
            if len(compressed_data) <= available_space:
                # Send as single message
                return self._send_single_message(metadata, compressed_data, target_peer)
            else:
                # Send as chunked message
                return self._send_chunked_message(metadata, compressed_data, target_peer)
                
        except Exception as e:
            print(f"Error sending large message: {e}")
            return False
    
    def _send_single_message(
        self, 
        metadata: Dict[str, Any], 
        data: bytes,
        target_peer: Optional[str] = None
    ) -> bool:
        """Send a single message"""
        try:
            message_data = metadata.copy()
            message_data["data"] = data.hex()
            message_data["chunked"] = False
            
            # Create and send message
            message = SharedMessage(data=message_data)
            
            if self.chaincraft_node:
                msg_hash, _ = self.chaincraft_node.create_shared_message(message_data)
                
                # Update stats
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(data)
                
                if len(data) > 4096:
                    self.stats["large_messages_sent"] += 1
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error sending single message: {e}")
            return False
    
    def _send_chunked_message(
        self, 
        metadata: Dict[str, Any], 
        data: bytes,
        target_peer: Optional[str] = None
    ) -> bool:
        """Send a message split into multiple chunks"""
        try:
            # Calculate chunks
            total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
            chunk_id = hashlib.sha256(data).hexdigest()[:16]
            
            print(f"Sending chunked message: {len(data)} bytes in {total_chunks} chunks")
            
            # Send metadata first
            metadata_msg = metadata.copy()
            metadata_msg.update({
                "chunked": True,
                "chunk_id": chunk_id,
                "total_chunks": total_chunks,
                "total_size": len(data),
                "data_hash": hashlib.sha256(data).hexdigest()
            })
            
            if not self._send_single_message(metadata_msg, b"", target_peer):
                return False
            
            # Send chunks
            for i in range(total_chunks):
                start_idx = i * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, len(data))
                chunk_data = data[start_idx:end_idx]
                
                chunk_msg = {
                    "type": "chunk",
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "chunk_hash": hashlib.sha256(chunk_data).hexdigest(),
                    "data": chunk_data.hex(),
                    "node_id": self.node_capabilities.node_id
                }
                
                if not self._send_single_message(chunk_msg, b"", target_peer):
                    print(f"Failed to send chunk {i}")
                    return False
                
                # Small delay between chunks to avoid overwhelming
                time.sleep(0.001)
            
            print(f"✓ Sent chunked message successfully")
            self.stats["messages_sent"] += total_chunks + 1
            self.stats["bytes_sent"] += len(data)
            
            return True
            
        except Exception as e:
            print(f"Error sending chunked message: {e}")
            return False
    
    def send_model_chunk(
        self, 
        model_id: str, 
        chunk_data: bytes,
        chunk_index: int,
        total_chunks: int
    ) -> bool:
        """Send a model chunk using enhanced messaging"""
        try:
            chunk = ModelChunk(
                model_id=model_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_hash=hashlib.sha256(chunk_data).hexdigest(),
                data=chunk_data
            )
            
            return self.send_large_message(
                MessageType.MODEL_CHUNK,
                chunk.to_dict(),
                compress=True
            )
            
        except Exception as e:
            print(f"Error sending model chunk: {e}")
            return False
    
    def request_model(self, model_id: str, preferred_peers: List[str] = None) -> bool:
        """Request a model from the network"""
        try:
            request_data = {
                "model_id": model_id,
                "requestor": self.node_capabilities.node_id,
                "timestamp": time.time(),
                "preferred_format": "gguf",
                "max_chunk_size": self.chunk_size
            }
            
            if preferred_peers:
                request_data["preferred_peers"] = preferred_peers
            
            success = self.send_large_message(
                MessageType.MODEL_REQUEST,
                request_data
            )
            
            if success:
                print(f"✓ Requested model {model_id} from network")
                # Initialize download tracking
                self.active_downloads[model_id] = {
                    "status": "requested",
                    "start_time": time.time(),
                    "chunks_received": {},
                    "total_chunks": None
                }
            
            return success
            
        except Exception as e:
            print(f"Error requesting model: {e}")
            return False
    
    def announce_capabilities(self) -> bool:
        """Announce node capabilities to the network"""
        try:
            return self.send_large_message(
                MessageType.CAPABILITY_ANNOUNCEMENT,
                asdict(self.node_capabilities)
            )
        except Exception as e:
            print(f"Error announcing capabilities: {e}")
            return False
    
    def announce_torrent(
        self, 
        model_id: str, 
        torrent_hash: str, 
        magnet_link: str,
        seeders: int = 1
    ) -> bool:
        """Announce a torrent to the network"""
        try:
            announcement = {
                "model_id": model_id,
                "torrent_hash": torrent_hash,
                "magnet_link": magnet_link,
                "seeders": seeders,
                "announcer": self.node_capabilities.node_id,
                "timestamp": time.time()
            }
            
            return self.send_large_message(
                MessageType.TORRENT_ANNOUNCEMENT,
                announcement
            )
            
        except Exception as e:
            print(f"Error announcing torrent: {e}")
            return False
    
    # Message handlers
    def _handle_model_chunk(self, message: Dict[str, Any]):
        """Handle incoming model chunk"""
        try:
            chunk = ModelChunk.from_dict(message)
            
            # Update download tracking
            if chunk.model_id in self.active_downloads:
                download = self.active_downloads[chunk.model_id]
                download["chunks_received"][chunk.chunk_index] = chunk
                download["total_chunks"] = chunk.total_chunks
                
                print(f"Received chunk {chunk.chunk_index + 1}/{chunk.total_chunks} "
                      f"for model {chunk.model_id}")
                
                # Check if download is complete
                if len(download["chunks_received"]) == chunk.total_chunks:
                    self._assemble_model(chunk.model_id)
            
        except Exception as e:
            print(f"Error handling model chunk: {e}")
    
    def _handle_model_metadata(self, message: Dict[str, Any]):
        """Handle model metadata message"""
        try:
            model_id = message.get("model_id")
            metadata = message.get("metadata", {})
            
            print(f"Received metadata for model {model_id}")
            # Store metadata for model assembly
            
        except Exception as e:
            print(f"Error handling model metadata: {e}")
    
    def _handle_model_request(self, message: Dict[str, Any]):
        """Handle model request from another peer"""
        try:
            model_id = message.get("model_id")
            requestor = message.get("requestor")
            
            print(f"Received model request for {model_id} from {requestor}")
            
            # Check if we have the model
            if model_id in self.model_manager.models:
                # Start upload process
                self._start_model_upload(model_id, requestor)
            
        except Exception as e:
            print(f"Error handling model request: {e}")
    
    def _handle_capability_announcement(self, message: Dict[str, Any]):
        """Handle peer capability announcement"""
        try:
            capabilities = NodeCapabilities(**message)
            self.peer_capabilities[capabilities.node_id] = capabilities
            
            print(f"Updated capabilities for peer {capabilities.node_id}")
            
        except Exception as e:
            print(f"Error handling capability announcement: {e}")
    
    def _handle_torrent_announcement(self, message: Dict[str, Any]):
        """Handle torrent announcement"""
        try:
            model_id = message.get("model_id")
            torrent_hash = message.get("torrent_hash")
            magnet_link = message.get("magnet_link")
            
            print(f"Received torrent announcement for {model_id}")
            print(f"  Torrent hash: {torrent_hash}")
            print(f"  Magnet link available")
            
        except Exception as e:
            print(f"Error handling torrent announcement: {e}")
    
    def _assemble_model(self, model_id: str):
        """Assemble a complete model from received chunks"""
        try:
            if model_id not in self.active_downloads:
                return
            
            download = self.active_downloads[model_id]
            chunks = download["chunks_received"]
            
            # Sort chunks by index
            sorted_chunks = sorted(chunks.items())
            
            # Assemble data
            assembled_data = b""
            for chunk_index, chunk in sorted_chunks:
                assembled_data += chunk.data
            
            # Save assembled model
            model_path = self.model_manager.cache_dir / f"{model_id}_downloaded.gguf"
            with open(model_path, "wb") as f:
                f.write(assembled_data)
            
            # Clean up download tracking
            del self.active_downloads[model_id]
            
            # Update stats
            self.stats["models_downloaded"] += 1
            
            print(f"✓ Successfully assembled model {model_id}")
            print(f"  Size: {len(assembled_data) / 1024 / 1024:.1f} MB")
            print(f"  Saved to: {model_path}")
            
        except Exception as e:
            print(f"Error assembling model: {e}")
    
    def _start_model_upload(self, model_id: str, requestor: str):
        """Start uploading a model to a peer"""
        try:
            # This would implement the upload logic
            print(f"Starting upload of {model_id} to {requestor}")
            # Implementation would chunk the model and send via send_model_chunk
            
        except Exception as e:
            print(f"Error starting model upload: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get enhanced networking statistics"""
        stats = self.stats.copy()
        stats.update({
            "known_peers": len(self.peer_capabilities),
            "active_downloads": len(self.active_downloads),
            "active_uploads": len(self.active_uploads),
            "max_message_size": self.max_message_size,
            "avg_compression_ratio": self.stats["compression_ratio"]
        })
        return stats
    
    def list_peer_capabilities(self) -> List[Dict[str, Any]]:
        """List capabilities of known peers"""
        return [asdict(cap) for cap in self.peer_capabilities.values()]
