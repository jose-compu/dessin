"""
Mock HuggingFace integration for testing without internet/API key.
"""

import os
import json
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from ..models.model_manager import ModelManager


@dataclass
class MockModelInfo:
    """Mock model info structure"""
    modelId: str
    downloads: int = 0
    created_at: str = "2024-01-01"


@dataclass 
class MockFileInfo:
    """Mock file info structure"""
    rfilename: str
    size: int = 1024*1024  # 1MB default


class MockHuggingFaceIntegration:
    """Mock HuggingFace integration for testing"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Mock data for testing
        self.mock_models = [
            {
                "repo_id": "microsoft/DialoGPT-small",
                "author": "microsoft", 
                "downloads": 150000,
                "estimated_params_b": 0.117,
                "gguf_files": ["dialogpt-small-q4_0.gguf"],
                "created_at": "2023-01-01"
            },
            {
                "repo_id": "bartowski/gemma-2-2b-it-GGUF",
                "author": "bartowski",
                "downloads": 50000,
                "estimated_params_b": 2.0,
                "gguf_files": ["gemma-2-2b-it-Q4_0.gguf", "gemma-2-2b-it-Q8_0.gguf"],
                "created_at": "2024-06-01"
            },
            {
                "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                "author": "TheBloke",
                "downloads": 500000, 
                "estimated_params_b": 7.0,
                "gguf_files": ["llama-2-7b-chat.Q4_0.gguf", "llama-2-7b-chat.Q8_0.gguf"],
                "created_at": "2023-07-01"
            },
            {
                "repo_id": "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF",
                "author": "NousResearch",
                "downloads": 75000,
                "estimated_params_b": 7.0,
                "gguf_files": ["Hermes-2-Pro-Mistral-7B.Q4_0.gguf"],
                "created_at": "2024-03-01"
            }
        ]
        
        self.trusted_publishers = {
            "microsoft", "google", "meta-llama", "huggingface", 
            "TheBloke", "NousResearch", "teknium", "Open-Orca",
            "bartowski", "QuantFactory", "mlabonne"
        }
    
    def search_gguf_models(
        self, 
        size_category: str = "small",
        max_results: int = 10
    ) -> List[Dict]:
        """Mock search for GGUF models"""
        
        # Filter by size category
        size_ranges = {
            "tiny": (0, 1),
            "small": (1, 7),
            "medium": (7, 13),
            "large": (13, 30),
        }
        
        min_params, max_params = size_ranges.get(size_category, (0, 10))
        
        filtered_models = []
        for model in self.mock_models:
            params = model["estimated_params_b"]
            if min_params <= params <= max_params:
                filtered_models.append(model)
        
        # Sort by downloads and limit results
        filtered_models.sort(key=lambda x: x["downloads"], reverse=True)
        return filtered_models[:max_results]
    
    def download_gguf_model(
        self, 
        repo_id: str, 
        filename: Optional[str] = None,
        prefer_4bit: bool = True,
        force_download: bool = False
    ) -> Optional[Tuple[str, str]]:
        """Mock download of a GGUF model"""
        
        # Find the model in our mock data
        model_data = None
        for model in self.mock_models:
            if model["repo_id"] == repo_id:
                model_data = model
                break
        
        if not model_data:
            print(f"Mock: Model {repo_id} not found in mock data")
            return None
        
        # Select filename
        if filename is None:
            gguf_files = model_data["gguf_files"]
            if prefer_4bit:
                # Look for Q4 files
                q4_files = [f for f in gguf_files if "Q4" in f or "q4" in f]
                filename = q4_files[0] if q4_files else gguf_files[0]
            else:
                filename = gguf_files[0]
        
        # Create a mock model file
        cache_dir = self.model_manager.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from repo_id
        safe_repo_id = repo_id.replace("/", "_")
        local_filename = f"{safe_repo_id}_{filename}"
        local_path = cache_dir / local_filename
        
        # Create mock GGUF file if it doesn't exist
        if not local_path.exists() or force_download:
            # Create mock GGUF content
            model_size_mb = int(model_data["estimated_params_b"] * 500)  # Rough size estimation
            mock_content = self._create_mock_gguf_content(repo_id, model_size_mb)
            
            with open(local_path, "wb") as f:
                f.write(mock_content)
            
            print(f"Mock: Created mock model file at {local_path}")
            print(f"Mock: File size: {len(mock_content) / 1024:.1f} KB")
        
        return repo_id, str(local_path)
    
    def _create_mock_gguf_content(self, repo_id: str, size_mb: int) -> bytes:
        """Create mock GGUF file content"""
        # Create a basic GGUF-like header
        header = b"GGUF"  # Magic number
        header += (2).to_bytes(4, 'little')  # Version
        header += (0).to_bytes(8, 'little')  # Tensor count
        header += (0).to_bytes(8, 'little')  # KV count
        
        # Add model info as mock metadata
        info = f"Mock model: {repo_id}".encode()
        header += len(info).to_bytes(4, 'little') + info
        
        # Fill the rest with pseudo-random data to reach target size
        remaining_size = max(1024, size_mb * 1024) - len(header)  # At least 1KB
        
        # Use repo_id hash as seed for consistent mock data
        import hashlib
        seed = int(hashlib.md5(repo_id.encode()).hexdigest()[:8], 16)
        
        # Generate pseudo-random content
        content = bytearray()
        for i in range(remaining_size):
            content.append((seed + i) % 256)
        
        return header + bytes(content)
    
    def download_recommended_model(
        self, 
        size_category: str = "tiny",
        prefer_4bit: bool = True
    ) -> Optional[Tuple[str, str]]:
        """Mock download of a recommended model"""
        
        if size_category == "tiny":
            repo_id = "microsoft/DialoGPT-small"
        elif size_category == "small":
            repo_id = "bartowski/gemma-2-2b-it-GGUF"
        else:
            repo_id = "TheBloke/Llama-2-7B-Chat-GGUF"
        
        return self.download_gguf_model(repo_id, prefer_4bit=prefer_4bit)
    
    def register_downloaded_model(
        self, 
        repo_id: str, 
        local_path: str,
        owner_address: str,
        storage_blocks: int = 1000
    ) -> Optional[str]:
        """Mock register a downloaded model"""
        
        # Find model data
        model_data = None
        for model in self.mock_models:
            if model["repo_id"] == repo_id:
                model_data = model
                break
        
        if not model_data:
            return None
        
        # Calculate properties
        model_hash = self.model_manager.get_model_hash(local_path)
        model_size = self.model_manager.estimate_model_size(local_path)
        
        # Generate model ID
        model_id = f"mock_{repo_id.replace('/', '_')}_{model_hash[:8]}"
        
        # Register with model manager
        success = self.model_manager.register_model(
            model_id=model_id,
            name=f"Mock HF: {repo_id.split('/')[-1]}",
            size_gb=model_size,
            format="gguf",
            quantization="4bit",
            parameters=int(model_data["estimated_params_b"] * 1e9),
            ipfs_hash=f"QmMock{model_hash[:40]}",
            owner=owner_address,
            upload_block=0,
            storage_expires=storage_blocks,
            model_hash=model_hash
        )
        
        if success:
            print(f"Mock: Registered model {model_id}")
            return model_id
        else:
            return None
    
    def download_and_register_model(
        self, 
        repo_id: str,
        owner_address: str,
        filename: Optional[str] = None,
        storage_blocks: int = 1000,
        prefer_4bit: bool = True
    ) -> Optional[str]:
        """Mock download and register a model in one step"""
        
        result = self.download_gguf_model(repo_id, filename=filename, prefer_4bit=prefer_4bit)
        
        if result is None:
            return None
        
        repo_id, local_path = result
        
        return self.register_downloaded_model(repo_id, local_path, owner_address, storage_blocks)
    
    def list_available_models(self, size_category: str = "small") -> None:
        """Mock list available models"""
        print(f"\nMock available GGUF models ({size_category} size):")
        print("-" * 60)
        
        models = self.search_gguf_models(size_category, max_results=10)
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['repo_id']}")
            print(f"    Author: {model['author']}")
            print(f"    Est. Parameters: {model['estimated_params_b']:.1f}B")
            print(f"    Downloads: {model['downloads']:,}")
            print(f"    GGUF files: {len(model['gguf_files'])}")
            print(f"    Files: {', '.join(model['gguf_files'])}")
            print()
    
    def validate_model_pairing(self, repo_id: str) -> bool:
        """Mock validate that a model has quantized versions"""
        model_data = None
        for model in self.mock_models:
            if model["repo_id"] == repo_id:
                model_data = model
                break
        
        if not model_data:
            return False
        
        # Check if any files have quantization indicators
        gguf_files = model_data["gguf_files"]
        has_quantized = any(
            any(q in f for q in ['Q4', 'Q8', 'q4', 'q8'])
            for f in gguf_files
        )
        
        return has_quantized


def create_hf_integration(model_manager: ModelManager, use_mock: bool = True) -> object:
    """Factory function to create HF integration (real or mock)"""
    
    if use_mock:
        return MockHuggingFaceIntegration(model_manager)
    else:
        # Check if we can import real HF integration
        try:
            from .hf_integration import HuggingFaceIntegration
            return HuggingFaceIntegration(model_manager)
        except ImportError as e:
            print(f"Warning: Could not import real HF integration: {e}")
            print("Falling back to mock integration")
            return MockHuggingFaceIntegration(model_manager)
