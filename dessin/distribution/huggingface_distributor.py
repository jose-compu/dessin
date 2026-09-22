"""
HuggingFace Distributor for DeSSIN

Handles downloading models from HuggingFace Hub, uploading specific models
for user convenience (not blockchain transactions), and pushing periodic
trained-model versions to HuggingFace as PoGO training progresses
(``HuggingFaceUpdater``, formerly ``dessin/hf_updater.py``).
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import asyncio

try:
    from huggingface_hub import hf_hub_download, upload_file, HfApi, create_repo
    from huggingface_hub.utils import HfHubHTTPError
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

    class HfHubHTTPError(Exception):  # type: ignore[no-redef]
        """Fallback when ``huggingface_hub`` is not installed."""

from ..models.model_manager import ModelManager, ModelInfo
from ..runtime.progress_tracker import track_upload

@dataclass
class HFUploadResult:
    """Result of uploading a model to HuggingFace"""
    success: bool
    model_id: str
    hf_repo_name: Optional[str] = None
    hf_url: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: float = 0.0
    file_size_mb: float = 0.0

class HuggingFaceDistributor:
    """Distributes models via HuggingFace Hub"""
    
    def __init__(self, model_manager: ModelManager, force_mock: bool = False):
        self.model_manager = model_manager
        self.force_mock = force_mock
        self.api = HfApi() if HUGGINGFACE_AVAILABLE and not force_mock else None
        self.download_cache = {}
        self.upload_history = []
        
        # Mock mode for testing
        if not HUGGINGFACE_AVAILABLE or force_mock:
            print("⚠️  Using mock HuggingFace distributor")
    
    def is_available(self) -> bool:
        """Check if HuggingFace is available"""
        return HUGGINGFACE_AVAILABLE and not self.force_mock
    
    async def download_model(
        self, 
        model_id: str, 
        save_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str], float]:
        """Download a model from HuggingFace Hub"""
        start_time = time.time()
        
        if not self.is_available():
            return await self._download_mock_model(model_id, save_path)
        
        try:
            # Use provided save path or default to cache
            if not save_path:
                save_path = self.model_manager.config.model_cache_dir
            
            # Create directory if it doesn't exist
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Download from HuggingFace
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=f"{model_id}.gguf",
                cache_dir=save_path,
                local_dir=save_path,
                local_dir_use_symlinks=False
            )
            
            download_time = time.time() - start_time
            
            # Register with model manager
            self.model_manager.register_model(
                model_id=model_id,
                name=model_id,
                owner="hf",
                ipfs_hash="",
                upload_block=0,
                storage_expires=10**9,
                model_path=local_path,
                size_gb=Path(local_path).stat().st_size / (1024**3),
                format="gguf",
                quantization="q4_0",
                parameters=7,  # Default assumption
                model_hash=self._calculate_file_hash(local_path)
            )
            
            print(f"✓ Downloaded {model_id} from HuggingFace")
            return True, local_path, download_time
            
        except Exception as e:
            print(f"❌ Failed to download {model_id} from HuggingFace: {e}")
            return False, None, time.time() - start_time
    
    async def upload_model_to_hf(
        self,
        model_id: str,
        hf_repo_name: str,
        model_path: Optional[str] = None,
        description: str = "",
        tags: List[str] = None,
        private: bool = False
    ) -> HFUploadResult:
        """
        Upload a specific model to HuggingFace Hub for user convenience.
        This is NOT a blockchain transaction - just a convenience command.
        """
        start_time = time.time()
        
        if not self.is_available():
            # In mock mode, if model not found locally and no custom path, return failure
            model_info = self.model_manager.get_model_info(model_id)
            if model_path is None and not model_info:
                return HFUploadResult(
                    success=False,
                    model_id=model_id,
                    hf_repo_name=hf_repo_name,
                    error_message=f"Model {model_id} not found locally"
                )
            return await self._upload_mock_model(model_id, hf_repo_name)
        
        try:
            # Get model path
            if not model_path:
                model_info = self.model_manager.get_model_info(model_id)
                if not model_info:
                    return HFUploadResult(
                        success=False,
                        model_id=model_id,
                        error_message=f"Model {model_id} not found locally"
                    )
                model_path = model_info.model_path
            
            if not Path(model_path).exists():
                return HFUploadResult(
                    success=False,
                    model_id=model_id,
                    error_message=f"Model file not found at {model_path}"
                )
            
            # Calculate file size
            file_size_bytes = Path(model_path).stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Start progress tracking
            operation_id = track_upload(
                f"Uploading {model_id} to HF",
                file_size_mb
            )
            
            # Upload to HuggingFace
            upload_result = upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"{model_id}.gguf",
                repo_id=hf_repo_name,
                repo_type="model",
                token=os.getenv("HF_TOKEN"),
                private=private,
                commit_message=f"Upload {model_id} via DeSSIN",
                commit_description=description
            )
            
            # Add tags if provided
            if tags:
                try:
                    self.api.add_model_tags(hf_repo_name, tags)
                except Exception as e:
                    print(f"Warning: Could not add tags: {e}")
            
            upload_time = time.time() - start_time
            
            # Complete progress tracking
            from ..runtime.progress_tracker import complete_operation
            complete_operation(operation_id)
            
            result = HFUploadResult(
                success=True,
                model_id=model_id,
                hf_repo_name=hf_repo_name,
                hf_url=f"https://huggingface.co/{hf_repo_name}",
                upload_time=upload_time,
                file_size_mb=file_size_mb
            )
            
            self.upload_history.append(result)
            print(f"✓ Successfully uploaded {model_id} to {hf_repo_name}")
            print(f"  URL: {result.hf_url}")
            print(f"  Size: {file_size_mb:.1f} MB")
            print(f"  Time: {upload_time:.1f}s")
            
            return result
            
        except Exception as e:
            upload_time = time.time() - start_time
            result = HFUploadResult(
                success=False,
                model_id=model_id,
                hf_repo_name=hf_repo_name,
                error_message=str(e),
                upload_time=upload_time
            )
            
            self.upload_history.append(result)
            print(f"❌ Failed to upload {model_id} to HuggingFace: {e}")
            return result
    
    def list_uploaded_models(self) -> List[HFUploadResult]:
        """List all models uploaded via this distributor"""
        return self.upload_history.copy()
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics"""
        if not self.upload_history:
            return {"total_uploads": 0, "successful_uploads": 0, "failed_uploads": 0, "total_size_mb": 0, "average_upload_time": 0}
        
        successful = [r for r in self.upload_history if r.success]
        total_size = sum(r.file_size_mb for r in successful)
        
        return {
            "total_uploads": len(self.upload_history),
            "successful_uploads": len(successful),
            "failed_uploads": len(self.upload_history) - len(successful),
            "total_size_mb": total_size,
            "average_upload_time": sum(r.upload_time for r in successful) / len(successful) if successful else 0
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old cached downloads"""
        if not self.is_available():
            # In mock mode, operate on the configured cache dir
            cache_dir = Path(self.model_manager.config.model_cache_dir)
            if not cache_dir.exists():
                return 0
            cutoff_time = time.time() - (max_age_hours * 3600)
            removed_count = 0
            for file_path in cache_dir.rglob("*.gguf"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            return removed_count
        
        try:
            cache_dir = Path(self.model_manager.config.model_cache_dir)
            if not cache_dir.exists():
                return 0
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            removed_count = 0
            
            for file_path in cache_dir.rglob("*.gguf"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            
            print(f"✓ Cleaned up {removed_count} old cached files")
            return removed_count
            
        except Exception as e:
            print(f"❌ Error cleaning cache: {e}")
            return 0
    
    async def _download_mock_model(
        self, 
        model_id: str, 
        save_path: Optional[str]
    ) -> Tuple[bool, Optional[str], float]:
        """Mock download for testing"""
        start_time = time.time()
        
        if not save_path:
            save_path = self.model_manager.config.model_cache_dir
        
        # Create mock file
        mock_file = Path(save_path) / f"{model_id}.gguf"
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create mock content
        content = f"Mock model: {model_id}\n"
        content += f"Created at: {time.time()}\n"
        content += "x" * 1024  # 1KB mock data
        
        with open(mock_file, "w") as f:
            f.write(content)
        
        download_time = time.time() - start_time
        
        # Register with model manager
        self.model_manager.register_model(
            model_id=model_id,
            name=model_id,
            owner="hf",
            ipfs_hash="",
            upload_block=0,
            storage_expires=10**9,
            model_path=str(mock_file),
            size_gb=len(content) / (1024**3),
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash=self._calculate_file_hash(str(mock_file))
        )
        
        print(f"✓ Mock downloaded {model_id}")
        return True, str(mock_file), download_time
    
    async def _upload_mock_model(
        self, 
        model_id: str, 
        hf_repo_name: str
    ) -> HFUploadResult:
        """Mock upload for testing"""
        start_time = time.time()
        
        # Simulate upload delay
        await asyncio.sleep(0.1)
        
        upload_time = time.time() - start_time
        
        result = HFUploadResult(
            success=True,
            model_id=model_id,
            hf_repo_name=hf_repo_name,
            hf_url=f"https://huggingface.co/{hf_repo_name}",
            upload_time=upload_time,
            file_size_mb=1.0  # Mock size
        )
        
        self.upload_history.append(result)
        print(f"✓ Mock uploaded {model_id} to {hf_repo_name}")
        return result
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

# Micro-models for testing
MICRO_MODELS = {
    "microsoft/DialoGPT-small": {
        "size_mb": 1.2,
        "parameters": 117,
        "format": "gguf",
        "quantization": "q4_0"
    },
    "distilbert-base-uncased": {
        "size_mb": 0.8,
        "parameters": 66,
        "format": "gguf", 
        "quantization": "q4_0"
    },
    "gpt2": {
        "size_mb": 2.1,
        "parameters": 124,
        "format": "gguf",
        "quantization": "q4_0"
    }
}

def get_micro_model_config(model_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a micro-model"""
    return MICRO_MODELS.get(model_id)

def list_available_micro_models() -> List[str]:
    """List available micro-models for testing"""
    return list(MICRO_MODELS.keys())
