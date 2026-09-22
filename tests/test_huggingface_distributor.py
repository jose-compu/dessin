"""
Tests for HuggingFace Distributor
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from dessin.distribution.huggingface_distributor import (
    HuggingFaceDistributor, 
    HFUploadResult,
    get_micro_model_config,
    list_available_micro_models
)
from dessin.models.model_manager import ModelManager, ModelConfig

class TestHuggingFaceDistributor:
    """Test HuggingFace Distributor functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig(model_cache_dir=self.temp_dir)
        self.model_manager = ModelManager(self.config)
        self.distributor = HuggingFaceDistributor(self.model_manager, force_mock=True)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test distributor initialization"""
        assert self.distributor.model_manager == self.model_manager
        assert self.distributor.force_mock is True
        assert len(self.distributor.upload_history) == 0
    
    def test_is_available(self):
        """Test availability check"""
        # Should be False when force_mock=True
        assert self.distributor.is_available() is False
        
        # Test with real HF available
        with patch('dessin.distribution.huggingface_distributor.HUGGINGFACE_AVAILABLE', True):
            real_distributor = HuggingFaceDistributor(self.model_manager, force_mock=False)
            assert real_distributor.is_available() is True
    
    @pytest.mark.asyncio
    async def test_download_model(self):
        """Test model download functionality"""
        model_id = "test-model"
        
        success, local_path, download_time = await self.distributor.download_model(model_id)
        
        assert success is True
        assert local_path is not None
        assert Path(local_path).exists()
        assert download_time >= 0
        
        # Check that model was registered
        model_info = self.model_manager.get_model_info(model_id)
        assert model_info is not None
        assert model_info.model_id == model_id
    
    @pytest.mark.asyncio
    async def test_upload_model_to_hf(self):
        """Test uploading model to HuggingFace"""
        # First create a model to upload
        model_id = "test-upload-model"
        model_path = Path(self.temp_dir) / f"{model_id}.gguf"
        
        # Create a mock model file
        with open(model_path, "w") as f:
            f.write("Mock model content for testing upload")
        
        # Register the model
        self.model_manager.register_model(
            model_id=model_id,
            name="Test Upload Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_path),
            size_gb=0.001,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Test upload
        result = await self.distributor.upload_model_to_hf(
            model_id=model_id,
            hf_repo_name="test-user/test-repo"
        )
        
        assert isinstance(result, HFUploadResult)
        assert result.success is True
        assert result.model_id == model_id
        assert result.hf_repo_name == "test-user/test-repo"
        assert result.hf_url == "https://huggingface.co/test-user/test-repo"
        assert result.upload_time > 0
        assert result.file_size_mb > 0
        
        # Check upload history
        assert len(self.distributor.upload_history) == 1
        assert self.distributor.upload_history[0] == result
    
    @pytest.mark.asyncio
    async def test_upload_model_not_found(self):
        """Test upload when model is not found"""
        result = await self.distributor.upload_model_to_hf(
            model_id="non-existent-model",
            hf_repo_name="test-user/test-repo"
        )
        
        assert result.success is False
        assert "not found locally" in result.error_message
    
    @pytest.mark.asyncio
    async def test_upload_with_custom_path(self):
        """Test upload with custom model path"""
        # Create a model file outside of model manager
        custom_path = Path(self.temp_dir) / "custom_model.gguf"
        with open(custom_path, "w") as f:
            f.write("Custom model content")
        
        result = await self.distributor.upload_model_to_hf(
            model_id="custom-model",
            hf_repo_name="test-user/custom-repo",
            model_path=str(custom_path)
        )
        
        assert result.success is True
        assert result.model_id == "custom-model"
    
    def test_list_uploaded_models(self):
        """Test listing uploaded models"""
        # Initially empty
        assert len(self.distributor.list_uploaded_models()) == 0
        
        # Add some mock uploads
        mock_upload1 = HFUploadResult(
            success=True,
            model_id="model1",
            hf_repo_name="user/repo1"
        )
        mock_upload2 = HFUploadResult(
            success=False,
            model_id="model2",
            hf_repo_name="user/repo2",
            error_message="Upload failed"
        )
        
        self.distributor.upload_history = [mock_upload1, mock_upload2]
        
        uploaded_models = self.distributor.list_uploaded_models()
        assert len(uploaded_models) == 2
        assert uploaded_models[0].model_id == "model1"
        assert uploaded_models[1].model_id == "model2"
    
    def test_get_upload_stats(self):
        """Test upload statistics"""
        # Initially empty stats
        stats = self.distributor.get_upload_stats()
        assert stats["total_uploads"] == 0
        assert stats["successful_uploads"] == 0
        assert stats["failed_uploads"] == 0
        assert stats["total_size_mb"] == 0
        
        # Add some mock uploads
        mock_upload1 = HFUploadResult(
            success=True,
            model_id="model1",
            file_size_mb=10.0,
            upload_time=5.0
        )
        mock_upload2 = HFUploadResult(
            success=True,
            model_id="model2",
            file_size_mb=20.0,
            upload_time=10.0
        )
        mock_upload3 = HFUploadResult(
            success=False,
            model_id="model3",
            upload_time=2.0
        )
        
        self.distributor.upload_history = [mock_upload1, mock_upload2, mock_upload3]
        
        stats = self.distributor.get_upload_stats()
        assert stats["total_uploads"] == 3
        assert stats["successful_uploads"] == 2
        assert stats["failed_uploads"] == 1
        assert stats["total_size_mb"] == 30.0
        assert stats["average_upload_time"] == 7.5  # (5.0 + 10.0) / 2
    
    @pytest.mark.asyncio
    async def test_cleanup_cache(self):
        """Test cache cleanup functionality"""
        # Create some mock files
        cache_dir = Path(self.temp_dir)
        old_file = cache_dir / "old_model.gguf"
        new_file = cache_dir / "new_model.gguf"
        
        with open(old_file, "w") as f:
            f.write("old content")
        with open(new_file, "w") as f:
            f.write("new content")
        
        # Mock file timestamps (old_file is older)
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        new_time = time.time() - (12 * 3600)  # 12 hours ago
        
        os.utime(old_file, (old_time, old_time))
        os.utime(new_file, (new_time, new_time))
        
        # Clean up files older than 24 hours
        removed_count = await self.distributor.cleanup_cache(max_age_hours=24)
        
        assert removed_count == 1
        assert not old_file.exists()
        assert new_file.exists()
    
    def test_micro_models_config(self):
        """Test micro-models configuration"""
        config = get_micro_model_config("microsoft/DialoGPT-small")
        assert config is not None
        assert config["size_mb"] == 1.2
        assert config["parameters"] == 117
        
        # Test non-existent model
        config = get_micro_model_config("non-existent-model")
        assert config is None
    
    def test_list_available_micro_models(self):
        """Test listing available micro-models"""
        models = list_available_micro_models()
        assert len(models) == 3
        assert "microsoft/DialoGPT-small" in models
        assert "distilbert-base-uncased" in models
        assert "gpt2" in models

# Import os for file operations
import os
