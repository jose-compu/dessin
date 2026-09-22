"""
Tests for Local Model Runner
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from dessin.nanochat.local_model_runner import (
    LocalModelRunner,
    QueryRequest,
    QueryResponse,
    ModelMetrics,
    ModelType,
    query_local_model,
    list_available_local_models
)
from dessin.models.model_manager import ModelManager, ModelConfig

class TestLocalModelRunner:
    """Test Local Model Runner functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig(model_cache_dir=self.temp_dir)
        self.model_manager = ModelManager(self.config)
        self.runner = LocalModelRunner(self.model_manager)
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test runner initialization"""
        assert self.runner.model_manager == self.model_manager
        assert len(self.runner.loaded_models) == 0
        assert len(self.runner.model_metrics) == 0
        assert len(self.runner.query_history) == 0
        assert self.runner.max_history_size == 1000
    
    def test_model_type_detection(self):
        """Test model type detection"""
        # Test GGUF detection
        gguf_path = "/path/to/model.gguf"
        assert self.runner.get_model_type(gguf_path) == ModelType.GGUF
        
        # Test Transformers detection
        bin_path = "/path/to/model.bin"
        assert self.runner.get_model_type(bin_path) == ModelType.TRANSFORMERS
        
        safetensors_path = "/path/to/model.safetensors"
        assert self.runner.get_model_type(safetensors_path) == ModelType.TRANSFORMERS
        
        # Test unknown type
        unknown_path = "/path/to/model.unknown"
        assert self.runner.get_model_type(unknown_path) == ModelType.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_load_model_not_found(self):
        """Test loading non-existent model"""
        success = await self.runner.load_model("non_existent_model")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_load_model_file_not_found(self):
        """Test loading model with missing file"""
        # Register model but don't create file
        self.model_manager.register_model(
            model_id="missing_file_model",
            name="Missing File Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path="/non/existent/path/model.gguf",
            size_gb=1.0,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
        
        success = await self.runner.load_model("missing_file_model")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_load_gguf_model_mock(self):
        """Test loading GGUF model with mock llama-cpp"""
        # Create mock model file
        model_path = Path(self.temp_dir) / "test_model.gguf"
        with open(model_path, "w") as f:
            f.write("Mock GGUF model content")
        
        # Register model
        self.model_manager.register_model(
            model_id="test_gguf_model",
            name="Test GGUF Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_path),
            size_gb=1.0,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Mock llama-cpp
        mock_llama = Mock()
        mock_llama.return_value = {
            "choices": [{"text": "Mock response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        with patch('dessin.nanochat.local_model_runner.LLAMA_CPP_AVAILABLE', True), \
             patch('dessin.nanochat.local_model_runner.Llama', return_value=mock_llama):
            
            success = await self.runner.load_model("test_gguf_model")
            assert success is True
            assert "test_gguf_model" in self.runner.loaded_models
    
    @pytest.mark.asyncio
    async def test_load_transformers_model_mock(self):
        """Test loading Transformers model with mock"""
        # Create mock model directory
        model_dir = Path(self.temp_dir) / "test_transformers_model"
        model_dir.mkdir()
        config_file = model_dir / "config.json"
        with open(config_file, "w") as f:
            f.write('{"model_type": "gpt2"}')
        
        # Register model
        self.model_manager.register_model(
            model_id="test_transformers_model",
            name="Test Transformers Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_dir),
            size_gb=1.0,
            format="transformers",
            quantization="fp16",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Mock transformers
        mock_tokenizer = Mock()
        mock_model = Mock()
        
        with patch('dessin.nanochat.local_model_runner.TORCH_AVAILABLE', True), \
             patch('dessin.nanochat.local_model_runner.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
             patch('dessin.nanochat.local_model_runner.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
            
            success = await self.runner.load_model("test_transformers_model")
            assert success is True
            assert "test_transformers_model" in self.runner.loaded_models
    
    @pytest.mark.asyncio
    async def test_query_model_not_loaded(self):
        """Test querying model that's not loaded"""
        request = QueryRequest(
            model_id="test_model",
            prompt="Hello, world!"
        )
        
        response = await self.runner.query_model(request)
        assert response is None
    
    @pytest.mark.asyncio
    async def test_query_gguf_model_mock(self):
        """Test querying GGUF model with mock"""
        # Setup mock model
        model_path = Path(self.temp_dir) / "test_model.gguf"
        with open(model_path, "w") as f:
            f.write("Mock GGUF model content")
        
        self.model_manager.register_model(
            model_id="test_model",
            name="Test Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_path),
            size_gb=1.0,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Mock llama-cpp
        mock_llama = Mock()
        mock_llama.return_value = {
            "choices": [{"text": "Hello! How can I help you?"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
        }
        
        with patch('dessin.nanochat.local_model_runner.LLAMA_CPP_AVAILABLE', True), \
             patch('dessin.nanochat.local_model_runner.Llama', return_value=mock_llama):
            
            # Load model
            await self.runner.load_model("test_model")
            
            # Query model
            request = QueryRequest(
                model_id="test_model",
                prompt="Hello, world!"
            )
            
            response = await self.runner.query_model(request)
            
            assert response is not None
            assert response.model_id == "test_model"
            assert response.response == "Hello! How can I help you?"
            assert response.model_type == ModelType.GGUF
            assert response.prompt_tokens == 5
            assert response.response_tokens == 8
            assert response.total_tokens == 13
            assert response.inference_time > 0
    
    @pytest.mark.asyncio
    async def test_query_transformers_model_mock(self):
        """Test querying Transformers model with mock"""
        # Setup mock model
        model_dir = Path(self.temp_dir) / "test_model"
        model_dir.mkdir()
        config_file = model_dir / "config.json"
        with open(config_file, "w") as f:
            f.write('{"model_type": "gpt2"}')
        
        self.model_manager.register_model(
            model_id="test_model",
            name="Test Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_dir),
            size_gb=1.0,
            format="transformers",
            quantization="fp16",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Mock transformers
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 5])}
        mock_tokenizer.decode.return_value = "Mock response"
        
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.__getitem__ = Mock(return_value=Mock())
        mock_outputs.__getitem__().__getitem__ = Mock(return_value=[1, 2, 3, 4, 5])
        mock_model.generate.return_value = mock_outputs
        
        with patch('dessin.nanochat.local_model_runner.TORCH_AVAILABLE', True), \
             patch('dessin.nanochat.local_model_runner.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
             patch('dessin.nanochat.local_model_runner.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('dessin.nanochat.local_model_runner.torch.no_grad'):
            
            # Load model
            await self.runner.load_model("test_model")
            
            # Query model
            request = QueryRequest(
                model_id="test_model",
                prompt="Hello, world!"
            )
            
            response = await self.runner.query_model(request)
            
            assert response is not None
            assert response.model_id == "test_model"
            assert response.response == "Mock response"
            assert response.model_type == ModelType.TRANSFORMERS
            assert response.inference_time > 0
    
    def test_update_metrics(self):
        """Test metrics update"""
        # Create mock response
        response = QueryResponse(
            model_id="test_model",
            response="Test response",
            prompt_tokens=10,
            response_tokens=5,
            total_tokens=15,
            inference_time=1.5,
            model_type=ModelType.GGUF,
            metadata={},
            timestamp=time.time()
        )
        
        # Update metrics
        self.runner._update_metrics("test_model", response)
        
        # Check metrics
        metrics = self.runner.model_metrics["test_model"]
        assert metrics.model_id == "test_model"
        assert metrics.total_queries == 1
        assert metrics.total_tokens == 15
        assert metrics.average_inference_time == 1.5
        assert metrics.total_inference_time == 1.5
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0
    
    def test_update_error_metrics(self):
        """Test error metrics update"""
        # Initialize metrics
        self.runner.model_metrics["test_model"] = ModelMetrics(
            model_id="test_model",
            total_queries=10,
            total_tokens=100,
            average_inference_time=1.0,
            total_inference_time=10.0,
            last_used=time.time(),
            error_count=1,
            success_rate=0.9
        )
        
        # Update error metrics
        self.runner._update_error_metrics("test_model")
        
        # Check updated metrics
        metrics = self.runner.model_metrics["test_model"]
        assert metrics.error_count == 2
        assert metrics.success_rate == 0.8  # (10-2)/10
    
    def test_list_loaded_models(self):
        """Test listing loaded models"""
        # Add mock loaded models
        self.runner.loaded_models["model1"] = {"type": ModelType.GGUF}
        self.runner.loaded_models["model2"] = {"type": ModelType.TRANSFORMERS}
        
        loaded_models = self.runner.list_loaded_models()
        assert len(loaded_models) == 2
        assert "model1" in loaded_models
        assert "model2" in loaded_models
    
    def test_get_model_metrics(self):
        """Test getting model metrics"""
        # Add mock metrics
        metrics = ModelMetrics(
            model_id="test_model",
            total_queries=10,
            total_tokens=100,
            average_inference_time=1.0,
            total_inference_time=10.0,
            last_used=time.time(),
            error_count=1,
            success_rate=0.9
        )
        self.runner.model_metrics["test_model"] = metrics
        
        retrieved_metrics = self.runner.get_model_metrics("test_model")
        assert retrieved_metrics == metrics
        
        # Test non-existent model
        assert self.runner.get_model_metrics("non_existent") is None
    
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        # Add multiple metrics
        self.runner.model_metrics["model1"] = ModelMetrics(
            model_id="model1",
            total_queries=5,
            total_tokens=50,
            average_inference_time=1.0,
            total_inference_time=5.0,
            last_used=time.time(),
            error_count=0,
            success_rate=1.0
        )
        self.runner.model_metrics["model2"] = ModelMetrics(
            model_id="model2",
            total_queries=10,
            total_tokens=100,
            average_inference_time=1.0,
            total_inference_time=10.0,
            last_used=time.time(),
            error_count=1,
            success_rate=0.9
        )
        
        all_metrics = self.runner.get_all_metrics()
        assert len(all_metrics) == 2
        assert "model1" in all_metrics
        assert "model2" in all_metrics
    
    def test_query_history_management(self):
        """Test query history management"""
        # Add mock responses to history
        response1 = QueryResponse(
            model_id="model1",
            response="Response 1",
            prompt_tokens=5,
            response_tokens=3,
            total_tokens=8,
            inference_time=1.0,
            model_type=ModelType.GGUF,
            metadata={},
            timestamp=time.time()
        )
        response2 = QueryResponse(
            model_id="model2",
            response="Response 2",
            prompt_tokens=10,
            response_tokens=5,
            total_tokens=15,
            inference_time=2.0,
            model_type=ModelType.TRANSFORMERS,
            metadata={},
            timestamp=time.time()
        )
        
        self.runner.query_history = [response1, response2]
        
        # Test getting history with limit
        history = self.runner.get_query_history(limit=1)
        assert len(history) == 1
        assert history[0] == response2
        
        # Test getting all history
        all_history = self.runner.get_query_history()
        assert len(all_history) == 2
        
        # Test clearing history
        self.runner.clear_history()
        assert len(self.runner.query_history) == 0
    
    def test_unload_model(self):
        """Test unloading model"""
        # Add mock loaded model
        self.runner.loaded_models["test_model"] = {
            "model": Mock(),
            "type": ModelType.GGUF,
            "path": "/path/to/model"
        }
        
        # Unload model
        success = self.runner.unload_model("test_model")
        assert success is True
        assert "test_model" not in self.runner.loaded_models
        
        # Test unloading non-existent model
        success = self.runner.unload_model("non_existent")
        assert success is False
    
    def test_unload_all_models(self):
        """Test unloading all models"""
        # Add multiple loaded models
        self.runner.loaded_models["model1"] = {"type": ModelType.GGUF}
        self.runner.loaded_models["model2"] = {"type": ModelType.TRANSFORMERS}
        
        # Unload all models
        self.runner.unload_all_models()
        assert len(self.runner.loaded_models) == 0
    
    def test_memory_usage(self):
        """Test memory usage reporting"""
        # Add mock loaded models
        self.runner.loaded_models["model1"] = {"type": ModelType.GGUF}
        self.runner.loaded_models["model2"] = {"type": ModelType.TRANSFORMERS}
        
        memory_usage = self.runner.get_memory_usage()
        assert memory_usage["loaded_models"] == 2
        assert memory_usage["query_history_size"] == 0
    
    @pytest.mark.asyncio
    async def test_query_local_model_convenience(self):
        """Test convenience function for local model queries"""
        # Setup mock model
        model_path = Path(self.temp_dir) / "test_model.gguf"
        with open(model_path, "w") as f:
            f.write("Mock GGUF model content")
        
        self.model_manager.register_model(
            model_id="test_model",
            name="Test Model",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(model_path),
            size_gb=1.0,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="test_hash"
        )
        
        # Mock llama-cpp
        mock_llama = Mock()
        mock_llama.return_value = {
            "choices": [{"text": "Convenience response"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
        }
        
        with patch('dessin.nanochat.local_model_runner.LLAMA_CPP_AVAILABLE', True), \
             patch('dessin.nanochat.local_model_runner.Llama', return_value=mock_llama):
            
            response = await query_local_model(
                self.model_manager,
                "test_model",
                "Hello, world!",
                max_tokens=256,
                temperature=0.8
            )
            
            assert response == "Convenience response"
    
    def test_list_available_local_models(self):
        """Test listing available local models"""
        # Register models
        self.model_manager.register_model(
            model_id="model1",
            name="Model 1",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(Path(self.temp_dir) / "model1.gguf"),
            size_gb=1.0,
            format="gguf",
            quantization="q4_0",
            parameters=7,
            model_hash="hash1"
        )
        
        self.model_manager.register_model(
            model_id="model2",
            name="Model 2",
            owner="test_owner",
            ipfs_hash="QmTestHash123456789",
            upload_block=1000,
            storage_expires=2000,
            model_path=str(Path(self.temp_dir) / "model2.gguf"),
            size_gb=2.0,
            format="gguf",
            quantization="q8_0",
            parameters=13,
            model_hash="hash2"
        )
        
        # Create one model file
        model1_path = Path(self.temp_dir) / "model1.gguf"
        with open(model1_path, "w") as f:
            f.write("Mock model content")
        
        models = list_available_local_models(self.model_manager)
        assert len(models) == 2
        
        # Check model1 (file exists)
        model1 = next(m for m in models if m["model_id"] == "model1")
        assert model1["available"] is True
        assert model1["size_gb"] == 1.0
        assert model1["format"] == "gguf"
        
        # Check model2 (file doesn't exist)
        model2 = next(m for m in models if m["model_id"] == "model2")
        assert model2["available"] is False
        assert model2["size_gb"] == 2.0
