"""
Tests for DeSSIN CLI interface
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from click.testing import CliRunner

from dessin.runtime.cli import cli
from dessin.models.model_manager import ModelManager, ModelConfig


class TestCLI:
    """Test CLI functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig(model_cache_dir=self.temp_dir)
        self.model_manager = ModelManager(self.config)
        self.runner = CliRunner()
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "DeSSIN - Decentralized AI Model Distribution System" in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_model_list_empty(self):
        """Test listing models when none exist"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock list_available_local_models to return empty list
            with patch('dessin.runtime.cli.list_available_local_models', return_value=[]):
                result = self.runner.invoke(cli, ['model', 'list'])
                
                assert result.exit_code == 0
                assert "No models found locally" in result.output
    
    def test_model_list_with_models(self):
        """Test listing models when models exist"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock models
            mock_models = [
                {
                    "model_id": "test-model-1",
                    "name": "Test Model 1",
                    "size_gb": 1.5,
                    "format": "gguf",
                    "parameters": 7000000000,
                    "available": True
                },
                {
                    "model_id": "test-model-2",
                    "name": "Test Model 2",
                    "size_gb": 2.0,
                    "format": "gguf",
                    "parameters": 13000000000,
                    "available": False
                }
            ]
            
            with patch('dessin.runtime.cli.list_available_local_models', return_value=mock_models):
                result = self.runner.invoke(cli, ['model', 'list'])
                
                assert result.exit_code == 0
                assert "test-model-1" in result.output
                assert "test-model-2" in result.output
                assert "1.50" in result.output  # Size formatting
                assert "✓" in result.output  # Available indicator
                assert "✗" in result.output  # Unavailable indicator
    
    def test_model_info_found(self):
        """Test getting model info when model exists"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock model info
            mock_model_info = Mock()
            mock_model_info.name = "Test Model"
            mock_model_info.size_gb = 1.5
            mock_model_info.format = "gguf"
            mock_model_info.quantization = "q4_0"
            mock_model_info.parameters = 7000000000
            mock_model_info.model_path = "/path/to/model.gguf"
            mock_manager.get_model_info.return_value = mock_model_info
            
            with patch('pathlib.Path.exists', return_value=True):
                result = self.runner.invoke(cli, ['model', 'info', 'test-model'])
                
                assert result.exit_code == 0
                assert "Test Model" in result.output
                assert "1.50 GB" in result.output
                assert "gguf" in result.output
                assert "q4_0" in result.output
    
    def test_model_info_not_found(self):
        """Test getting model info when model doesn't exist"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.get_model_info.return_value = None
            
            result = self.runner.invoke(cli, ['model', 'info', 'non-existent-model'])
            
            assert result.exit_code == 0
            assert "not found" in result.output
    
    @pytest.mark.asyncio
    async def test_query_run_success(self):
        """Test running a model query successfully"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock model info
            mock_model_info = Mock()
            mock_model_info.model_path = "/path/to/model.gguf"
            mock_manager.get_model_info.return_value = mock_model_info
            
            # Mock local model runner
            mock_response = Mock()
            mock_response.response = "Hello! How can I help you?"
            mock_response.prompt_tokens = 5
            mock_response.response_tokens = 8
            mock_response.total_tokens = 13
            mock_response.inference_time = 1.5
            mock_response.model_type.value = "gguf"
            
            with patch('dessin.runtime.cli.LocalModelRunner') as mock_runner_class:
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner
                mock_runner.query_model = AsyncMock(return_value=mock_response)
                
                result = self.runner.invoke(cli, [
                    'query', 'run', 'test-model', 'Hello, world!'
                ])
                
                assert result.exit_code == 0
                assert "Hello! How can I help you?" in result.output
                assert "Tokens: 5 → 8 (total: 13)" in result.output
                assert "1.50s" in result.output
    
    @pytest.mark.asyncio
    async def test_query_run_model_not_found(self):
        """Test running query with non-existent model"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.get_model_info.return_value = None
            
            result = self.runner.invoke(cli, [
                'query', 'run', 'non-existent-model', 'Hello, world!'
            ])
            
            assert result.exit_code == 0
            assert "not found locally" in result.output
    
    @pytest.mark.asyncio
    async def test_query_run_with_output_file(self):
        """Test running query and saving to output file"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock model info
            mock_model_info = Mock()
            mock_model_info.model_path = "/path/to/model.gguf"
            mock_manager.get_model_info.return_value = mock_model_info
            
            # Mock response
            mock_response = Mock()
            mock_response.response = "Test response"
            mock_response.prompt_tokens = 3
            mock_response.response_tokens = 5
            mock_response.total_tokens = 8
            mock_response.inference_time = 1.0
            mock_response.model_type.value = "gguf"
            mock_response.timestamp = 1234567890.0
            mock_response.model_id = "test-model"
            
            with patch('dessin.runtime.cli.LocalModelRunner') as mock_runner_class:
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner
                mock_runner.query_model = AsyncMock(return_value=mock_response)
                
                # Create temporary output file
                output_file = Path(self.temp_dir) / "output.json"
                
                result = self.runner.invoke(cli, [
                    'query', 'run', 'test-model', 'Test prompt',
                    '--output', str(output_file)
                ])
                
                assert result.exit_code == 0
                assert "Response saved to" in result.output
                
                # Check output file was created
                assert output_file.exists()
                
                # Check JSON content
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    assert data["model_id"] == "test-model"
                    assert data["prompt"] == "Test prompt"
                    assert data["response"] == "Test response"
                    assert data["metrics"]["total_tokens"] == 8
    
    @pytest.mark.asyncio
    async def test_query_interactive_no_models(self):
        """Test interactive query with no available models"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            with patch('dessin.runtime.cli.list_available_local_models', return_value=[]):
                result = self.runner.invoke(cli, ['query', 'interactive'])
                
                assert result.exit_code == 0
                assert "No models available" in result.output
    
    @pytest.mark.asyncio
    async def test_download_model_success(self):
        """Test downloading model successfully"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock hybrid distributor
            mock_result = {
                "success": True,
                "local_path": "/path/to/downloaded/model.gguf",
                "download_time": 45.2
            }
            
            with patch('dessin.runtime.cli.HybridDistributor') as mock_distributor_class:
                mock_distributor = Mock()
                mock_distributor_class.return_value = mock_distributor
                mock_distributor.download_model = AsyncMock(return_value=mock_result)
                
                result = self.runner.invoke(cli, [
                    'download', 'model', 'test-model'
                ])
                
                assert result.exit_code == 0
                assert "downloaded successfully" in result.output
                assert "45.20s" in result.output
    
    @pytest.mark.asyncio
    async def test_download_model_failure(self):
        """Test downloading model failure"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock failed download
            mock_result = {
                "success": False,
                "error_message": "Model not found"
            }
            
            with patch('dessin.runtime.cli.HybridDistributor') as mock_distributor_class:
                mock_distributor = Mock()
                mock_distributor_class.return_value = mock_distributor
                mock_distributor.download_model = AsyncMock(return_value=mock_result)
                
                result = self.runner.invoke(cli, [
                    'download', 'model', 'test-model'
                ])
                
                assert result.exit_code == 0
                assert "Failed to download" in result.output
    
    @pytest.mark.asyncio
    async def test_upload_hf_success(self):
        """Test uploading model to HuggingFace successfully"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock HF upload result
            mock_result = Mock()
            mock_result.success = True
            mock_result.hf_url = "https://huggingface.co/test-user/test-repo"
            mock_result.file_size_mb = 1500.5
            mock_result.upload_time = 120.3
            
            with patch('dessin.runtime.cli.HuggingFaceDistributor') as mock_distributor_class:
                mock_distributor = Mock()
                mock_distributor_class.return_value = mock_distributor
                mock_distributor.upload_model_to_hf = AsyncMock(return_value=mock_result)
                
                result = self.runner.invoke(cli, [
                    'upload', 'hf', 'test-model', 'test-user/test-repo'
                ])
                
                assert result.exit_code == 0
                assert "uploaded to test-user/test-repo" in result.output
                assert "1500.5 MB" in result.output
                assert "120.30s" in result.output
    
    @pytest.mark.asyncio
    async def test_upload_hf_failure(self):
        """Test uploading model to HuggingFace failure"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock failed upload
            mock_result = Mock()
            mock_result.success = False
            mock_result.error_message = "Authentication failed"
            
            with patch('dessin.runtime.cli.HuggingFaceDistributor') as mock_distributor_class:
                mock_distributor = Mock()
                mock_distributor_class.return_value = mock_distributor
                mock_distributor.upload_model_to_hf = AsyncMock(return_value=mock_result)
                
                result = self.runner.invoke(cli, [
                    'upload', 'hf', 'test-model', 'test-user/test-repo'
                ])
                
                assert result.exit_code == 0
                assert "Failed to upload model" in result.output
                assert "Authentication failed" in result.output
    
    def test_status_command(self):
        """Test status command"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock available models
            mock_models = [
                {
                    "model_id": "test-model",
                    "name": "Test Model",
                    "size_gb": 1.5,
                    "format": "gguf",
                    "parameters": 7000000000,
                    "available": True
                }
            ]
            
            # Mock local model runner
            mock_runner = Mock()
            mock_runner.loaded_models = {"test-model": {}}
            mock_runner.get_all_metrics.return_value = {}
            mock_runner.get_memory_usage.return_value = {
                "loaded_models": 1,
                "gpu_memory": None,
                "query_history_size": 0
            }
            mock_runner.get_query_history.return_value = []
            
            with patch('dessin.runtime.cli.list_available_local_models', return_value=mock_models), \
                 patch('dessin.runtime.cli.LocalModelRunner', return_value=mock_runner):
                
                result = self.runner.invoke(cli, ['status'])
                
                assert result.exit_code == 0
                assert "DeSSIN System Status" in result.output
                assert "Local Models: 1" in result.output
                assert "Loaded Models: 1" in result.output
    
    def test_cli_with_config_option(self):
        """Test CLI with config option"""
        result = self.runner.invoke(cli, ['--config', '/path/to/config.json', '--help'])
        assert result.exit_code == 0
        assert "DeSSIN - Decentralized AI Model Distribution System" in result.output
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command"""
        result = self.runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0
    
    def test_model_commands_help(self):
        """Test model commands help"""
        result = self.runner.invoke(cli, ['model', '--help'])
        assert result.exit_code == 0
        assert "Model management commands" in result.output
    
    def test_query_commands_help(self):
        """Test query commands help"""
        result = self.runner.invoke(cli, ['query', '--help'])
        assert result.exit_code == 0
        assert "Local model query commands" in result.output
    
    def test_download_commands_help(self):
        """Test download commands help"""
        result = self.runner.invoke(cli, ['download', '--help'])
        assert result.exit_code == 0
        assert "Model download commands" in result.output
    
    def test_upload_commands_help(self):
        """Test upload commands help"""
        result = self.runner.invoke(cli, ['upload', '--help'])
        assert result.exit_code == 0
        assert "Model upload commands" in result.output
    
    def test_query_run_with_options(self):
        """Test query run with various options"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock model info
            mock_model_info = Mock()
            mock_model_info.model_path = "/path/to/model.gguf"
            mock_manager.get_model_info.return_value = mock_model_info
            
            # Mock response
            mock_response = Mock()
            mock_response.response = "Test response"
            mock_response.prompt_tokens = 3
            mock_response.response_tokens = 5
            mock_response.total_tokens = 8
            mock_response.inference_time = 1.0
            mock_response.model_type.value = "gguf"
            
            with patch('dessin.runtime.cli.LocalModelRunner') as mock_runner_class:
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner
                mock_runner.query_model = AsyncMock(return_value=mock_response)
                
                result = self.runner.invoke(cli, [
                    'query', 'run', 'test-model', 'Test prompt',
                    '--max-tokens', '256',
                    '--temperature', '0.8',
                    '--top-p', '0.9',
                    '--system-prompt', 'You are a helpful assistant.'
                ])
                
                assert result.exit_code == 0
                assert "Test response" in result.output
    
    def test_upload_hf_with_options(self):
        """Test upload hf with various options"""
        with patch('dessin.runtime.cli.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock successful upload
            mock_result = Mock()
            mock_result.success = True
            mock_result.hf_url = "https://huggingface.co/test-user/test-repo"
            mock_result.file_size_mb = 1500.5
            mock_result.upload_time = 120.3
            
            with patch('dessin.runtime.cli.HuggingFaceDistributor') as mock_distributor_class:
                mock_distributor = Mock()
                mock_distributor_class.return_value = mock_distributor
                mock_distributor.upload_model_to_hf = AsyncMock(return_value=mock_result)
                
                result = self.runner.invoke(cli, [
                    'upload', 'hf', 'test-model', 'test-user/test-repo',
                    '--description', 'A test model for DeSSIN',
                    '--private'
                ])
                
                assert result.exit_code == 0
                assert "uploaded to test-user/test-repo" in result.output
