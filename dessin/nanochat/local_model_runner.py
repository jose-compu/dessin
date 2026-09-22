"""
Local Model Runner for DeSSIN

Provides local model inference capabilities without blockchain transactions.
Users can query locally stored models directly.
"""

import asyncio
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    class Llama:  # type: ignore
        pass

from ..models.model_manager import ModelManager, ModelInfo
# from .progress_tracker import track_inference  # Not implemented yet


class ModelType(Enum):
    """Supported model types"""
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    GGUF = "gguf"
    UNKNOWN = "unknown"


@dataclass
class QueryRequest:
    """Local query request"""
    model_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class QueryResponse:
    """Local query response"""
    model_id: str
    response: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    inference_time: float
    model_type: ModelType
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    total_queries: int
    total_tokens: int
    average_inference_time: float
    total_inference_time: float
    last_used: float
    error_count: int
    success_rate: float


class LocalModelRunner:
    """Local model inference runner"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.loaded_models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.query_history: List[QueryResponse] = []
        self.max_history_size = 1000
        
        # Model type detection
        self.model_type_map = {
            ".gguf": ModelType.GGUF,
            ".bin": ModelType.TRANSFORMERS,
            ".safetensors": ModelType.TRANSFORMERS,
        }
    
    def get_model_type(self, model_path: str) -> ModelType:
        """Detect model type from file extension"""
        path = Path(model_path)
        extension = path.suffix.lower()
        
        if extension in self.model_type_map:
            return self.model_type_map[extension]
        
        # Try to detect from directory structure
        if (path / "config.json").exists():
            return ModelType.TRANSFORMERS
        
        return ModelType.UNKNOWN
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """Load a model into memory for local inference"""
        if model_id in self.loaded_models and not force_reload:
            return True
        
        # Get model info
        model_info = self.model_manager.get_model_info(model_id)
        if not model_info:
            print(f"❌ Model {model_id} not found in local storage")
            return False
        
        model_path = model_info.model_path
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        model_type = self.get_model_type(model_path)
        
        try:
            if model_type == ModelType.GGUF and LLAMA_CPP_AVAILABLE:
                # Load GGUF model with llama-cpp-python
                model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                self.loaded_models[model_id] = {
                    "model": model,
                    "type": ModelType.GGUF,
                    "path": model_path
                }
                print(f"✓ Loaded GGUF model: {model_id}")
                
            elif model_type == ModelType.TRANSFORMERS and TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
                # Load transformers model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                self.loaded_models[model_id] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "type": ModelType.TRANSFORMERS,
                    "path": model_path
                }
                print(f"✓ Loaded Transformers model: {model_id}")
                
            else:
                print(f"❌ Unsupported model type or missing dependencies: {model_type}")
                return False
            
            # Initialize metrics
            if model_id not in self.model_metrics:
                self.model_metrics[model_id] = ModelMetrics(
                    model_id=model_id,
                    total_queries=0,
                    total_tokens=0,
                    average_inference_time=0.0,
                    total_inference_time=0.0,
                    last_used=time.time(),
                    error_count=0,
                    success_rate=1.0
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_id}: {e}")
            return False
    
    async def query_model(self, request: QueryRequest) -> Optional[QueryResponse]:
        """Run local inference on a model"""
        start_time = time.time()
        
        # Ensure model is loaded
        if not await self.load_model(request.model_id):
            return None
        
        model_data = self.loaded_models[request.model_id]
        model_type = model_data["type"]
        
        try:
            if model_type == ModelType.GGUF:
                response = await self._query_gguf_model(model_data, request)
            elif model_type == ModelType.TRANSFORMERS:
                response = await self._query_transformers_model(model_data, request)
            else:
                print(f"❌ Unsupported model type: {model_type}")
                return None
            
            # Update metrics
            self._update_metrics(request.model_id, response)
            
            # Add to history
            self.query_history.append(response)
            if len(self.query_history) > self.max_history_size:
                self.query_history.pop(0)
            
            return response
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            self._update_error_metrics(request.model_id)
            return None
    
    async def _query_gguf_model(self, model_data: Dict[str, Any], request: QueryRequest) -> QueryResponse:
        """Query GGUF model using llama-cpp-python"""
        model = model_data["model"]
        start_time = time.time()
        
        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        
        # Run inference
        result = model(
            full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences or [],
            echo=False
        )
        
        response_text = result["choices"][0]["text"]
        usage = result.get("usage", {})
        
        inference_time = time.time() - start_time
        
        return QueryResponse(
            model_id=request.model_id,
            response=response_text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            response_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            inference_time=inference_time,
            model_type=ModelType.GGUF,
            metadata={
                "model_path": model_data["path"],
                "user_id": request.user_id,
                "session_id": request.session_id
            },
            timestamp=time.time()
        )
    
    async def _query_transformers_model(self, model_data: Dict[str, Any], request: QueryRequest) -> QueryResponse:
        """Query Transformers model"""
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        start_time = time.time()
        
        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt")
        # Support both dict-returning mocks and HF BatchEncoding objects
        input_ids = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
        else:
            input_ids = getattr(inputs, "input_ids", None)
        # Derive prompt token count robustly for mocks
        if hasattr(input_ids, "shape") and len(getattr(input_ids, "shape", [])) > 1:
            prompt_tokens = input_ids.shape[1]
        else:
            prompt_tokens = 0
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=getattr(tokenizer, 'eos_token_id', None)
            )
        
        # Decode response (handle mocks that return simple lists)
        # Try to extract token ids compatibly with various mocks
        response_token_ids: List[int] = []
        try:
            output_ids = outputs[0]
            # slice off the prompt token ids if we know the length
            response_token_ids = output_ids[prompt_tokens:] if prompt_tokens else output_ids
        except Exception:
            response_token_ids = []
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        return QueryResponse(
            model_id=request.model_id,
            response=response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=len(response_token_ids),
            total_tokens=prompt_tokens + len(response_token_ids),
            inference_time=inference_time,
            model_type=ModelType.TRANSFORMERS,
            metadata={
                "model_path": model_data["path"],
                "user_id": request.user_id,
                "session_id": request.session_id
            },
            timestamp=time.time()
        )
    
    def _update_metrics(self, model_id: str, response: QueryResponse):
        """Update model performance metrics"""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = ModelMetrics(
                model_id=model_id,
                total_queries=0,
                total_tokens=0,
                average_inference_time=0.0,
                total_inference_time=0.0,
                last_used=time.time(),
                error_count=0,
                success_rate=1.0
            )
        
        metrics = self.model_metrics[model_id]
        metrics.total_queries += 1
        metrics.total_tokens += response.total_tokens
        metrics.total_inference_time += response.inference_time
        metrics.average_inference_time = metrics.total_inference_time / metrics.total_queries
        metrics.last_used = time.time()
        metrics.success_rate = (metrics.total_queries - metrics.error_count) / metrics.total_queries
    
    def _update_error_metrics(self, model_id: str):
        """Update error metrics"""
        if model_id in self.model_metrics:
            self.model_metrics[model_id].error_count += 1
            if self.model_metrics[model_id].total_queries > 0:
                self.model_metrics[model_id].success_rate = (
                    self.model_metrics[model_id].total_queries - 
                    self.model_metrics[model_id].error_count
                ) / self.model_metrics[model_id].total_queries
            else:
                self.model_metrics[model_id].success_rate = 0.0
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get performance metrics for a model"""
        return self.model_metrics.get(model_id)
    
    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models"""
        return self.model_metrics.copy()
    
    def get_query_history(self, limit: int = 100) -> List[QueryResponse]:
        """Get recent query history"""
        return self.query_history[-limit:] if limit > 0 else self.query_history.copy()
    
    def clear_history(self):
        """Clear query history"""
        self.query_history.clear()
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            # Clean up model resources
            model_data = self.loaded_models[model_id]
            if model_data["type"] == ModelType.TRANSFORMERS:
                # Clear CUDA cache for transformers models
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del self.loaded_models[model_id]
            print(f"✓ Unloaded model: {model_id}")
            return True
        
        return False
    
    def unload_all_models(self):
        """Unload all models from memory"""
        model_ids = list(self.loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        else:
            gpu_memory = None
        
        return {
            "loaded_models": len(self.loaded_models),
            "gpu_memory": gpu_memory,
            "query_history_size": len(self.query_history)
        }


# Convenience functions for easy usage
async def query_local_model(
    model_manager: ModelManager,
    model_id: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs
) -> Optional[str]:
    """Convenience function for simple local model queries"""
    runner = LocalModelRunner(model_manager)
    
    request = QueryRequest(
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    
    response = await runner.query_model(request)
    return response.response if response else None


def list_available_local_models(model_manager: ModelManager) -> List[Dict[str, Any]]:
    """List all locally available models"""
    models = []
    for model_id, model_info in model_manager.models.items():
        models.append({
            "model_id": model_id,
            "name": getattr(model_info, 'name', model_id),
            "size_gb": getattr(model_info, 'size_gb', 0),
            "format": getattr(model_info, 'format', 'unknown'),
            "parameters": getattr(model_info, 'parameters', 0),
            "local_path": getattr(model_info, 'model_path', ''),
            "available": Path(getattr(model_info, 'model_path', '')).exists()
        })
    return models
