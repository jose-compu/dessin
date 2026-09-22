"""
Model management for DeSSIN blockchain.
Handles GGUF model loading, quantization, and HuggingFace integration.
"""

import os
import json
import hashlib
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import gguf
except ImportError:  # optional; e2e / slim installs use HF/transformers without GGUF tooling
    gguf = None  # type: ignore

from huggingface_hub import hf_hub_download, model_info
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..runtime.config import ModelConfig


@dataclass
class ModelInfo:
    """Information about a model in the network"""
    model_id: str
    name: str
    size_gb: float
    format: str  # "gguf", "pytorch", "nanochat", etc.
    quantization: Optional[str]  # "4bit", "8bit", "16bit", "32bit"
    parameters: int  # number of parameters
    ipfs_hash: str
    owner: str
    upload_block: int
    storage_expires: int
    model_hash: str  # hash of model file
    # Optional local path to the model file, used by local runner/tests
    model_path: Optional[str] = ""
    
    # Nanochat-specific fields
    depth: Optional[int] = None  # Number of transformer layers (d20, d26, d32, etc.)
    device_batch_size: Optional[int] = None  # Batch size per device
    dataset_name: Optional[str] = None  # Training dataset name
    learning_rate: Optional[float] = None  # Learning rate
    max_training_steps: Optional[int] = None  # Maximum training steps
    
    # Storage and payment tracking
    storage_payment: float = 0.0  # Upfront storage payment
    training_payment_per_iteration: float = 0.0  # Payment per training iteration
    total_training_cost: float = 0.0  # Total training cost paid
    training_iterations_completed: int = 0  # Number of training iterations completed
    # Owner-registered training data refresh (see ModelTrainingDataRefreshTransaction)
    training_data_manifest_hash: Optional[str] = None
    training_data_uri: Optional[str] = None


@dataclass
class ModelQueryResult:
    """Result of a model query/inference"""
    query: str
    response: str
    tokens_used: int
    compute_cost: float
    success: bool
    error_message: Optional[str] = None


class ModelManager:
    """Manages models for the DeSSIN network"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.cache_dir = Path(config.model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry: model_id -> ModelInfo
        self.models: Dict[str, ModelInfo] = {}
        
        # Loaded models cache: model_id -> (model, tokenizer)
        self.loaded_models: Dict[str, Tuple[Any, Any]] = {}
        
        # Supported HuggingFace publishers for quantized models
        self.trusted_publishers = {
            "microsoft", "google", "meta-llama", "huggingface", 
            "TheBloke", "NousResearch", "teknium", "Open-Orca"
        }
        
        # Nanochat integration will be set later to avoid circular imports
        self.nanochat_integration = None
    
    def register_model(
        self, 
        model_id: str, 
        name: str, 
        owner: str,
        ipfs_hash: str,
        upload_block: int,
        storage_expires: int,
        **kwargs
    ) -> bool:
        """Register a new model in the network"""
        try:
            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                name=name,
                owner=owner,
                ipfs_hash=ipfs_hash,
                upload_block=upload_block,
                storage_expires=storage_expires,
                **kwargs
            )
            
            self.models[model_id] = model_info
            return True
            
        except Exception as e:
            print(f"Error registering model {model_id}: {e}")
            return False
    
    def download_model_from_hf(
        self, 
        repo_id: str, 
        filename: Optional[str] = None,
        force_download: bool = False
    ) -> Optional[str]:
        """Download a model from HuggingFace Hub"""
        try:
            # Get model info to check file list
            info = model_info(repo_id)
            
            # Find GGUF file if filename not specified
            if filename is None:
                gguf_files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.gguf')]
                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {repo_id}")
                
                # Prefer quantized versions (4bit)
                preferred_files = [f for f in gguf_files if any(q in f.lower() for q in ['q4', '4bit', 'q4_0', 'q4_1'])]
                filename = preferred_files[0] if preferred_files else gguf_files[0]
            
            # Download the file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                force_download=force_download
            )
            
            return local_path
            
        except Exception as e:
            print(f"Error downloading model from HF: {e}")
            return None
    
    def load_gguf_model(self, model_path: str) -> Optional[Tuple[Any, Any]]:
        """Load a GGUF model file"""
        if gguf is None:
            print("GGUF support unavailable (install the `gguf` package).")
            return None
        try:
            # Load GGUF file
            reader = gguf.GGUFReader(model_path)
            
            # Extract metadata
            metadata = {}
            for key, field in reader.fields.items():
                metadata[key] = field.parts[field.data[0]] if field.data else None
                
            # For now, return reader and metadata as a simple interface
            # In a full implementation, you'd load this into a runtime like llama.cpp
            return reader, metadata
            
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            return None
    
    def get_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of a model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB"""
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 ** 3)  # Convert to GB
    
    def create_merkle_tree(self, model_path: str, leaf_size_mb: float = 10.0) -> Tuple[str, List[str]]:
        """Create Merkle tree for model verification"""
        leaf_size_bytes = int(leaf_size_mb * 1024 * 1024)
        leaves = []
        
        # Read model in chunks and create leaves
        with open(model_path, "rb") as f:
            while True:
                chunk = f.read(leaf_size_bytes)
                if not chunk:
                    break
                leaf_hash = hashlib.sha256(chunk).hexdigest()
                leaves.append(leaf_hash)
        
        # Build Merkle tree (simplified implementation)
        if not leaves:
            return "", []
            
        # Calculate tree levels
        current_level = leaves[:]
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)
            current_level = next_level
        
        merkle_root = current_level[0] if current_level else ""
        return merkle_root, leaves
    
    def verify_merkle_proof(
        self, 
        leaf_hash: str, 
        leaf_index: int, 
        proof: List[str], 
        root: str
    ) -> bool:
        """Verify a Merkle proof for a leaf"""
        current_hash = leaf_hash
        
        for proof_hash in proof:
            if leaf_index % 2 == 0:
                # Left leaf
                current_hash = hashlib.sha256((current_hash + proof_hash).encode()).hexdigest()
            else:
                # Right leaf  
                current_hash = hashlib.sha256((proof_hash + current_hash).encode()).hexdigest()
            leaf_index //= 2
        
        return current_hash == root
    
    def query_model(
        self, 
        model_id: str, 
        query: str, 
        max_tokens: int = 100
    ) -> ModelQueryResult:
        """Query a loaded model"""
        try:
            if model_id not in self.models:
                return ModelQueryResult(
                    query=query,
                    response="",
                    tokens_used=0,
                    compute_cost=0.0,
                    success=False,
                    error_message=f"Model {model_id} not found"
                )
            
            # Load model if not already loaded. If loading fails, fall back to a simulated response
            if model_id not in self.loaded_models:
                success = self._load_model(model_id)
                if not success:
                    # Simulate response for registered-but-not-loaded models
                    simulated_response = f"[Simulated Model Response to: {query[:50]}...]"
                    tokens_used = max(1, min(max_tokens, max(1, len(simulated_response.split()))))
                    compute_cost = tokens_used * 0.001
                    return ModelQueryResult(
                        query=query,
                        response=simulated_response,
                        tokens_used=tokens_used,
                        compute_cost=compute_cost,
                        success=True
                    )
            
            model, tokenizer = self.loaded_models[model_id]
            
            # For GGUF models, allow simulated inference even when using a mock loader
            _gguf_reader = gguf is not None and isinstance(model, gguf.GGUFReader)
            if _gguf_reader or model is None:
                # Simulate model inference for testing
                response = f"[GGUF Model Response to: {query[:50]}...]"
                tokens_used = max(1, min(max_tokens, len(response.split())))
            else:
                # Standard transformer model
                inputs = tokenizer.encode(query, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                tokens_used = len(outputs[0]) - len(inputs[0])
            
            # Calculate compute cost
            compute_cost = tokens_used * 0.001  # Simple pricing model
            
            return ModelQueryResult(
                query=query,
                response=response,
                tokens_used=tokens_used,
                compute_cost=compute_cost,
                success=True
            )
            
        except Exception as e:
            return ModelQueryResult(
                query=query,
                response="",
                tokens_used=0,
                compute_cost=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        try:
            model_info = self.models[model_id]
            
            # Find local model file
            model_path = self._get_model_path(model_id)
            if not model_path or not os.path.exists(model_path):
                print(f"Model file not found for {model_id}")
                return False
            
            if model_info.format == "gguf":
                result = self.load_gguf_model(model_path)
                if result is None:
                    return False
                model, metadata = result
                tokenizer = None  # GGUF handles tokenization internally
            else:
                # Load as transformer model
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.loaded_models[model_id] = (model, tokenizer)
            return True
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return False
    
    def _get_model_path(self, model_id: str) -> Optional[str]:
        """Get local path for a model"""
        # This would normally involve downloading from IPFS
        # For now, check cache directory
        cache_path = self.cache_dir / f"{model_id}.gguf"
        if cache_path.exists():
            return str(cache_path)
        return None
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            return True
        return False
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models"""
        return list(self.models.values())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.models.get(model_id)
    
    def is_model_available(self, model_id: str, current_block: int) -> bool:
        """Check if a model is still available (storage not expired)"""
        if model_id not in self.models:
            return False
        
        model_info = self.models[model_id]
        return current_block <= model_info.storage_expires
    
    def cleanup_expired_models(self, current_block: int) -> List[str]:
        """Remove expired models and return list of removed model IDs"""
        expired_models = []
        
        for model_id, model_info in list(self.models.items()):
            if current_block > model_info.storage_expires:
                # Unload from memory if loaded
                self.unload_model(model_id)
                
                # Remove from registry
                del self.models[model_id]
                expired_models.append(model_id)
                
                # Remove local cache files
                if model_info.format == "gguf":
                    cache_path = self.cache_dir / f"{model_id}.gguf"
                    if cache_path.exists():
                        cache_path.unlink()
                elif model_info.format == "nanochat":
                    # Clean up nanochat model files
                    cache_path = self.cache_dir / "nanochat" / model_id
                    if cache_path.exists():
                        import shutil
                        shutil.rmtree(cache_path)
        
        # Also cleanup in nanochat integration
        if self.nanochat_integration:
            self.nanochat_integration.cleanup_expired_models(current_block)
        
        return expired_models
    
    def get_storage_cost(
        self,
        model_size_gb: float,
        storage_blocks: int,
        price_per_gb_per_block: float = 0.001
    ) -> float:
        """
        Calculate storage cost for a model
        
        Args:
            model_size_gb: Model size in GB
            storage_blocks: Number of blocks to store
            price_per_gb_per_block: Base price per GB per block (can be adjusted dynamically)
        
        Returns:
            Total storage cost
        """
        return model_size_gb * storage_blocks * price_per_gb_per_block
    
    def extend_storage(
        self,
        model_id: str,
        additional_blocks: int,
        payment: float
    ) -> bool:
        """
        Extend storage for a model with additional payment
        
        Args:
            model_id: Model ID
            additional_blocks: Number of additional blocks to pay for
            payment: Payment amount
        
        Returns:
            True if successful
        """
        if model_id not in self.models:
            return False
        
        model_info = self.models[model_id]
        
        # Calculate required payment
        required_payment = self.get_storage_cost(
            model_info.size_gb,
            additional_blocks
        )
        
        if payment < required_payment:
            print(f"Insufficient payment for storage extension: {payment} < {required_payment}")
            return False
        
        # Extend storage
        model_info.storage_expires += additional_blocks
        model_info.storage_payment += payment
        
        print(f"✓ Extended storage for model {model_id}")
        print(f"  Additional blocks: {additional_blocks}")
        print(f"  New expiration: {model_info.storage_expires}")
        
        return True
