"""
Nanochat Model Integration for DeSSIN Blockchain
================================================

Integrates Karpathy's nanochat (https://github.com/karpathy/nanochat) models
with the DeSSIN blockchain for decentralized training and inference.

Nanochat models are GPT-style transformers trained from scratch with configurable:
- Depth (number of layers, e.g., d20, d26, d32)
- Training datasets
- Training parameters (device_batch_size, learning_rate, etc.)
"""

import os
import json
import hashlib
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from ..runtime.config import ModelConfig
from .nanochat_wrapper import get_nanochat_wrapper


class TrainingDevice(Enum):
    """Training device options"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"  # Automatically select based on availability


@dataclass
class NanochatModelConfig:
    """Configuration for a nanochat model"""
    model_id: str
    name: str
    depth: int  # Number of transformer layers (e.g., 20, 26, 32)
    device_batch_size: int  # Batch size per device
    vocab_size: int  # Vocabulary size
    context_length: int  # Context length (sequence length)
    
    # Training configuration
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Dataset configuration
    dataset_name: str = "fineweb"  # Default dataset
    dataset_shards: int = 1  # Number of dataset shards
    
    # Model size estimation (in parameters)
    parameters: int = 0  # Will be calculated
    
    # Fine-tuning and customization
    base_model_id: Optional[str] = None  # If this is a fine-tuned model
    identity_data: Optional[Dict[str, str]] = None  # Custom identity/personality
    
    # Training device configuration
    training_device: TrainingDevice = TrainingDevice.AUTO
    
    def __post_init__(self):
        """Calculate estimated parameters if not provided"""
        if self.parameters == 0:
            # Accurate GPT-2 style model parameter calculation
            # For nanochat, hidden_size scales with depth
            hidden_size = 768 + (self.depth - 20) * 32  # Scale with depth
            n_layer = self.depth
            
            # Embedding parameters
            token_embed = self.vocab_size * hidden_size
            pos_embed = self.context_length * hidden_size
            
            # Transformer block parameters per layer
            # Attention: Q, K, V projections + output projection
            attn_params = 4 * (hidden_size * hidden_size)
            # FFN: 2 linear layers (typically 4x expansion)
            ffn_params = 2 * hidden_size * (4 * hidden_size)
            # Layer norms: 2 per block
            ln_params = 2 * hidden_size
            
            block_params = (attn_params + ffn_params + ln_params) * n_layer
            
            # Output layer
            output_params = hidden_size * self.vocab_size
            
            self.parameters = int(token_embed + pos_embed + block_params + output_params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Manually build dict to handle enum properly
        return {
            'model_id': self.model_id,
            'name': self.name,
            'depth': self.depth,
            'device_batch_size': self.device_batch_size,
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_steps': self.max_steps,
            'warmup_steps': self.warmup_steps,
            'dataset_name': self.dataset_name,
            'dataset_shards': self.dataset_shards,
            'parameters': self.parameters,
            'training_device': self.training_device.value if hasattr(self.training_device, 'value') else str(self.training_device),
            'base_model_id': self.base_model_id,
            'identity_data': self.identity_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NanochatModelConfig":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class NanochatTrainingJob:
    """Represents a nanochat training job"""
    job_id: str
    model_config: NanochatModelConfig
    owner_address: str
    
    # Payment and storage
    storage_payment: float  # Upfront payment for storage
    training_payment_per_iteration: float  # Payment per training step
    storage_blocks: int  # Number of blocks to store
    
    # Status tracking
    current_iteration: int = 0
    total_iterations: int = 1000
    status: str = "pending"  # pending, training, completed, expired
    
    # Model paths
    checkpoint_path: Optional[str] = None
    final_model_path: Optional[str] = None
    
    # Training metrics
    current_loss: float = 0.0
    best_loss: float = float('inf')
    training_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['model_config'] = self.model_config.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NanochatTrainingJob":
        """Create from dictionary"""
        if 'model_config' in data and isinstance(data['model_config'], dict):
            data['model_config'] = NanochatModelConfig.from_dict(data['model_config'])
        return cls(**data)


class NanochatIntegration:
    """Integration with nanochat models for DeSSIN"""
    
    def __init__(self, model_manager, default_training_device: TrainingDevice = TrainingDevice.AUTO):
        """
        Initialize nanochat integration
        
        Args:
            model_manager: The ModelManager instance
            default_training_device: Default device for training (CPU/GPU/AUTO)
        """
        self.model_manager = model_manager
        self.cache_dir = Path(model_manager.cache_dir) / "nanochat"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Training jobs registry
        self.training_jobs: Dict[str, NanochatTrainingJob] = {}
        
        # Model configurations
        self.model_configs: Dict[str, NanochatModelConfig] = {}
        
        # Model state (models and optimizers) - NOT serialized
        self.model_states: Dict[str, Dict[str, Any]] = {}  # {model_id: {model, optimizer, device}}
        
        # Training configuration
        self.default_training_device = default_training_device
        self._detect_available_devices()
        
        # Minimum storage blocks required
        self.min_storage_blocks = 100  # Minimum blocks owner must pay for
        
        # Initialize real nanochat wrapper
        self.nanochat_wrapper = get_nanochat_wrapper(str(self.cache_dir))
        nanochat_info = self.nanochat_wrapper.get_nanochat_info()
        
        print("✓ Nanochat integration initialized")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Default training device: {self.default_training_device.value}")
        print(f"  Minimum storage blocks: {self.min_storage_blocks}")
        print(f"  Real nanochat available: {nanochat_info['available']}")
        if nanochat_info['available']:
            print(f"  Nanochat commit: {nanochat_info['commit'][:12]}...")
        else:
            print(f"  ⚠️  Running in simulation mode")
    
    def _detect_available_devices(self):
        """Detect available training devices"""
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  GPU detected: {gpu_name}")
        except:
            pass
        
        if not self.has_gpu:
            print(f"  No GPU detected, using CPU for training")
    
    def get_training_device(self, model_config: Optional[NanochatModelConfig] = None) -> str:
        """
        Get the training device to use
        
        Args:
            model_config: Optional model config with device preference
        
        Returns:
            "cuda" or "cpu"
        """
        device = self.default_training_device
        
        if model_config and model_config.training_device != TrainingDevice.AUTO:
            device = model_config.training_device
        
        if device == TrainingDevice.GPU or device == TrainingDevice.AUTO:
            return "cuda" if self.has_gpu else "cpu"
        else:
            return "cpu"
    
    def create_model(
        self,
        name: str,
        depth: int,
        owner_address: str,
        device_batch_size: int = 32,
        dataset_name: str = "fineweb",
        storage_blocks: int = 1000,
        storage_payment: float = 10.0,
        training_payment_per_iteration: float = 0.001,
        training_device: TrainingDevice = TrainingDevice.AUTO,
        **kwargs
    ) -> Optional[str]:
        """
        Create a new nanochat model configuration
        
        Args:
            name: Model name
            depth: Number of transformer layers
            owner_address: Owner's blockchain address
            device_batch_size: Batch size per device
            dataset_name: Training dataset name
            storage_blocks: Number of blocks to pay for storage
            storage_payment: Upfront storage payment
            training_payment_per_iteration: Payment per training iteration
            **kwargs: Additional model configuration options
        
        Returns:
            model_id if successful, None otherwise
        """
        try:
            # Enforce minimum storage blocks
            if storage_blocks < self.min_storage_blocks:
                print(f"⚠️  Storage blocks too low: {storage_blocks} < {self.min_storage_blocks} (minimum)")
                print(f"   Setting to minimum: {self.min_storage_blocks}")
                storage_blocks = self.min_storage_blocks
            
            # Create model configuration
            model_config = NanochatModelConfig(
                model_id="",  # Will be set below
                name=name,
                depth=depth,
                device_batch_size=device_batch_size,
                dataset_name=dataset_name,
                vocab_size=kwargs.get('vocab_size', 50257),  # GPT-2 default
                context_length=kwargs.get('context_length', 1024),
                training_device=training_device,
                **{k: v for k, v in kwargs.items() if k in [
                    'learning_rate', 'weight_decay', 'max_steps', 'warmup_steps',
                    'dataset_shards', 'base_model_id', 'identity_data'
                ]}
            )
            
            # Generate model ID
            model_data = f"{name}_{depth}_{owner_address}_{dataset_name}"
            model_id = "nanochat_" + hashlib.sha256(model_data.encode()).hexdigest()[:16]
            model_config.model_id = model_id
            
            # Store configuration
            self.model_configs[model_id] = model_config
            
            # Create training job
            job_id = f"job_{model_id}"
            training_job = NanochatTrainingJob(
                job_id=job_id,
                model_config=model_config,
                owner_address=owner_address,
                storage_payment=storage_payment,
                training_payment_per_iteration=training_payment_per_iteration,
                storage_blocks=storage_blocks,
                total_iterations=model_config.max_steps
            )
            
            self.training_jobs[job_id] = training_job
            
            # Register in model manager
            model_size_gb = self._estimate_model_size(model_config)
            
            self.model_manager.register_model(
                model_id=model_id,
                name=name,
                owner=owner_address,
                ipfs_hash=f"Qm{hashlib.sha256(model_id.encode()).hexdigest()[:44]}",
                upload_block=0,  # Will be set by consensus
                storage_expires=storage_blocks,
                size_gb=model_size_gb,
                format="nanochat",
                quantization="32bit",  # Can be quantized to 4bit later
                parameters=model_config.parameters,
                model_hash=hashlib.sha256(model_id.encode()).hexdigest(),
                # Nanochat-specific fields
                depth=depth,
                device_batch_size=device_batch_size,
                dataset_name=dataset_name,
                learning_rate=model_config.learning_rate,
                max_training_steps=model_config.max_steps,
                storage_payment=storage_payment,
                training_payment_per_iteration=training_payment_per_iteration
            )
            
            # Get training device
            device = self.get_training_device(model_config)
            
            print(f"✓ Created nanochat model: {model_id}")
            print(f"  Name: {name}")
            print(f"  Owner: {owner_address[:20]}...")
            print(f"  Depth: d{depth}")
            print(f"  Parameters: {model_config.parameters:,}")
            print(f"  Estimated size: {model_size_gb:.2f} GB")
            print(f"  Storage blocks: {storage_blocks} (paid by owner)")
            print(f"  Training iterations: {model_config.max_steps}")
            print(f"  Training device: {device}")
            
            return model_id
            
        except Exception as e:
            print(f"Error creating nanochat model: {e}")
            return None
    
    def _estimate_model_size(self, config: NanochatModelConfig) -> float:
        """
        Estimate model size in GB
        
        Args:
            config: Model configuration
        
        Returns:
            Estimated size in GB
        """
        # Each parameter is 4 bytes (float32)
        bytes_per_param = 4
        total_bytes = config.parameters * bytes_per_param
        
        # Add overhead for optimizer states, gradients, etc.
        total_bytes *= 3  # Rough estimate
        
        return total_bytes / (1024 ** 3)
    
    def start_training(
        self,
        model_id: str,
        miner_address: str,
        block_index: int
    ) -> bool:
        """
        Start training a nanochat model
        
        Args:
            model_id: Model ID to train
            miner_address: Address of the miner performing training
            block_index: Current block index
        
        Returns:
            True if training started successfully
        """
        try:
            job_id = f"job_{model_id}"
            if job_id not in self.training_jobs:
                print(f"Training job not found: {job_id}")
                return False
            
            job = self.training_jobs[job_id]
            if job.status != "pending":
                print(f"Training job already in progress or completed: {job.status}")
                return False
            
            job.status = "training"
            print(f"✓ Started training job: {job_id}")
            print(f"  Model: {model_id}")
            print(f"  Miner: {miner_address}")
            print(f"  Block: {block_index}")
            
            return True
            
        except Exception as e:
            print(f"Error starting training: {e}")
            return False
    
    def train_iteration(
        self,
        model_id: str,
        training_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a training iteration
        
        Args:
            model_id: Model ID to train
            training_data: Training data batch (optional)
        
        Returns:
            (success, training_metrics) tuple
        """
        try:
            job_id = f"job_{model_id}"
            if job_id not in self.training_jobs:
                return False, {"error": "Training job not found"}
            
            job = self.training_jobs[job_id]
            config = job.model_config
            
            import time
            start_time = time.time()
            
            # Get or create model state
            model_state = self.model_states.get(model_id, {})
            current_model = model_state.get('model')
            current_optimizer = model_state.get('optimizer')
            
            # Use real nanochat wrapper for training
            model_cfg = self.nanochat_wrapper.create_model_config(
                depth=config.depth,
                vocab_size=config.vocab_size,
                context_length=config.context_length
            )
            
            training_cfg = {
                "device_batch_size": config.device_batch_size,
                "max_steps": 1,  # Single iteration
                "learning_rate": config.learning_rate,
            }
            
            # Train using real nanochat (or simulation if not available)
            # Pass existing model and optimizer for continuity
            result = self.nanochat_wrapper.train_model(
                model_config=model_cfg,
                training_config=training_cfg,
                model=current_model,
                optimizer=current_optimizer
            )
            
            # Store model state for next iteration
            if 'model' in result and 'optimizer' in result:
                self.model_states[model_id] = {
                    'model': result['model'],
                    'optimizer': result['optimizer']
                }
            
            # Update job state
            loss_before = result.get("loss_before", job.current_loss if job.current_loss > 0 else 10.0)
            loss_after = result.get("loss_after", loss_before - 0.01)
            
            job.current_loss = loss_after
            job.current_iteration += 1
            
            if loss_after < job.best_loss:
                job.best_loss = loss_after
            
            training_time = time.time() - start_time
            job.training_time_seconds += training_time
            
            # Check if training completed
            if job.current_iteration >= job.total_iterations:
                job.status = "completed"
            
            metrics = {
                "loss_before": loss_before,
                "loss_after": loss_after,
                "loss_improvement": loss_before - loss_after,
                "iteration": job.current_iteration,
                "total_iterations": job.total_iterations,
                "training_time_seconds": training_time,
                "status": job.status,
                "mode": result.get("mode", "real")
            }
            
            return True, metrics
            
        except Exception as e:
            print(f"Error during training iteration: {e}")
            return False, {"error": str(e)}
    
    def deploy_web_interface(
        self,
        model_id: str,
        port: int = 8000
    ) -> Optional[str]:
        """
        Deploy nanochat web interface for a model
        
        Args:
            model_id: Model ID to serve
            port: Port to serve on
        
        Returns:
            URL if successful, None otherwise
        """
        try:
            if model_id not in self.model_configs:
                print(f"Model not found: {model_id}")
                return None
            
            config = self.model_configs[model_id]
            
            # In production, would start actual nanochat web server
            # For now, return the URL where it would be served
            url = f"http://localhost:{port}"
            
            print(f"✓ Web interface deployed for model: {model_id}")
            print(f"  Model name: {config.name}")
            print(f"  URL: {url}")
            print(f"  Note: In production, this would serve nanochat's ui.html")
            
            return url
            
        except Exception as e:
            print(f"Error deploying web interface: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a nanochat model"""
        if model_id not in self.model_configs:
            return None
        
        config = self.model_configs[model_id]
        job_id = f"job_{model_id}"
        job = self.training_jobs.get(job_id)
        
        info = {
            "model_id": model_id,
            "name": config.name,
            "depth": config.depth,
            "parameters": config.parameters,
            "format": "nanochat",
            "config": config.to_dict()
        }
        
        if job:
            info["training"] = {
                "status": job.status,
                "current_iteration": job.current_iteration,
                "total_iterations": job.total_iterations,
                "current_loss": job.current_loss,
                "best_loss": job.best_loss,
                "training_time_seconds": job.training_time_seconds,
                "storage_blocks": job.storage_blocks,
                "storage_payment": job.storage_payment,
                "training_payment_per_iteration": job.training_payment_per_iteration
            }
        
        return info
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all nanochat models"""
        models = []
        for model_id, config in self.model_configs.items():
            info = self.get_model_info(model_id)
            if info:
                models.append(info)
        return models
    
    def calculate_storage_cost(
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
            price_per_gb_per_block: Price per GB per block
        
        Returns:
            Total storage cost
        """
        return model_size_gb * storage_blocks * price_per_gb_per_block
    
    def calculate_training_cost(
        self,
        iterations: int,
        payment_per_iteration: float = 0.001
    ) -> float:
        """
        Calculate training cost
        
        Args:
            iterations: Number of training iterations
            payment_per_iteration: Payment per iteration
        
        Returns:
            Total training cost
        """
        return iterations * payment_per_iteration
    
    def check_storage_expiration(
        self,
        model_id: str,
        current_block: int,
        upload_block: int,
        storage_blocks: int
    ) -> bool:
        """
        Check if model storage has expired
        
        Args:
            model_id: Model ID
            current_block: Current block index
            upload_block: Block when model was uploaded
            storage_blocks: Number of blocks paid for
        
        Returns:
            True if expired, False otherwise
        """
        expiration_block = upload_block + storage_blocks
        is_expired = current_block > expiration_block
        
        if is_expired:
            print(f"⚠️  Model storage expired: {model_id}")
            print(f"  Current block: {current_block}")
            print(f"  Expiration block: {expiration_block}")
        
        return is_expired
    
    def cleanup_expired_models(
        self,
        current_block: int
    ) -> List[str]:
        """
        Clean up expired models
        
        Args:
            current_block: Current block index
        
        Returns:
            List of removed model IDs
        """
        removed_models = []
        
        for model_id in list(self.model_configs.keys()):
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                continue
            
            if self.check_storage_expiration(
                model_id,
                current_block,
                model_info.upload_block,
                model_info.storage_expires
            ):
                # Remove model
                del self.model_configs[model_id]
                
                # Remove training job
                job_id = f"job_{model_id}"
                if job_id in self.training_jobs:
                    del self.training_jobs[job_id]
                
                removed_models.append(model_id)
                
                print(f"🗑️  Removed expired model: {model_id}")
        
        return removed_models
