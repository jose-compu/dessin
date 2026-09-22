"""
Modern Fine-Tuning Techniques for DeSSIN
=========================================

Supports state-of-the-art fine-tuning methods that the community can configure
and experiment with on a per-model basis.

Techniques Supported:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Full Fine-Tuning
- Prompt Tuning
- Prefix Tuning
- Adapter Layers
- P-Tuning v2
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib
import time


class FineTuningMethod(Enum):
    """Available fine-tuning methods"""
    FULL = "full"  # Full parameter fine-tuning
    LORA = "lora"  # Low-Rank Adaptation
    QLORA = "qlora"  # Quantized LoRA
    PROMPT_TUNING = "prompt_tuning"  # Prompt tuning
    PREFIX_TUNING = "prefix_tuning"  # Prefix tuning
    ADAPTER = "adapter"  # Adapter layers
    P_TUNING_V2 = "p_tuning_v2"  # P-Tuning v2
    IA3 = "ia3"  # IA3 (Infused Adapter)


class OptimizerType(Enum):
    """Supported optimizers"""
    ADAMW = "adamw"  # AdamW (default for most cases)
    ADAM = "adam"  # Classic Adam
    SGD = "sgd"  # Stochastic Gradient Descent
    ADAFACTOR = "adafactor"  # Memory-efficient optimizer
    LION = "lion"  # LION optimizer (newer)
    SOPHIA = "sophia"  # Sophia optimizer (second-order)


class LRSchedulerType(Enum):
    """Learning rate schedulers"""
    CONSTANT = "constant"
    LINEAR = "linear"  # Linear decay
    COSINE = "cosine"  # Cosine annealing
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class QuantizationType(Enum):
    """Quantization types for QLoRA"""
    NONE = "none"
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization (NF4)
    FP16 = "fp16"  # Half precision
    BF16 = "bf16"  # Brain float 16


@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA"""
    rank: int = 8  # LoRA rank (r)
    alpha: int = 16  # LoRA alpha (scaling factor)
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", or "lora_only"
    
    # QLoRA specific
    use_quantization: bool = False
    quantization_type: QuantizationType = QuantizationType.INT4
    use_double_quant: bool = True  # Double quantization for QLoRA
    
    # Trainable parameters percentage (for info)
    trainable_params_ratio: float = 0.0


@dataclass
class AdapterConfig:
    """Configuration for Adapter layers"""
    adapter_size: int = 64  # Bottleneck dimension
    adapter_non_linearity: str = "gelu"  # Activation function
    adapter_dropout: float = 0.1
    adapter_residual: bool = True  # Add residual connection
    adapter_layers: List[int] = field(default_factory=list)  # Which layers to add adapters


@dataclass
class PromptTuningConfig:
    """Configuration for Prompt/Prefix Tuning"""
    num_virtual_tokens: int = 20  # Number of virtual tokens
    prompt_tuning_init: str = "random"  # "random" or "text"
    prompt_tuning_init_text: Optional[str] = None
    
    # For prefix tuning
    prefix_projection: bool = False  # Use MLP reparameterization
    encoder_hidden_size: Optional[int] = None


@dataclass
class TrainingConfig:
    """General training configuration"""
    # Optimizer settings
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Learning rate schedule
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE
    warmup_steps: int = 100
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = -1  # -1 means use num_epochs
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False  # Brain float 16 (better for training)
    
    # Gradient checkpointing (saves memory)
    gradient_checkpointing: bool = False
    
    # Data
    max_seq_length: int = 2048
    dataloader_num_workers: int = 0
    
    # Evaluation
    eval_steps: int = 100
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    save_steps: int = 500
    
    # Logging
    logging_steps: int = 10


@dataclass
class FineTuningConfiguration:
    """Complete fine-tuning configuration for a model"""
    config_id: str
    model_id: str
    owner: str
    
    # Fine-tuning method
    method: FineTuningMethod
    
    # Training configuration
    training_config: TrainingConfig
    
    # Method-specific configs (optional)
    lora_config: Optional[LoRAConfig] = None
    adapter_config: Optional[AdapterConfig] = None
    prompt_config: Optional[PromptTuningConfig] = None
    
    # Metadata
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    
    # Experimentation tracking
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration"""
        # Ensure method-specific config is provided
        if self.method in [FineTuningMethod.LORA, FineTuningMethod.QLORA]:
            if self.lora_config is None:
                self.lora_config = LoRAConfig()
                if self.method == FineTuningMethod.QLORA:
                    self.lora_config.use_quantization = True
        
        elif self.method == FineTuningMethod.ADAPTER:
            if self.adapter_config is None:
                self.adapter_config = AdapterConfig()
        
        elif self.method in [FineTuningMethod.PROMPT_TUNING, FineTuningMethod.PREFIX_TUNING]:
            if self.prompt_config is None:
                self.prompt_config = PromptTuningConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {
            'config_id': self.config_id,
            'model_id': self.model_id,
            'owner': self.owner,
            'method': self.method.value,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version,
            'experiment_name': self.experiment_name,
            'tags': self.tags,
            'training_config': {
                'optimizer': self.training_config.optimizer.value,
                'learning_rate': self.training_config.learning_rate,
                'weight_decay': self.training_config.weight_decay,
                'lr_scheduler': self.training_config.lr_scheduler.value,
                'warmup_steps': self.training_config.warmup_steps,
                'num_epochs': self.training_config.num_epochs,
                'batch_size': self.training_config.batch_size,
                'gradient_accumulation_steps': self.training_config.gradient_accumulation_steps,
                'max_steps': self.training_config.max_steps,
                'fp16': self.training_config.fp16,
                'bf16': self.training_config.bf16,
                'gradient_checkpointing': self.training_config.gradient_checkpointing,
            }
        }
        
        if self.lora_config:
            result['lora_config'] = {
                'rank': self.lora_config.rank,
                'alpha': self.lora_config.alpha,
                'dropout': self.lora_config.dropout,
                'target_modules': self.lora_config.target_modules,
                'use_quantization': self.lora_config.use_quantization,
                'quantization_type': self.lora_config.quantization_type.value,
            }
        
        if self.adapter_config:
            result['adapter_config'] = {
                'adapter_size': self.adapter_config.adapter_size,
                'adapter_dropout': self.adapter_config.adapter_dropout,
            }
        
        if self.prompt_config:
            result['prompt_config'] = {
                'num_virtual_tokens': self.prompt_config.num_virtual_tokens,
                'prompt_tuning_init': self.prompt_config.prompt_tuning_init,
            }
        
        return result
    
    def get_trainable_params_estimate(self, total_params: int) -> Dict[str, Any]:
        """Estimate trainable parameters based on method"""
        if self.method == FineTuningMethod.FULL:
            trainable = total_params
            ratio = 1.0
        
        elif self.method in [FineTuningMethod.LORA, FineTuningMethod.QLORA]:
            if self.lora_config:
                # LoRA adds r * (d_in + d_out) parameters per weight matrix
                # Rough estimate: ~0.1-2% of original parameters depending on rank
                ratio = (self.lora_config.rank / 512) * 0.02  # Rough estimate
                trainable = int(total_params * ratio)
            else:
                trainable = int(total_params * 0.01)
                ratio = 0.01
        
        elif self.method == FineTuningMethod.ADAPTER:
            # Adapters typically add 0.5-3% parameters
            trainable = int(total_params * 0.02)
            ratio = 0.02
        
        elif self.method in [FineTuningMethod.PROMPT_TUNING, FineTuningMethod.PREFIX_TUNING]:
            if self.prompt_config:
                # Only virtual tokens are trainable
                trainable = self.prompt_config.num_virtual_tokens * 768  # Assuming 768 hidden size
                ratio = trainable / total_params
            else:
                trainable = 20 * 768
                ratio = trainable / total_params
        
        else:
            trainable = int(total_params * 0.1)
            ratio = 0.1
        
        return {
            'total_params': total_params,
            'trainable_params': trainable,
            'trainable_ratio': ratio,
            'trainable_percentage': ratio * 100,
            'frozen_params': total_params - trainable
        }


class FineTuningConfigurationManager:
    """Manages fine-tuning configurations for models"""
    
    def __init__(self):
        self.configurations: Dict[str, FineTuningConfiguration] = {}
        self.model_configs: Dict[str, str] = {}  # model_id -> config_id
        self.config_history: Dict[str, List[str]] = {}  # model_id -> [config_ids]
    
    def create_configuration(
        self,
        model_id: str,
        owner: str,
        method: FineTuningMethod,
        training_config: Optional[TrainingConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        adapter_config: Optional[AdapterConfig] = None,
        prompt_config: Optional[PromptTuningConfig] = None,
        description: str = "",
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new fine-tuning configuration"""
        # Generate config ID
        config_data = f"{model_id}:{owner}:{method.value}:{time.time()}"
        config_id = f"ftconfig_{hashlib.sha256(config_data.encode()).hexdigest()[:16]}"
        
        # Use default training config if not provided
        if training_config is None:
            training_config = TrainingConfig()
        
        # Create configuration
        config = FineTuningConfiguration(
            config_id=config_id,
            model_id=model_id,
            owner=owner,
            method=method,
            training_config=training_config,
            lora_config=lora_config,
            adapter_config=adapter_config,
            prompt_config=prompt_config,
            description=description,
            experiment_name=experiment_name,
            tags=tags or []
        )
        
        # Store configuration
        self.configurations[config_id] = config
        self.model_configs[model_id] = config_id
        
        # Track history
        if model_id not in self.config_history:
            self.config_history[model_id] = []
        self.config_history[model_id].append(config_id)
        
        print(f"✓ Created fine-tuning configuration: {config_id}")
        print(f"  Model: {model_id}")
        print(f"  Method: {method.value}")
        print(f"  Owner: {owner}")
        
        return config_id
    
    def get_configuration(self, config_id: str) -> Optional[FineTuningConfiguration]:
        """Get a configuration by ID"""
        return self.configurations.get(config_id)
    
    def get_model_configuration(self, model_id: str) -> Optional[FineTuningConfiguration]:
        """Get the current configuration for a model"""
        config_id = self.model_configs.get(model_id)
        if config_id:
            return self.configurations.get(config_id)
        return None
    
    def update_configuration(
        self,
        config_id: str,
        owner: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing configuration (owner only)"""
        config = self.configurations.get(config_id)
        if not config:
            print(f"Configuration not found: {config_id}")
            return False
        
        if config.owner != owner:
            print(f"Unauthorized: {owner} is not the owner of {config_id}")
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = time.time()
        config.version += 1
        
        print(f"✓ Updated configuration: {config_id} (v{config.version})")
        return True
    
    def list_configurations(self, model_id: Optional[str] = None) -> List[FineTuningConfiguration]:
        """List all configurations, optionally filtered by model"""
        if model_id:
            config_ids = self.config_history.get(model_id, [])
            return [self.configurations[cid] for cid in config_ids if cid in self.configurations]
        else:
            return list(self.configurations.values())
    
    def create_preset_configuration(
        self,
        model_id: str,
        owner: str,
        preset: str
    ) -> Optional[str]:
        """Create a configuration from a preset"""
        presets = {
            'lora_efficient': {
                'method': FineTuningMethod.LORA,
                'description': 'Efficient LoRA with rank 8 for fast training',
                'lora_config': LoRAConfig(rank=8, alpha=16, dropout=0.1),
                'training_config': TrainingConfig(
                    learning_rate=3e-4,
                    num_epochs=3,
                    batch_size=4,
                    lr_scheduler=LRSchedulerType.COSINE
                )
            },
            'lora_quality': {
                'method': FineTuningMethod.LORA,
                'description': 'High-quality LoRA with rank 64 for better performance',
                'lora_config': LoRAConfig(rank=64, alpha=128, dropout=0.05),
                'training_config': TrainingConfig(
                    learning_rate=1e-4,
                    num_epochs=5,
                    batch_size=2,
                    gradient_accumulation_steps=2,
                    lr_scheduler=LRSchedulerType.COSINE
                )
            },
            'qlora_4bit': {
                'method': FineTuningMethod.QLORA,
                'description': '4-bit QLoRA for memory-efficient training',
                'lora_config': LoRAConfig(
                    rank=16,
                    alpha=32,
                    use_quantization=True,
                    quantization_type=QuantizationType.INT4
                ),
                'training_config': TrainingConfig(
                    learning_rate=2e-4,
                    num_epochs=3,
                    batch_size=8,
                    bf16=True
                )
            },
            'full_finetuning': {
                'method': FineTuningMethod.FULL,
                'description': 'Full parameter fine-tuning for maximum quality',
                'training_config': TrainingConfig(
                    learning_rate=5e-5,
                    num_epochs=3,
                    batch_size=1,
                    gradient_accumulation_steps=4,
                    lr_scheduler=LRSchedulerType.COSINE,
                    gradient_checkpointing=True
                )
            },
            'adapter_lightweight': {
                'method': FineTuningMethod.ADAPTER,
                'description': 'Lightweight adapter layers',
                'adapter_config': AdapterConfig(adapter_size=64),
                'training_config': TrainingConfig(
                    learning_rate=1e-3,
                    num_epochs=5,
                    batch_size=8
                )
            },
            'prompt_tuning': {
                'method': FineTuningMethod.PROMPT_TUNING,
                'description': 'Prompt tuning with 20 virtual tokens',
                'prompt_config': PromptTuningConfig(num_virtual_tokens=20),
                'training_config': TrainingConfig(
                    learning_rate=3e-2,
                    num_epochs=10,
                    batch_size=16
                )
            }
        }
        
        if preset not in presets:
            print(f"Unknown preset: {preset}")
            print(f"Available presets: {', '.join(presets.keys())}")
            return None
        
        preset_config = presets[preset]
        
        return self.create_configuration(
            model_id=model_id,
            owner=owner,
            method=preset_config['method'],
            training_config=preset_config.get('training_config'),
            lora_config=preset_config.get('lora_config'),
            adapter_config=preset_config.get('adapter_config'),
            prompt_config=preset_config.get('prompt_config'),
            description=preset_config['description'],
            tags=['preset', preset]
        )
