"""
Blockchain Transactions for Fine-Tuning Configuration
======================================================

Allows model owners to configure and update fine-tuning settings via blockchain transactions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..consensus.transactions import BaseTransaction


@dataclass
class CreateFineTuningConfigTransaction(BaseTransaction):
    """Transaction to create a fine-tuning configuration for a model"""
    model_id: str
    method: str  # "lora", "qlora", "full", "adapter", etc.
    config_data: Dict[str, Any]  # Configuration parameters
    description: str = ""
    experiment_name: Optional[str] = None
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'create_finetuning_config',
            'model_id': self.model_id,
            'method': self.method,
            'config_data': self.config_data,
            'description': self.description,
            'experiment_name': self.experiment_name,
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


@dataclass
class UpdateFineTuningConfigTransaction(BaseTransaction):
    """Transaction to update an existing fine-tuning configuration"""
    config_id: str
    updates: Dict[str, Any]
    reason: str = ""
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'update_finetuning_config',
            'config_id': self.config_id,
            'updates': self.updates,
            'reason': self.reason,
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


@dataclass
class SetModelFineTuningPresetTransaction(BaseTransaction):
    """Transaction to set a preset configuration for a model"""
    model_id: str
    preset_name: str  # "lora_efficient", "qlora_4bit", "full_finetuning", etc.
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'set_finetuning_preset',
            'model_id': self.model_id,
            'preset_name': self.preset_name,
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


@dataclass
class StartExperimentTransaction(BaseTransaction):
    """Transaction to start a fine-tuning experiment with specific config"""
    model_id: str
    config_id: str
    experiment_name: str
    dataset_id: str
    max_cost_dessin: float
    tags: Dict[str, str] = None  # Additional experiment tags
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'start_experiment',
            'model_id': self.model_id,
            'config_id': self.config_id,
            'experiment_name': self.experiment_name,
            'dataset_id': self.dataset_id,
            'max_cost_dessin': self.max_cost_dessin,
            'tags': self.tags or {},
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


class FineTuningConfigTransactionProcessor:
    """Processes fine-tuning configuration transactions"""
    
    def __init__(self, config_manager, model_manager):
        self.config_manager = config_manager
        self.model_manager = model_manager
    
    def process_transaction(self, transaction: BaseTransaction) -> bool:
        """Process a fine-tuning configuration transaction"""
        if isinstance(transaction, CreateFineTuningConfigTransaction):
            return self._process_create_config(transaction)
        
        elif isinstance(transaction, UpdateFineTuningConfigTransaction):
            return self._process_update_config(transaction)
        
        elif isinstance(transaction, SetModelFineTuningPresetTransaction):
            return self._process_set_preset(transaction)
        
        elif isinstance(transaction, StartExperimentTransaction):
            return self._process_start_experiment(transaction)
        
        return False
    
    def _process_create_config(self, tx: CreateFineTuningConfigTransaction) -> bool:
        """Process configuration creation"""
        # Verify model exists and sender is owner
        model = self.model_manager.models.get(tx.model_id)
        if not model:
            print(f"Model not found: {tx.model_id}")
            return False
        
        if model.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of {tx.model_id}")
            return False
        
        # Create configuration from transaction data
        from .fine_tuning_techniques import FineTuningMethod
        
        try:
            method = FineTuningMethod(tx.method)
        except ValueError:
            print(f"Invalid method: {tx.method}")
            return False
        
        # Parse config data and create configuration
        # This would need to parse the config_data dict and create appropriate config objects
        config_id = self.config_manager.create_configuration(
            model_id=tx.model_id,
            owner=tx.sender,
            method=method,
            description=tx.description,
            experiment_name=tx.experiment_name
        )
        
        print(f"✓ Processed CreateFineTuningConfigTransaction")
        print(f"  Config ID: {config_id}")
        print(f"  Model: {tx.model_id}")
        print(f"  Method: {tx.method}")
        
        return True
    
    def _process_update_config(self, tx: UpdateFineTuningConfigTransaction) -> bool:
        """Process configuration update"""
        success = self.config_manager.update_configuration(
            config_id=tx.config_id,
            owner=tx.sender,
            updates=tx.updates
        )
        
        if success:
            print(f"✓ Processed UpdateFineTuningConfigTransaction")
            print(f"  Config ID: {tx.config_id}")
            print(f"  Reason: {tx.reason}")
        
        return success
    
    def _process_set_preset(self, tx: SetModelFineTuningPresetTransaction) -> bool:
        """Process preset configuration"""
        # Verify model exists and sender is owner
        model = self.model_manager.models.get(tx.model_id)
        if not model:
            print(f"Model not found: {tx.model_id}")
            return False
        
        if model.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of {tx.model_id}")
            return False
        
        # Create preset configuration
        config_id = self.config_manager.create_preset_configuration(
            model_id=tx.model_id,
            owner=tx.sender,
            preset=tx.preset_name
        )
        
        if config_id:
            print(f"✓ Processed SetModelFineTuningPresetTransaction")
            print(f"  Model: {tx.model_id}")
            print(f"  Preset: {tx.preset_name}")
            print(f"  Config ID: {config_id}")
            return True
        
        return False
    
    def _process_start_experiment(self, tx: StartExperimentTransaction) -> bool:
        """Process experiment start"""
        # Verify model, config, and dataset exist
        model = self.model_manager.models.get(tx.model_id)
        if not model:
            print(f"Model not found: {tx.model_id}")
            return False
        
        if model.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of {tx.model_id}")
            return False
        
        config = self.config_manager.get_configuration(tx.config_id)
        if not config:
            print(f"Configuration not found: {tx.config_id}")
            return False
        
        print(f"✓ Processed StartExperimentTransaction")
        print(f"  Experiment: {tx.experiment_name}")
        print(f"  Model: {tx.model_id}")
        print(f"  Config: {tx.config_id}")
        print(f"  Dataset: {tx.dataset_id}")
        print(f"  Max cost: {tx.max_cost_dessin} DESSIN")
        
        return True
