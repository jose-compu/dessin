"""
Multi-Layer Fine-Tuning Pipeline for DeSSIN
============================================

Allows model owners to define multiple fine-tuning layers with:
- Automatic progression based on conditions
- Manual progression via blockchain transactions
- Custom datasets per layer
- Configurable training parameters per layer
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..consensus.transactions import BaseTransaction


class ProgressionCondition(Enum):
    """Conditions for automatic layer progression"""
    LOSS_THRESHOLD = "loss_threshold"
    ITERATION_COUNT = "iteration_count"
    ACCURACY_THRESHOLD = "accuracy_threshold"
    TIME_ELAPSED = "time_elapsed"
    MANUAL_ONLY = "manual_only"


@dataclass
class FineTuningLayer:
    """Single layer in a fine-tuning pipeline"""
    layer_id: str
    layer_index: int  # 0, 1, 2, ...
    name: str
    description: str
    
    # Training configuration
    dataset_id: str
    learning_rate: float
    num_iterations: int
    batch_size: int
    
    # Progression conditions
    progression_condition: ProgressionCondition
    condition_value: float  # Threshold for automatic progression
    
    # Status
    status: str = "pending"  # pending, active, completed, skipped
    iterations_completed: int = 0
    current_loss: float = 0.0
    best_loss: float = float('inf')
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Results
    model_checkpoint_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_progression_condition_met(self) -> bool:
        """Check if automatic progression condition is met"""
        if self.progression_condition == ProgressionCondition.MANUAL_ONLY:
            return False
        
        elif self.progression_condition == ProgressionCondition.LOSS_THRESHOLD:
            return self.best_loss <= self.condition_value
        
        elif self.progression_condition == ProgressionCondition.ITERATION_COUNT:
            return self.iterations_completed >= int(self.condition_value)
        
        elif self.progression_condition == ProgressionCondition.ACCURACY_THRESHOLD:
            accuracy = self.metrics.get('accuracy', 0.0)
            return accuracy >= self.condition_value
        
        elif self.progression_condition == ProgressionCondition.TIME_ELAPSED:
            if self.start_time is None:
                return False
            elapsed_minutes = (time.time() - self.start_time) / 60.0
            return elapsed_minutes >= self.condition_value
        
        return False


@dataclass
class FineTuningPipeline:
    """Multi-layer fine-tuning pipeline"""
    pipeline_id: str
    name: str
    description: str
    base_model_id: str
    owner: str
    
    # Layers
    layers: List[FineTuningLayer] = field(default_factory=list)
    current_layer_index: int = 0
    
    # Status
    status: str = "pending"  # pending, training, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Economics
    max_cost_dessin: float = 100.0
    total_cost: float = 0.0
    
    # Final results
    final_model_id: Optional[str] = None
    total_iterations: int = 0
    
    def get_current_layer(self) -> Optional[FineTuningLayer]:
        """Get the currently active layer"""
        if 0 <= self.current_layer_index < len(self.layers):
            return self.layers[self.current_layer_index]
        return None
    
    def progress_to_next_layer(self) -> bool:
        """Move to the next layer in the pipeline"""
        current = self.get_current_layer()
        if current and current.status == "active":
            current.status = "completed"
            current.end_time = time.time()
        
        self.current_layer_index += 1
        
        next_layer = self.get_current_layer()
        if next_layer:
            next_layer.status = "active"
            next_layer.start_time = time.time()
            return True
        else:
            # All layers completed
            self.status = "completed"
            self.end_time = time.time()
            return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get pipeline progress summary"""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "status": self.status,
            "current_layer": self.current_layer_index,
            "total_layers": len(self.layers),
            "progress_percent": (self.current_layer_index / len(self.layers)) * 100 if self.layers else 0,
            "total_iterations": self.total_iterations,
            "total_cost": self.total_cost,
            "max_cost": self.max_cost_dessin
        }


@dataclass
class LayerProgressionTransaction(BaseTransaction):
    """Transaction to manually progress to next fine-tuning layer"""
    
    pipeline_id: str
    current_layer_index: int
    force_progression: bool = False  # Force even if conditions not met
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "layer_progression",
            "sender": self.sender,
            "pipeline_id": self.pipeline_id,
            "current_layer_index": self.current_layer_index,
            "force_progression": self.force_progression,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class CreatePipelineTransaction(BaseTransaction):
    """Transaction to create a new fine-tuning pipeline"""
    
    pipeline_name: str
    base_model_id: str
    layer_configs: List[Dict[str, Any]]
    max_cost_dessin: float
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "create_pipeline",
            "sender": self.sender,
            "pipeline_name": self.pipeline_name,
            "base_model_id": self.base_model_id,
            "layer_configs": self.layer_configs,
            "max_cost_dessin": self.max_cost_dessin,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


class FineTuningPipelineManager:
    """Manages multi-layer fine-tuning pipelines"""
    
    def __init__(self, model_manager, fine_tuning_manager):
        self.model_manager = model_manager
        self.fine_tuning_manager = fine_tuning_manager
        
        self.pipelines: Dict[str, FineTuningPipeline] = {}
        self.active_pipelines: List[str] = []
        
        print("✓ Fine-tuning pipeline manager initialized")
    
    def create_pipeline(
        self,
        name: str,
        base_model_id: str,
        owner: str,
        layer_configs: List[Dict[str, Any]],
        max_cost_dessin: float = 100.0,
        description: str = ""
    ) -> Optional[str]:
        """Create a new multi-layer fine-tuning pipeline"""
        
        # Validate base model
        if base_model_id not in self.model_manager.models:
            print(f"Base model not found: {base_model_id}")
            return None
        
        # Validate layers
        if not layer_configs:
            print("At least one layer required")
            return None
        
        # Generate pipeline ID
        pipeline_data = f"{name}:{base_model_id}:{owner}:{time.time()}"
        pipeline_id = "pipeline_" + hashlib.sha256(pipeline_data.encode()).hexdigest()[:16]
        
        # Create layers
        layers = []
        for i, config in enumerate(layer_configs):
            layer_id = f"{pipeline_id}_layer_{i}"
            
            # Parse condition
            condition_type = ProgressionCondition(
                config.get('progression_condition', 'iteration_count')
            )
            condition_value = config.get('condition_value', 100)
            
            layer = FineTuningLayer(
                layer_id=layer_id,
                layer_index=i,
                name=config.get('name', f'Layer {i+1}'),
                description=config.get('description', ''),
                dataset_id=config['dataset_id'],
                learning_rate=config.get('learning_rate', 3e-4),
                num_iterations=config.get('num_iterations', 100),
                batch_size=config.get('batch_size', 4),
                progression_condition=condition_type,
                condition_value=condition_value
            )
            
            layers.append(layer)
        
        # Create pipeline
        pipeline = FineTuningPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            base_model_id=base_model_id,
            owner=owner,
            layers=layers,
            max_cost_dessin=max_cost_dessin
        )
        
        self.pipelines[pipeline_id] = pipeline
        
        print(f"✓ Created pipeline {pipeline_id}")
        print(f"  Name: {name}")
        print(f"  Layers: {len(layers)}")
        print(f"  Owner: {owner[:20]}...")
        
        for layer in layers:
            print(f"    Layer {layer.layer_index}: {layer.name}")
            print(f"      Condition: {layer.progression_condition.value}")
            print(f"      Value: {layer.condition_value}")
        
        return pipeline_id
    
    def start_pipeline(self, pipeline_id: str) -> bool:
        """Start a fine-tuning pipeline"""
        if pipeline_id not in self.pipelines:
            print(f"Pipeline not found: {pipeline_id}")
            return False
        
        pipeline = self.pipelines[pipeline_id]
        
        if pipeline.status != "pending":
            print(f"Pipeline not pending: {pipeline.status}")
            return False
        
        # Start first layer
        pipeline.status = "training"
        pipeline.start_time = time.time()
        
        first_layer = pipeline.get_current_layer()
        if first_layer:
            first_layer.status = "active"
            first_layer.start_time = time.time()
            self.active_pipelines.append(pipeline_id)
            
            print(f"✓ Started pipeline {pipeline_id}")
            print(f"  Current layer: {first_layer.name}")
            return True
        
        return False
    
    def train_iteration(self, pipeline_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute one training iteration on current layer"""
        if pipeline_id not in self.pipelines:
            return False, {"error": "Pipeline not found"}
        
        pipeline = self.pipelines[pipeline_id]
        
        if pipeline.status != "training":
            return False, {"error": f"Pipeline not training: {pipeline.status}"}
        
        current_layer = pipeline.get_current_layer()
        if not current_layer:
            return False, {"error": "No active layer"}
        
        # Simulate training iteration
        import random
        loss_before = current_layer.current_loss if current_layer.current_loss > 0 else random.uniform(8.0, 12.0)
        loss_after = loss_before - random.uniform(0.01, 0.1)
        
        current_layer.current_loss = loss_after
        current_layer.iterations_completed += 1
        pipeline.total_iterations += 1
        
        if loss_after < current_layer.best_loss:
            current_layer.best_loss = loss_after
        
        # Update metrics
        current_layer.metrics['loss'] = loss_after
        current_layer.metrics['loss_improvement'] = loss_before - loss_after
        
        # Calculate cost (simplified)
        iteration_cost = 0.001
        pipeline.total_cost += iteration_cost
        
        metrics = {
            "pipeline_id": pipeline_id,
            "layer_index": current_layer.layer_index,
            "layer_name": current_layer.name,
            "iteration": current_layer.iterations_completed,
            "max_iterations": current_layer.num_iterations,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_improvement": loss_before - loss_after,
            "best_loss": current_layer.best_loss,
            "total_cost": pipeline.total_cost,
            "condition_type": current_layer.progression_condition.value,
            "condition_met": current_layer.is_progression_condition_met()
        }
        
        # Check automatic progression
        if current_layer.is_progression_condition_met():
            print(f"  ⚡ Condition met! Auto-progressing to next layer...")
            if pipeline.progress_to_next_layer():
                next_layer = pipeline.get_current_layer()
                metrics["progressed_to_layer"] = next_layer.layer_index
                metrics["next_layer_name"] = next_layer.name
            else:
                metrics["pipeline_completed"] = True
        
        return True, metrics
    
    def manual_progress_layer(
        self, 
        pipeline_id: str, 
        force: bool = False
    ) -> bool:
        """Manually progress to next layer (via transaction)"""
        if pipeline_id not in self.pipelines:
            print(f"Pipeline not found: {pipeline_id}")
            return False
        
        pipeline = self.pipelines[pipeline_id]
        current_layer = pipeline.get_current_layer()
        
        if not current_layer:
            print("No active layer")
            return False
        
        # Check conditions unless forced
        if not force and not current_layer.is_progression_condition_met():
            print(f"Progression condition not met")
            print(f"  Condition: {current_layer.progression_condition.value}")
            print(f"  Required: {current_layer.condition_value}")
            print(f"  Current: {current_layer.iterations_completed}")
            return False
        
        print(f"Manual progression: {current_layer.name} → ", end="")
        
        if pipeline.progress_to_next_layer():
            next_layer = pipeline.get_current_layer()
            print(f"{next_layer.name}")
            return True
        else:
            print("Pipeline completed")
            return True
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed pipeline status"""
        if pipeline_id not in self.pipelines:
            return None
        
        pipeline = self.pipelines[pipeline_id]
        current_layer = pipeline.get_current_layer()
        
        status = {
            "pipeline_id": pipeline_id,
            "name": pipeline.name,
            "status": pipeline.status,
            "base_model_id": pipeline.base_model_id,
            "owner": pipeline.owner,
            "current_layer_index": pipeline.current_layer_index,
            "total_layers": len(pipeline.layers),
            "total_iterations": pipeline.total_iterations,
            "total_cost": pipeline.total_cost,
            "max_cost": pipeline.max_cost_dessin,
            "layers": []
        }
        
        for layer in pipeline.layers:
            layer_info = {
                "index": layer.layer_index,
                "name": layer.name,
                "status": layer.status,
                "iterations_completed": layer.iterations_completed,
                "max_iterations": layer.num_iterations,
                "current_loss": layer.current_loss,
                "best_loss": layer.best_loss,
                "progression_condition": layer.progression_condition.value,
                "condition_value": layer.condition_value,
                "condition_met": layer.is_progression_condition_met()
            }
            status["layers"].append(layer_info)
        
        if current_layer:
            status["current_layer"] = {
                "index": current_layer.layer_index,
                "name": current_layer.name,
                "iterations_completed": current_layer.iterations_completed,
                "max_iterations": current_layer.num_iterations,
                "best_loss": current_layer.best_loss
            }
        
        return status
    
    def list_pipelines(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all pipelines"""
        pipelines = []
        for pipeline in self.pipelines.values():
            if owner is None or pipeline.owner == owner:
                status = self.get_pipeline_status(pipeline.pipeline_id)
                if status:
                    pipelines.append(status)
        return pipelines
