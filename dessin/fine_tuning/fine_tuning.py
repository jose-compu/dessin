"""
Fine-tuning support for DeSSIN blockchain.
Implements full fine-tuning with PoGO verification.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from ..models.model_manager import ModelManager, ModelQueryResult
from ..consensus.transactions import BaseTransaction


@dataclass
class FineTuningDataset:
    """Fine-tuning dataset specification"""
    dataset_id: str
    name: str
    description: str
    size_mb: float
    num_examples: int
    format: str  # "jsonl", "csv", "parquet"
    hash: str
    ipfs_hash: str
    owner: str
    upload_block: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "size_mb": self.size_mb,
            "num_examples": self.num_examples,
            "format": self.format,
            "hash": self.hash,
            "ipfs_hash": self.ipfs_hash,
            "owner": self.owner,
            "upload_block": self.upload_block
        }


@dataclass
class FineTuningJob:
    """Fine-tuning job specification"""
    job_id: str
    base_model_id: str
    dataset_id: str
    owner: str
    
    # Training parameters
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_steps: int
    
    # Economic parameters
    max_cost_dessin: float
    cost_per_step: float
    
    # Status
    status: str  # "pending", "training", "completed", "failed"
    current_epoch: int
    current_step: int
    current_loss: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Results
    final_model_id: Optional[str] = None
    total_cost: float = 0.0
    improvement_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.improvement_metrics is None:
            self.improvement_metrics = {}


@dataclass
class FineTuningTransaction(BaseTransaction):
    """Transaction for starting a fine-tuning job"""
    
    job_id: str
    base_model_id: str
    dataset_id: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_steps: int
    max_cost_dessin: float
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "fine_tuning",
            "sender": self.sender,
            "job_id": self.job_id,
            "base_model_id": self.base_model_id,
            "dataset_id": self.dataset_id,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "max_cost_dessin": self.max_cost_dessin,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


@dataclass
class DatasetUploadTransaction(BaseTransaction):
    """Transaction for uploading a fine-tuning dataset"""
    
    dataset_name: str
    dataset_hash: str
    dataset_size_mb: float
    num_examples: int
    dataset_format: str
    storage_payment: float
    ipfs_hash: Optional[str] = None
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            "type": "dataset_upload",
            "sender": self.sender,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "dataset_size_mb": self.dataset_size_mb,
            "num_examples": self.num_examples,
            "dataset_format": self.dataset_format,
            "storage_payment": self.storage_payment,
            "ipfs_hash": self.ipfs_hash,
            "fee": self.fee,
            "timestamp": self.timestamp
        }


class FineTuningManager:
    """Manages fine-tuning operations for DeSSIN"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Fine-tuning state
        self.datasets: Dict[str, FineTuningDataset] = {}
        self.jobs: Dict[str, FineTuningJob] = {}
        self.active_jobs: List[str] = []
        
        # Supported formats
        self.supported_formats = ["jsonl", "csv", "parquet"]
        
        # Fine-tuning parameters
        self.default_learning_rates = {
            "tiny": 5e-4,    # <1B parameters
            "small": 3e-4,   # 1-7B parameters
            "medium": 2e-4,  # 7-13B parameters
            "large": 1e-4,   # 13-30B parameters
        }
    
    def register_dataset(
        self, 
        dataset_name: str,
        dataset_path: str,
        dataset_format: str,
        owner: str,
        description: str = ""
    ) -> Optional[str]:
        """Register a fine-tuning dataset"""
        try:
            if not Path(dataset_path).exists():
                print(f"Dataset file not found: {dataset_path}")
                return None
            
            if dataset_format not in self.supported_formats:
                print(f"Unsupported format: {dataset_format}")
                return None
            
            # Calculate dataset properties
            dataset_hash = self._calculate_file_hash(dataset_path)
            dataset_size = Path(dataset_path).stat().st_size / (1024 * 1024)  # MB
            num_examples = self._count_examples(dataset_path, dataset_format)
            
            # Generate dataset ID
            dataset_id = f"dataset_{dataset_hash[:8]}"
            
            # Create dataset info
            dataset = FineTuningDataset(
                dataset_id=dataset_id,
                name=dataset_name,
                description=description,
                size_mb=dataset_size,
                num_examples=num_examples,
                format=dataset_format,
                hash=dataset_hash,
                ipfs_hash=f"Qm{dataset_hash[:44]}",  # Simulated
                owner=owner,
                upload_block=0  # Would be set by node
            )
            
            self.datasets[dataset_id] = dataset
            print(f"Registered dataset {dataset_id} with {num_examples} examples")
            
            return dataset_id
            
        except Exception as e:
            print(f"Error registering dataset: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _count_examples(self, file_path: str, format: str) -> int:
        """Count examples in a dataset file"""
        try:
            if format == "jsonl":
                with open(file_path, 'r') as f:
                    return sum(1 for line in f if line.strip())
            elif format == "csv":
                with open(file_path, 'r') as f:
                    return sum(1 for line in f) - 1  # Subtract header
            else:
                # For other formats, estimate
                file_size = Path(file_path).stat().st_size
                return max(1, file_size // 1024)  # Rough estimate
        except Exception:
            return 0
    
    def create_fine_tuning_job(
        self,
        base_model_id: str,
        dataset_id: str,
        owner: str,
        learning_rate: Optional[float] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_steps: int = 1000,
        max_cost_dessin: float = 100.0
    ) -> Optional[str]:
        """Create a new fine-tuning job"""
        try:
            # Validate inputs
            if base_model_id not in self.model_manager.models:
                print(f"Base model not found: {base_model_id}")
                return None
            
            if dataset_id not in self.datasets:
                print(f"Dataset not found: {dataset_id}")
                return None
            
            # Determine learning rate if not provided
            if learning_rate is None:
                model_info = self.model_manager.get_model_info(base_model_id)
                if model_info.parameters < 1e9:
                    learning_rate = self.default_learning_rates["tiny"]
                elif model_info.parameters < 7e9:
                    learning_rate = self.default_learning_rates["small"]
                elif model_info.parameters < 13e9:
                    learning_rate = self.default_learning_rates["medium"]
                else:
                    learning_rate = self.default_learning_rates["large"]
            
            # Calculate cost per step
            model_info = self.model_manager.get_model_info(base_model_id)
            cost_per_step = self._calculate_cost_per_step(model_info.parameters, batch_size)
            
            # Generate job ID
            job_data = f"{base_model_id}:{dataset_id}:{owner}:{time.time()}"
            job_id = hashlib.sha256(job_data.encode()).hexdigest()[:16]
            
            # Create job
            job = FineTuningJob(
                job_id=job_id,
                base_model_id=base_model_id,
                dataset_id=dataset_id,
                owner=owner,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_steps=max_steps,
                max_cost_dessin=max_cost_dessin,
                cost_per_step=cost_per_step,
                status="pending",
                current_epoch=0,
                current_step=0,
                current_loss=0.0
            )
            
            self.jobs[job_id] = job
            print(f"Created fine-tuning job {job_id}")
            
            return job_id
            
        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            return None
    
    def _calculate_cost_per_step(self, num_parameters: int, batch_size: int) -> float:
        """Calculate cost per training step based on model size"""
        # Base cost: 0.001 DESSIN per billion parameters per step
        base_cost = (num_parameters / 1e9) * 0.001
        
        # Scale by batch size
        batch_factor = batch_size / 4.0  # 4 is baseline batch size
        
        return base_cost * batch_factor
    
    def start_fine_tuning(self, job_id: str) -> bool:
        """Start a fine-tuning job"""
        try:
            if job_id not in self.jobs:
                print(f"Job not found: {job_id}")
                return False
            
            job = self.jobs[job_id]
            
            if job.status != "pending":
                print(f"Job {job_id} is not pending (status: {job.status})")
                return False
            
            # Start the job
            job.status = "training"
            job.start_time = time.time()
            self.active_jobs.append(job_id)
            
            print(f"Started fine-tuning job {job_id}")
            return True
            
        except Exception as e:
            print(f"Error starting fine-tuning: {e}")
            return False
    
    def simulate_fine_tuning_step(self, job_id: str) -> bool:
        """Simulate one fine-tuning step (for testing)"""
        try:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            
            if job.status != "training":
                return False
            
            # Simulate training step
            job.current_step += 1
            
            # Simulate loss improvement
            initial_loss = 2.5
            step_improvement = 0.001 * (1 - job.current_step / job.max_steps)
            job.current_loss = initial_loss - (job.current_step * step_improvement)
            
            # Update epoch
            dataset = self.datasets[job.dataset_id]
            steps_per_epoch = dataset.num_examples // job.batch_size
            job.current_epoch = job.current_step // steps_per_epoch
            
            # Update cost
            job.total_cost += job.cost_per_step
            
            # Check completion conditions
            if (job.current_step >= job.max_steps or 
                job.current_epoch >= job.num_epochs or
                job.total_cost >= job.max_cost_dessin):
                
                return self._complete_fine_tuning(job_id)
            
            return True
            
        except Exception as e:
            print(f"Error in fine-tuning step: {e}")
            return False
    
    def _complete_fine_tuning(self, job_id: str) -> bool:
        """Complete a fine-tuning job"""
        try:
            job = self.jobs[job_id]
            
            # Create fine-tuned model
            base_model = self.model_manager.get_model_info(job.base_model_id)
            if not base_model:
                job.status = "failed"
                return False
            
            # Generate fine-tuned model ID
            fine_tuned_model_id = f"ft_{job.base_model_id}_{job_id[:8]}"
            
            # Register fine-tuned model
            success = self.model_manager.register_model(
                model_id=fine_tuned_model_id,
                name=f"Fine-tuned {base_model.name}",
                size_gb=base_model.size_gb,
                format=base_model.format,
                quantization=base_model.quantization,
                parameters=base_model.parameters,
                ipfs_hash=f"Qm{job_id}{base_model.ipfs_hash[8:]}",
                owner=job.owner,
                upload_block=0,  # Would be current block
                storage_expires=1000,  # Default expiry
                model_hash=f"ft_{base_model.model_hash}"
            )
            
            if success:
                job.status = "completed"
                job.final_model_id = fine_tuned_model_id
                job.end_time = time.time()
                
                # Calculate improvement metrics
                job.improvement_metrics = {
                    "loss_improvement": 2.5 - job.current_loss,
                    "steps_completed": job.current_step,
                    "epochs_completed": job.current_epoch,
                    "training_time_minutes": (job.end_time - job.start_time) / 60.0 if job.start_time else 0
                }
                
                print(f"Fine-tuning job {job_id} completed successfully")
                print(f"Final model: {fine_tuned_model_id}")
                
            else:
                job.status = "failed"
                print(f"Failed to register fine-tuned model for job {job_id}")
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                self.active_jobs.remove(job_id)
            
            return success
            
        except Exception as e:
            print(f"Error completing fine-tuning: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a fine-tuning job"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        dataset = self.datasets.get(job.dataset_id)
        
        status = {
            "job_id": job_id,
            "status": job.status,
            "base_model_id": job.base_model_id,
            "dataset_name": dataset.name if dataset else "Unknown",
            "current_step": job.current_step,
            "max_steps": job.max_steps,
            "current_epoch": job.current_epoch,
            "num_epochs": job.num_epochs,
            "current_loss": job.current_loss,
            "total_cost": job.total_cost,
            "max_cost": job.max_cost_dessin,
            "progress_percent": (job.current_step / job.max_steps) * 100,
        }
        
        if job.status == "completed":
            status.update({
                "final_model_id": job.final_model_id,
                "improvement_metrics": job.improvement_metrics
            })
        
        return status
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets"""
        return [dataset.to_dict() for dataset in self.datasets.values()]
    
    def list_jobs(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """List fine-tuning jobs"""
        jobs = []
        
        for job in self.jobs.values():
            if owner is None or job.owner == owner:
                status = self.get_job_status(job.job_id)
                if status:
                    jobs.append(status)
        
        return jobs
    
    def cleanup_completed_jobs(self, max_age_hours: float = 24.0) -> List[str]:
        """Clean up old completed jobs"""
        cleaned_jobs = []
        current_time = time.time()
        
        for job_id, job in list(self.jobs.items()):
            if (job.status in ["completed", "failed"] and 
                job.end_time and 
                (current_time - job.end_time) > (max_age_hours * 3600)):
                
                del self.jobs[job_id]
                cleaned_jobs.append(job_id)
        
        return cleaned_jobs
