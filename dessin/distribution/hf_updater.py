"""
HuggingFace model updater for DeSSIN blockchain.
Handles updating models on HF after additional training iterations.
"""

import os
import json
import time
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

from huggingface_hub import HfApi, create_repo, upload_file, model_info
from huggingface_hub.utils import HfHubHTTPError

from ..models.model_manager import ModelManager, ModelInfo


@dataclass
class ModelVersion:
    """Represents a version of a model"""
    version_id: str
    block_height: int
    loss_value: float
    training_iterations: int
    timestamp: float
    model_hash: str
    commit_sha: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelUpdateRecord:
    """Records updates made to a model"""
    model_id: str
    base_model_repo: str
    versions: List[ModelVersion]
    latest_version: str
    total_iterations: int
    best_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model_repo": self.base_model_repo,
            "versions": [v.to_dict() for v in self.versions],
            "latest_version": self.latest_version,
            "total_iterations": self.total_iterations,
            "best_loss": self.best_loss
        }


class HuggingFaceUpdater:
    """Manages updating models on HuggingFace after training iterations"""
    
    def __init__(
        self, 
        model_manager: ModelManager,
        hf_token: Optional[str] = None,
        organization: Optional[str] = None
    ):
        self.model_manager = model_manager
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.organization = organization or os.getenv("HF_ORGANIZATION")
        
        # Initialize HF API if token is available
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        
        # Model update tracking
        self.update_records: Dict[str, ModelUpdateRecord] = {}
        
        # Configuration
        self.min_iterations_for_update = 10  # Minimum iterations before updating HF
        self.min_loss_improvement = 0.01     # Minimum loss improvement to warrant update
        self.max_versions_per_model = 50     # Maximum versions to keep
        
        # Repository naming
        self.repo_prefix = "dessin-trained"
        
        print(f"HuggingFace Updater initialized")
        if self.api:
            print(f"✓ HF API authenticated")
        else:
            print("⚠ No HF token provided - updates will be simulated")
        if self.organization:
            print(f"✓ Organization: {self.organization}")
    
    def create_model_repository(
        self, 
        model_id: str, 
        base_model_info: ModelInfo,
        private: bool = False
    ) -> Optional[str]:
        """Create a new repository on HuggingFace for a trained model"""
        try:
            # Generate repository name
            repo_name = f"{self.repo_prefix}-{model_id.replace('_', '-')}"
            
            if self.organization:
                repo_id = f"{self.organization}/{repo_name}"
            else:
                repo_id = repo_name
            
            if not self.api:
                print(f"Simulated: Would create repository {repo_id}")
                return repo_id
            
            # Create repository
            repo_url = create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            
            # Create initial README
            readme_content = self._generate_model_readme(model_id, base_model_info)
            
            # Upload README
            upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=self.hf_token,
                commit_message="Initial model documentation"
            )
            
            print(f"✓ Created HuggingFace repository: {repo_id}")
            return repo_id
            
        except HfHubHTTPError as e:
            print(f"HF HTTP error creating repository: {e}")
            return None
        except Exception as e:
            print(f"Error creating repository: {e}")
            return None
    
    def _generate_model_readme(self, model_id: str, base_model_info: ModelInfo) -> str:
        """Generate README content for the model repository"""
        readme = f"""---
license: apache-2.0
base_model: {base_model_info.name}
tags:
- dessin
- pogo-consensus
- blockchain-trained
- {base_model_info.format}
pipeline_tag: text-generation
---

# {model_id}

This model has been trained using the DeSSIN blockchain with PoGO (Proof of Gradient Optimization) consensus.

## Model Details

- **Base Model**: {base_model_info.name}
- **Parameters**: {base_model_info.parameters:,}
- **Format**: {base_model_info.format.upper()}
- **Quantization**: {base_model_info.quantization}
- **Blockchain Trained**: Yes
- **Consensus**: PoGO (Proof of Gradient Optimization)

## Training Process

This model was trained using decentralized training on the DeSSIN blockchain network. Each training iteration was verified by multiple validators using cryptographic proofs.

## Usage

```python
# Load the model (example for GGUF format)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Note: Adjust loading method based on model format
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}")

# Generate text
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Version History

This model is continuously updated through blockchain consensus. Check the git history for version updates.

## DeSSIN Blockchain

Learn more about DeSSIN (Decentralized Secure Super Intelligence Network):
- GitHub: [DeSSIN Repository](https://github.com/your-org/dessin)
- Paper: PoGO - Proof of Gradient Optimization

## License

Apache 2.0
"""
        return readme
    
    def record_training_iteration(
        self,
        model_id: str,
        block_height: int,
        loss_value: float,
        training_iterations: int,
        model_hash: str
    ) -> bool:
        """Record a training iteration for potential HF update"""
        try:
            # Get or create update record
            if model_id not in self.update_records:
                base_model_info = self.model_manager.get_model_info(model_id)
                if not base_model_info:
                    print(f"Model {model_id} not found")
                    return False
                
                self.update_records[model_id] = ModelUpdateRecord(
                    model_id=model_id,
                    base_model_repo="",  # Will be set when HF repo is created
                    versions=[],
                    latest_version="",
                    total_iterations=0,
                    best_loss=float('inf')
                )
            
            record = self.update_records[model_id]
            
            # Create new version
            version_id = f"v{len(record.versions) + 1}"
            version = ModelVersion(
                version_id=version_id,
                block_height=block_height,
                loss_value=loss_value,
                training_iterations=training_iterations,
                timestamp=time.time(),
                model_hash=model_hash
            )
            
            # Add to record
            record.versions.append(version)
            record.latest_version = version_id
            record.total_iterations += training_iterations
            record.best_loss = min(record.best_loss, loss_value)
            
            print(f"✓ Recorded training iteration for {model_id}")
            print(f"  Version: {version_id}")
            print(f"  Block: {block_height}")
            print(f"  Loss: {loss_value:.6f}")
            print(f"  Total iterations: {record.total_iterations}")
            
            # Check if we should update HuggingFace
            if self._should_update_hf(record):
                return self._update_huggingface_model(model_id, version)
            
            return True
            
        except Exception as e:
            print(f"Error recording training iteration: {e}")
            return False
    
    def _should_update_hf(self, record: ModelUpdateRecord) -> bool:
        """Determine if we should update the HuggingFace model"""
        latest_version = record.versions[-1]
        
        # Check minimum iterations
        if record.total_iterations < self.min_iterations_for_update:
            return False
        
        # Check if this is the first version
        if len(record.versions) == 1:
            return True
        
        # Check loss improvement since last update
        if len(record.versions) >= 2:
            previous_loss = record.versions[-2].loss_value
            improvement = previous_loss - latest_version.loss_value
            
            if improvement >= self.min_loss_improvement:
                return True
        
        # Update every 10 versions regardless
        if len(record.versions) % 10 == 0:
            return True
        
        return False
    
    def _update_huggingface_model(
        self, 
        model_id: str, 
        version: ModelVersion
    ) -> bool:
        """Update the model on HuggingFace"""
        try:
            record = self.update_records[model_id]
            
            # Create HF repository if it doesn't exist
            if not record.base_model_repo:
                base_model_info = self.model_manager.get_model_info(model_id)
                repo_id = self.create_model_repository(model_id, base_model_info)
                
                if repo_id:
                    record.base_model_repo = repo_id
                else:
                    print(f"Failed to create HF repository for {model_id}")
                    return False
            
            # Get model file path
            model_path = self._get_model_file_path(model_id)
            if not model_path or not Path(model_path).exists():
                print(f"Model file not found for {model_id}")
                return False
            
            # Upload model to HuggingFace
            success = self._upload_model_to_hf(
                model_path, 
                record.base_model_repo, 
                version
            )
            
            if success:
                # Update version metadata
                self._update_model_metadata(record.base_model_repo, record)
                print(f"✓ Updated HuggingFace model: {record.base_model_repo}")
                print(f"  Version: {version.version_id}")
                print(f"  Loss: {version.loss_value:.6f}")
                return True
            else:
                print(f"Failed to upload model to HuggingFace")
                return False
                
        except Exception as e:
            print(f"Error updating HuggingFace model: {e}")
            return False
    
    def _get_model_file_path(self, model_id: str) -> Optional[str]:
        """Get the file path for a model"""
        # This would normally find the actual model file
        # For now, return the cache path
        cache_path = self.model_manager.cache_dir / f"{model_id}.gguf"
        if cache_path.exists():
            return str(cache_path)
        
        # Try alternative naming
        cache_path = self.model_manager.cache_dir / f"{model_id}_model.gguf"
        if cache_path.exists():
            return str(cache_path)
        
        return None
    
    def _upload_model_to_hf(
        self, 
        model_path: str, 
        repo_id: str, 
        version: ModelVersion
    ) -> bool:
        """Upload model file to HuggingFace"""
        try:
            if not self.api:
                print(f"Simulated: Would upload {model_path} to {repo_id}")
                return True
            
            # Determine filename
            model_filename = f"model-{version.version_id}.gguf"
            
            # Upload model file
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_filename,
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Update model to {version.version_id} "
                              f"(loss: {version.loss_value:.6f}, "
                              f"block: {version.block_height})"
            )
            
            # Also upload as latest.gguf for easy access
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo="latest.gguf",
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Update latest model to {version.version_id}"
            )
            
            return True
            
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            return False
    
    def _update_model_metadata(
        self, 
        repo_id: str, 
        record: ModelUpdateRecord
    ) -> bool:
        """Update model metadata on HuggingFace"""
        try:
            if not self.api:
                print(f"Simulated: Would update metadata for {repo_id}")
                return True
            
            # Create metadata file
            metadata = {
                "model_id": record.model_id,
                "total_versions": len(record.versions),
                "latest_version": record.latest_version,
                "total_iterations": record.total_iterations,
                "best_loss": record.best_loss,
                "last_updated": time.time(),
                "versions": [v.to_dict() for v in record.versions[-10:]]  # Last 10 versions
            }
            
            # Upload metadata
            upload_file(
                path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                path_in_repo="dessin_metadata.json",
                repo_id=repo_id,
                token=self.hf_token,
                commit_message=f"Update metadata for {record.latest_version}"
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        if model_id in self.update_records:
            return self.update_records[model_id].versions
        return []
    
    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.get_model_versions(model_id)
        return versions[-1] if versions else None
    
    def get_best_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the version with the best (lowest) loss"""
        versions = self.get_model_versions(model_id)
        if not versions:
            return None
        
        return min(versions, key=lambda v: v.loss_value)
    
    def cleanup_old_versions(self, model_id: str) -> int:
        """Clean up old versions to keep storage manageable"""
        if model_id not in self.update_records:
            return 0
        
        record = self.update_records[model_id]
        
        if len(record.versions) <= self.max_versions_per_model:
            return 0
        
        # Keep the best versions and most recent
        versions_to_keep = []
        
        # Keep the 10 most recent
        versions_to_keep.extend(record.versions[-10:])
        
        # Keep the 10 best (lowest loss)
        best_versions = sorted(record.versions, key=lambda v: v.loss_value)[:10]
        versions_to_keep.extend(best_versions)
        
        # Remove duplicates
        unique_versions = {}
        for v in versions_to_keep:
            unique_versions[v.version_id] = v
        
        # Update record
        removed_count = len(record.versions) - len(unique_versions)
        record.versions = list(unique_versions.values())
        
        print(f"Cleaned up {removed_count} old versions for {model_id}")
        return removed_count
    
    def export_training_history(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Export complete training history for a model"""
        if model_id not in self.update_records:
            return None
        
        record = self.update_records[model_id]
        
        return {
            "model_id": model_id,
            "training_history": record.to_dict(),
            "statistics": {
                "total_versions": len(record.versions),
                "total_iterations": record.total_iterations,
                "best_loss": record.best_loss,
                "latest_loss": record.versions[-1].loss_value if record.versions else None,
                "improvement": record.versions[0].loss_value - record.best_loss if record.versions else 0,
                "training_duration": record.versions[-1].timestamp - record.versions[0].timestamp if len(record.versions) > 1 else 0
            }
        }
    
    def list_tracked_models(self) -> List[str]:
        """List all models being tracked for updates"""
        return list(self.update_records.keys())
    
    def get_update_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get update status for a model"""
        if model_id not in self.update_records:
            return None
        
        record = self.update_records[model_id]
        latest = record.versions[-1] if record.versions else None
        
        return {
            "model_id": model_id,
            "hf_repo": record.base_model_repo,
            "total_versions": len(record.versions),
            "latest_version": record.latest_version,
            "latest_loss": latest.loss_value if latest else None,
            "total_iterations": record.total_iterations,
            "best_loss": record.best_loss,
            "needs_update": self._should_update_hf(record) if record.versions else False
        }
