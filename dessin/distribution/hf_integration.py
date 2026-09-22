"""
HuggingFace integration for DeSSIN blockchain.
Downloads and manages GGUF models from trusted publishers.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info, HfApi
from huggingface_hub.utils import HfHubHTTPError

from ..models.model_manager import ModelManager


class HuggingFaceIntegration:
    """Manages HuggingFace model integration for DeSSIN"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.api = HfApi()
        
        # Trusted publishers for quantized models
        self.trusted_publishers = {
            "microsoft", "google", "meta-llama", "huggingface", 
            "TheBloke", "NousResearch", "teknium", "Open-Orca",
            "bartowski", "QuantFactory", "mlabonne"
        }
        
        # Model size categories (in billions of parameters)
        self.size_categories = {
            "tiny": (0, 1),      # <1B parameters
            "small": (1, 7),     # 1-7B parameters  
            "medium": (7, 13),   # 7-13B parameters
            "large": (13, 30),   # 13-30B parameters
            "xlarge": (30, 100), # 30-100B parameters
        }
        
        # Recommended models for testing
        self.recommended_models = [
            "microsoft/DialoGPT-small",  # ~117M parameters
            "microsoft/DialoGPT-medium", # ~354M parameters
            "TheBloke/Llama-2-7B-Chat-GGUF",  # 7B parameters
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # 7B parameters
            "bartowski/gemma-2-2b-it-GGUF",  # 2B parameters
        ]
    
    def search_gguf_models(
        self, 
        size_category: str = "small",
        max_results: int = 10
    ) -> List[Dict]:
        """Search for GGUF models from trusted publishers"""
        try:
            models = []
            
            # Search for GGUF models
            search_results = self.api.list_models(
                filter="gguf",
                limit=100,  # Get more to filter
                sort="downloads",
                direction=-1
            )
            
            min_params, max_params = self.size_categories.get(size_category, (0, 10))
            
            for model in search_results:
                if len(models) >= max_results:
                    break
                
                # Check if from trusted publisher
                author = model.modelId.split("/")[0] if "/" in model.modelId else ""
                if author not in self.trusted_publishers:
                    continue
                
                try:
                    # Get model info
                    info = model_info(model.modelId)
                    
                    # Look for GGUF files
                    gguf_files = [f for f in info.siblings if f.rfilename.endswith('.gguf')]
                    if not gguf_files:
                        continue
                    
                    # Estimate parameters from model size/name
                    estimated_params = self._estimate_parameters(model.modelId, gguf_files)
                    
                    if min_params <= estimated_params <= max_params:
                        models.append({
                            "repo_id": model.modelId,
                            "author": author,
                            "downloads": getattr(model, 'downloads', 0),
                            "estimated_params_b": estimated_params,
                            "gguf_files": [f.rfilename for f in gguf_files],
                            "created_at": getattr(model, 'created_at', None)
                        })
                        
                except Exception as e:
                    print(f"Error processing model {model.modelId}: {e}")
                    continue
            
            return sorted(models, key=lambda x: x['downloads'], reverse=True)
            
        except Exception as e:
            print(f"Error searching GGUF models: {e}")
            return []
    
    def _estimate_parameters(self, repo_id: str, gguf_files: List) -> float:
        """Estimate model parameters from repo name and file sizes"""
        repo_lower = repo_id.lower()
        
        # Extract parameter count from model name
        if "7b" in repo_lower:
            return 7.0
        elif "13b" in repo_lower:
            return 13.0
        elif "30b" in repo_lower or "27b" in repo_lower:
            return 27.0
        elif "2b" in repo_lower:
            return 2.0
        elif "1b" in repo_lower:
            return 1.0
        elif "small" in repo_lower:
            return 0.1
        elif "medium" in repo_lower:
            return 0.4
        elif "large" in repo_lower:
            return 1.5
        
        # Fallback: estimate from largest file size
        if gguf_files:
            # Rough estimation: 1GB ≈ 1B parameters for 4-bit quantized
            max_size = max(getattr(f, 'size', 0) for f in gguf_files if hasattr(f, 'size'))
            return max_size / (1024**3) if max_size > 0 else 1.0
        
        return 1.0  # Default fallback
    
    def download_recommended_model(
        self, 
        size_category: str = "tiny",
        prefer_4bit: bool = True
    ) -> Optional[Tuple[str, str]]:
        """Download a recommended model for testing"""
        try:
            if size_category == "tiny":
                repo_id = "microsoft/DialoGPT-small"
            elif size_category == "small":
                repo_id = "bartowski/gemma-2-2b-it-GGUF"
            else:
                repo_id = "TheBloke/Llama-2-7B-Chat-GGUF"
            
            return self.download_gguf_model(repo_id, prefer_4bit=prefer_4bit)
            
        except Exception as e:
            print(f"Error downloading recommended model: {e}")
            return None
    
    def download_gguf_model(
        self, 
        repo_id: str, 
        filename: Optional[str] = None,
        prefer_4bit: bool = True,
        force_download: bool = False
    ) -> Optional[Tuple[str, str]]:
        """Download a GGUF model from HuggingFace"""
        try:
            # Verify it's from a trusted publisher
            author = repo_id.split("/")[0] if "/" in repo_id else ""
            if author not in self.trusted_publishers:
                print(f"Warning: {author} is not in trusted publishers list")
            
            # Get model info
            info = model_info(repo_id)
            
            # Find GGUF files
            gguf_files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.gguf')]
            if not gguf_files:
                print(f"No GGUF files found in {repo_id}")
                return None
            
            # Select file if not specified
            if filename is None:
                if prefer_4bit:
                    # Look for 4-bit quantized versions
                    q4_files = [f for f in gguf_files if any(q in f.lower() for q in ['q4', '4bit', 'q4_0', 'q4_1'])]
                    if q4_files:
                        filename = q4_files[0]
                    else:
                        # Fall back to smallest file
                        filename = min(gguf_files, key=len)
                else:
                    # Use first file
                    filename = gguf_files[0]
            
            print(f"Downloading {repo_id}/{filename}...")
            
            # Download the file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.model_manager.cache_dir),
                force_download=force_download
            )
            
            print(f"Downloaded to: {local_path}")
            return repo_id, local_path
            
        except HfHubHTTPError as e:
            print(f"HTTP error downloading {repo_id}: {e}")
            return None
        except Exception as e:
            print(f"Error downloading {repo_id}: {e}")
            return None
    
    def register_downloaded_model(
        self, 
        repo_id: str, 
        local_path: str,
        owner_address: str,
        storage_blocks: int = 1000
    ) -> Optional[str]:
        """Register a downloaded model with the DeSSIN network"""
        try:
            # Calculate model properties
            model_hash = self.model_manager.get_model_hash(local_path)
            model_size = self.model_manager.estimate_model_size(local_path)
            
            # Extract model name and parameters
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
            estimated_params = self._estimate_parameters(repo_id, [])
            
            # Generate model ID
            model_id = f"hf_{repo_id.replace('/', '_')}_{model_hash[:8]}"
            
            # Register with model manager
            success = self.model_manager.register_model(
                model_id=model_id,
                name=f"HF: {model_name}",
                size_gb=model_size,
                format="gguf",
                quantization="4bit",  # Assume 4bit for most GGUF files
                parameters=int(estimated_params * 1e9),
                ipfs_hash=f"Qm{model_hash[:44]}",  # Simulated IPFS hash
                owner=owner_address,
                upload_block=0,  # Current block would be set by node
                storage_expires=storage_blocks,
                model_hash=model_hash
            )
            
            if success:
                print(f"Registered model {model_id} with {estimated_params:.1f}B parameters")
                return model_id
            else:
                print(f"Failed to register model {model_id}")
                return None
                
        except Exception as e:
            print(f"Error registering downloaded model: {e}")
            return None
    
    def download_and_register_model(
        self, 
        repo_id: str,
        owner_address: str,
        filename: Optional[str] = None,
        storage_blocks: int = 1000,
        prefer_4bit: bool = True
    ) -> Optional[str]:
        """Download and register a model in one step"""
        try:
            # Download the model
            result = self.download_gguf_model(
                repo_id, 
                filename=filename, 
                prefer_4bit=prefer_4bit
            )
            
            if result is None:
                return None
            
            repo_id, local_path = result
            
            # Register the model
            model_id = self.register_downloaded_model(
                repo_id, 
                local_path, 
                owner_address, 
                storage_blocks
            )
            
            return model_id
            
        except Exception as e:
            print(f"Error downloading and registering model: {e}")
            return None
    
    def list_available_models(self, size_category: str = "small") -> None:
        """List available models for download"""
        print(f"\nAvailable GGUF models ({size_category} size):")
        print("-" * 80)
        
        models = self.search_gguf_models(size_category, max_results=15)
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['repo_id']}")
            print(f"    Author: {model['author']}")
            print(f"    Est. Parameters: {model['estimated_params_b']:.1f}B")
            print(f"    Downloads: {model['downloads']:,}")
            print(f"    GGUF files: {len(model['gguf_files'])}")
            if len(model['gguf_files']) <= 3:
                print(f"    Files: {', '.join(model['gguf_files'])}")
            print()
    
    def validate_model_pairing(self, repo_id: str) -> bool:
        """Validate that a model has both full and quantized versions"""
        try:
            info = model_info(repo_id)
            
            # Look for both quantized and full precision files
            gguf_files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.gguf')]
            
            has_quantized = any(
                any(q in f.lower() for q in ['q4', 'q8', '4bit', '8bit']) 
                for f in gguf_files
            )
            
            has_full_precision = any(
                any(fp in f.lower() for fp in ['f16', 'f32', 'fp16', 'fp32', 'full'])
                for f in gguf_files
            )
            
            return has_quantized  # For now, just require quantized version
            
        except Exception as e:
            print(f"Error validating model pairing for {repo_id}: {e}")
            return False
