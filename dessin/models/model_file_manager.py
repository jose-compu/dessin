"""
Model File Manager for DeSSIN
==============================

Manages file uploads for model owners via BitTorrent distribution.
Supports datasets, checkpoints, configs, and any other files needed for training.
"""

import os
import hashlib
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class FileType(Enum):
    """Types of files that can be uploaded"""
    DATASET = "dataset"  # Training/fine-tuning datasets
    CHECKPOINT = "checkpoint"  # Model checkpoints
    CONFIG = "config"  # Configuration files
    TOKENIZER = "tokenizer"  # Tokenizer files
    VOCABULARY = "vocabulary"  # Vocabulary files
    METADATA = "metadata"  # Model metadata
    WEIGHTS = "weights"  # Model weights
    OPTIMIZER_STATE = "optimizer_state"  # Optimizer state for resume
    TRAINING_ARGS = "training_args"  # Training arguments
    CUSTOM = "custom"  # Any other file type


class TrainingPhase(Enum):
    """Training phase for file organization"""
    PRETRAINING = "pretraining"
    FINE_TUNING = "fine_tuning"
    INSTRUCTION_TUNING = "instruction_tuning"
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback
    DPO = "dpo"  # Direct Preference Optimization
    CUSTOM = "custom"


@dataclass
class ModelFile:
    """Information about an uploaded file"""
    file_id: str
    model_id: str
    owner: str
    file_name: str
    file_path: str
    file_type: FileType
    phase: TrainingPhase
    file_size: int
    file_hash: str  # SHA256 hash
    
    # BitTorrent distribution
    torrent_hash: Optional[str] = None
    magnet_link: Optional[str] = None
    is_seeding: bool = False
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    mime_type: Optional[str] = None
    
    # Timestamps
    uploaded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # Usage tracking
    download_count: int = 0
    seeder_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_id': self.file_id,
            'model_id': self.model_id,
            'owner': self.owner,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'file_type': self.file_type.value,
            'phase': self.phase.value,
            'file_size': self.file_size,
            'file_hash': self.file_hash,
            'torrent_hash': self.torrent_hash,
            'magnet_link': self.magnet_link,
            'is_seeding': self.is_seeding,
            'description': self.description,
            'tags': self.tags,
            'mime_type': self.mime_type,
            'uploaded_at': self.uploaded_at,
            'last_accessed': self.last_accessed,
            'download_count': self.download_count,
            'seeder_count': self.seeder_count
        }


@dataclass
class FileBundle:
    """Collection of related files (e.g., all files for a training phase)"""
    bundle_id: str
    model_id: str
    owner: str
    name: str
    phase: TrainingPhase
    files: List[str] = field(default_factory=list)  # List of file_ids
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    # Bundle torrent (multi-file torrent)
    bundle_torrent_hash: Optional[str] = None
    bundle_magnet_link: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bundle_id': self.bundle_id,
            'model_id': self.model_id,
            'owner': self.owner,
            'name': self.name,
            'phase': self.phase.value,
            'files': self.files,
            'description': self.description,
            'created_at': self.created_at,
            'bundle_torrent_hash': self.bundle_torrent_hash,
            'bundle_magnet_link': self.bundle_magnet_link
        }


class ModelFileManager:
    """Manages file uploads and distribution for models"""
    
    def __init__(
        self,
        storage_dir: str = "model_files",
        torrent_distributor=None
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File tracking
        self.files: Dict[str, ModelFile] = {}
        self.bundles: Dict[str, FileBundle] = {}
        
        # Index for quick lookups
        self.model_files: Dict[str, List[str]] = {}  # model_id -> [file_ids]
        self.phase_files: Dict[TrainingPhase, List[str]] = {}  # phase -> [file_ids]
        
        # BitTorrent distributor
        self.torrent_distributor = torrent_distributor
        
        print("✓ Model file manager initialized")
        print(f"  Storage directory: {self.storage_dir}")
    
    def upload_file(
        self,
        model_id: str,
        owner: str,
        file_path: str,
        file_type: FileType,
        phase: TrainingPhase,
        description: str = "",
        tags: Optional[List[str]] = None,
        create_torrent: bool = True
    ) -> Optional[ModelFile]:
        """Upload a file for a model"""
        
        # Validate file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        # Generate file ID
        file_id = self._generate_file_id(model_id, file_name, file_hash)
        
        # Copy file to storage
        storage_path = self._get_storage_path(model_id, phase, file_name)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(file_path, storage_path)
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_name)
        
        # Create file record
        model_file = ModelFile(
            file_id=file_id,
            model_id=model_id,
            owner=owner,
            file_name=file_name,
            file_path=str(storage_path),
            file_type=file_type,
            phase=phase,
            file_size=file_size,
            file_hash=file_hash,
            description=description,
            tags=tags or [],
            mime_type=mime_type
        )
        
        # Create torrent if requested and distributor available
        if create_torrent and self.torrent_distributor:
            torrent_info = self._create_file_torrent(model_file)
            if torrent_info:
                model_file.torrent_hash = torrent_info['hash']
                model_file.magnet_link = torrent_info['magnet']
                model_file.is_seeding = True
        
        # Store file record
        self.files[file_id] = model_file
        
        # Update indices
        if model_id not in self.model_files:
            self.model_files[model_id] = []
        self.model_files[model_id].append(file_id)
        
        if phase not in self.phase_files:
            self.phase_files[phase] = []
        self.phase_files[phase].append(file_id)
        
        print(f"✓ Uploaded file: {file_name}")
        print(f"  File ID: {file_id}")
        print(f"  Type: {file_type.value}")
        print(f"  Phase: {phase.value}")
        print(f"  Size: {file_size / 1024:.2f} KB")
        if model_file.torrent_hash:
            print(f"  Torrent: {model_file.torrent_hash[:16]}...")
        
        return model_file
    
    def upload_directory(
        self,
        model_id: str,
        owner: str,
        directory_path: str,
        file_type: FileType,
        phase: TrainingPhase,
        description: str = "",
        recursive: bool = True,
        create_bundle: bool = True
    ) -> Optional[FileBundle]:
        """Upload an entire directory of files"""
        
        if not os.path.isdir(directory_path):
            print(f"Directory not found: {directory_path}")
            return None
        
        print(f"Uploading directory: {directory_path}")
        print(f"  Type: {file_type.value}")
        print(f"  Phase: {phase.value}")
        print()
        
        uploaded_files = []
        
        # Walk directory
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(file_path, directory_path)
                    
                    model_file = self.upload_file(
                        model_id=model_id,
                        owner=owner,
                        file_path=file_path,
                        file_type=file_type,
                        phase=phase,
                        description=f"{description} - {rel_path}",
                        create_torrent=False  # Create bundle torrent later
                    )
                    
                    if model_file:
                        uploaded_files.append(model_file.file_id)
        else:
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    model_file = self.upload_file(
                        model_id=model_id,
                        owner=owner,
                        file_path=file_path,
                        file_type=file_type,
                        phase=phase,
                        description=description,
                        create_torrent=False
                    )
                    
                    if model_file:
                        uploaded_files.append(model_file.file_id)
        
        # Create bundle
        if create_bundle and uploaded_files:
            bundle = self.create_bundle(
                model_id=model_id,
                owner=owner,
                name=f"{phase.value}_{file_type.value}",
                phase=phase,
                file_ids=uploaded_files,
                description=description,
                create_bundle_torrent=True
            )
            
            return bundle
        
        return None
    
    def create_bundle(
        self,
        model_id: str,
        owner: str,
        name: str,
        phase: TrainingPhase,
        file_ids: List[str],
        description: str = "",
        create_bundle_torrent: bool = True
    ) -> Optional[FileBundle]:
        """Create a bundle of related files"""
        
        # Validate all files belong to same model
        for file_id in file_ids:
            if file_id not in self.files:
                print(f"File not found: {file_id}")
                return None
            if self.files[file_id].model_id != model_id:
                print(f"File {file_id} does not belong to model {model_id}")
                return None
        
        # Generate bundle ID
        bundle_data = f"{model_id}:{name}:{time.time()}"
        bundle_id = f"bundle_{hashlib.sha256(bundle_data.encode()).hexdigest()[:16]}"
        
        # Create bundle
        bundle = FileBundle(
            bundle_id=bundle_id,
            model_id=model_id,
            owner=owner,
            name=name,
            phase=phase,
            files=file_ids,
            description=description
        )
        
        # Create multi-file torrent if requested
        if create_bundle_torrent and self.torrent_distributor:
            file_paths = [self.files[fid].file_path for fid in file_ids]
            bundle_torrent = self._create_bundle_torrent(bundle_id, file_paths)
            
            if bundle_torrent:
                bundle.bundle_torrent_hash = bundle_torrent['hash']
                bundle.bundle_magnet_link = bundle_torrent['magnet']
        
        # Store bundle
        self.bundles[bundle_id] = bundle
        
        print(f"✓ Created bundle: {name}")
        print(f"  Bundle ID: {bundle_id}")
        print(f"  Files: {len(file_ids)}")
        print(f"  Phase: {phase.value}")
        if bundle.bundle_torrent_hash:
            print(f"  Bundle torrent: {bundle.bundle_torrent_hash[:16]}...")
        
        return bundle
    
    def get_model_files(
        self,
        model_id: str,
        file_type: Optional[FileType] = None,
        phase: Optional[TrainingPhase] = None
    ) -> List[ModelFile]:
        """Get all files for a model, optionally filtered"""
        
        file_ids = self.model_files.get(model_id, [])
        files = [self.files[fid] for fid in file_ids if fid in self.files]
        
        # Apply filters
        if file_type:
            files = [f for f in files if f.file_type == file_type]
        
        if phase:
            files = [f for f in files if f.phase == phase]
        
        return files
    
    def get_model_bundles(self, model_id: str, phase: Optional[TrainingPhase] = None) -> List[FileBundle]:
        """Get all bundles for a model"""
        
        bundles = [b for b in self.bundles.values() if b.model_id == model_id]
        
        if phase:
            bundles = [b for b in bundles if b.phase == phase]
        
        return bundles
    
    def get_file(self, file_id: str) -> Optional[ModelFile]:
        """Get a specific file"""
        return self.files.get(file_id)
    
    def get_bundle(self, bundle_id: str) -> Optional[FileBundle]:
        """Get a specific bundle"""
        return self.bundles.get(bundle_id)
    
    def download_file(self, file_id: str) -> Optional[str]:
        """Download a file (returns local path or downloads via torrent)"""
        
        file = self.files.get(file_id)
        if not file:
            return None
        
        # Update access time and download count
        file.last_accessed = time.time()
        file.download_count += 1
        
        # If file exists locally, return path
        if os.path.exists(file.file_path):
            return file.file_path
        
        # Otherwise, try to download via torrent
        if file.torrent_hash and self.torrent_distributor:
            print(f"Downloading {file.file_name} via BitTorrent...")
            # Would implement actual torrent download here
            return None
        
        return None
    
    def list_model_files_summary(self, model_id: str) -> Dict[str, Any]:
        """Get summary of all files for a model"""
        
        files = self.get_model_files(model_id)
        bundles = self.get_model_bundles(model_id)
        
        # Calculate totals by phase
        phase_stats = {}
        for phase in TrainingPhase:
            phase_files = [f for f in files if f.phase == phase]
            if phase_files:
                total_size = sum(f.file_size for f in phase_files)
                phase_stats[phase.value] = {
                    'file_count': len(phase_files),
                    'total_size': total_size,
                    'total_size_mb': total_size / (1024 * 1024)
                }
        
        # File type breakdown
        type_stats = {}
        for file_type in FileType:
            type_files = [f for f in files if f.file_type == file_type]
            if type_files:
                type_stats[file_type.value] = len(type_files)
        
        return {
            'model_id': model_id,
            'total_files': len(files),
            'total_bundles': len(bundles),
            'total_size': sum(f.file_size for f in files),
            'total_size_mb': sum(f.file_size for f in files) / (1024 * 1024),
            'phase_stats': phase_stats,
            'type_stats': type_stats,
            'seeding_files': len([f for f in files if f.is_seeding])
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _generate_file_id(self, model_id: str, file_name: str, file_hash: str) -> str:
        """Generate unique file ID"""
        data = f"{model_id}:{file_name}:{file_hash}"
        return f"file_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def _get_storage_path(self, model_id: str, phase: TrainingPhase, file_name: str) -> Path:
        """Get storage path for a file"""
        return self.storage_dir / model_id / phase.value / file_name
    
    def _detect_mime_type(self, file_name: str) -> str:
        """Detect MIME type from filename"""
        ext = os.path.splitext(file_name)[1].lower()
        
        mime_types = {
            '.json': 'application/json',
            '.jsonl': 'application/jsonl',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.parquet': 'application/parquet',
            '.pt': 'application/pytorch',
            '.pth': 'application/pytorch',
            '.safetensors': 'application/safetensors',
            '.bin': 'application/octet-stream',
            '.pkl': 'application/pickle',
            '.h5': 'application/hdf5',
            '.yaml': 'application/yaml',
            '.yml': 'application/yaml',
        }
        
        return mime_types.get(ext, 'application/octet-stream')
    
    def _create_file_torrent(self, model_file: ModelFile) -> Optional[Dict[str, str]]:
        """Create torrent for a single file"""
        
        if not self.torrent_distributor:
            return None
        
        try:
            # Use real torrent distributor if available
            if hasattr(self.torrent_distributor, 'create_real_torrent'):
                torrent_info = self.torrent_distributor.create_real_torrent(
                    model_file.file_path,
                    model_file.file_id
                )
                
                if torrent_info:
                    return {
                        'hash': torrent_info.torrent_hash,
                        'magnet': torrent_info.magnet_link
                    }
            
            # Fallback to mock
            return {
                'hash': f"mock_{model_file.file_hash[:16]}",
                'magnet': f"magnet:?xt=urn:btih:{model_file.file_hash[:40]}"
            }
        
        except Exception as e:
            print(f"Error creating torrent: {e}")
            return None
    
    def _create_bundle_torrent(self, bundle_id: str, file_paths: List[str]) -> Optional[Dict[str, str]]:
        """Create multi-file torrent for a bundle"""
        
        # For now, create a simple hash based on all files
        combined_hash = hashlib.sha256(bundle_id.encode())
        for path in file_paths:
            with open(path, 'rb') as f:
                combined_hash.update(f.read())
        
        bundle_hash = combined_hash.hexdigest()
        
        return {
            'hash': f"bundle_{bundle_hash[:16]}",
            'magnet': f"magnet:?xt=urn:btih:{bundle_hash[:40]}&dn={bundle_id}"
        }
