# File Upload System for Model Training

**Implementation note:** File records, torrent linkage, and transaction types live in `dessin/model_file_manager.py`, `dessin/file_upload_transactions.py`, and related transaction factories. Behaviour may be narrower than the generic file-type list below; confirm against tests and demos before relying on edge formats in production.

## Overview

The DeSSIN file upload system allows model owners to upload all necessary files for pretraining and fine-tuning phases via BitTorrent. The system is flexible and supports any file type needed for AI model training.

## Supported File Types

### Training Data
- **Datasets**: Training and fine-tuning datasets in any format
  - JSONL, CSV, Parquet
  - Text files
  - Binary formats
  - HuggingFace datasets

### Model Files
- **Checkpoints**: Model checkpoints for resume training
- **Weights**: Model weight files (`.pt`, `.pth`, `.safetensors`)
- **Optimizer State**: Optimizer states for continuing training

### Configuration
- **Config Files**: Training configuration (JSON, YAML)
- **Training Args**: Hyperparameters and settings
- **Tokenizer**: Tokenizer files and vocabularies
- **Vocabulary**: Vocabulary files

### Custom
- **Metadata**: Model metadata and documentation
- **Custom**: Any other files needed for training

## Training Phases

Files are organized by training phase:

1. **Pretraining**: Initial model pretraining files
2. **Fine-Tuning**: Domain/task-specific fine-tuning
3. **Instruction Tuning**: Instruction-following datasets
4. **RLHF**: Reinforcement Learning from Human Feedback
5. **DPO**: Direct Preference Optimization
6. **Custom**: User-defined phases

## Usage

### Upload a Single File

```python
from dessin.model_file_manager import (
    ModelFileManager,
    FileType,
    TrainingPhase
)

file_manager = ModelFileManager(
    storage_dir="model_files",
    torrent_distributor=torrent_dist
)

# Upload dataset
dataset_file = file_manager.upload_file(
    model_id=model_id,
    owner=owner_address,
    file_path="/path/to/dataset.jsonl",
    file_type=FileType.DATASET,
    phase=TrainingPhase.PRETRAINING,
    description="Main pretraining dataset",
    tags=["pretraining", "baseline"],
    create_torrent=True
)

print(f"Uploaded: {dataset_file.file_name}")
print(f"Torrent: {dataset_file.torrent_hash}")
print(f"Magnet: {dataset_file.magnet_link}")
```

### Upload a Directory

```python
# Upload entire directory
bundle = file_manager.upload_directory(
    model_id=model_id,
    owner=owner_address,
    directory_path="/path/to/datasets/",
    file_type=FileType.DATASET,
    phase=TrainingPhase.FINE_TUNING,
    description="All fine-tuning datasets",
    recursive=True,
    create_bundle=True
)

print(f"Bundle: {bundle.name}")
print(f"Files: {len(bundle.files)}")
print(f"Bundle torrent: {bundle.bundle_torrent_hash}")
```

### Create a File Bundle

Group related files into a bundle with a single multi-file torrent:

```python
# Create bundle from existing files
bundle = file_manager.create_bundle(
    model_id=model_id,
    owner=owner_address,
    name="fine_tuning_complete",
    phase=TrainingPhase.FINE_TUNING,
    file_ids=[file1_id, file2_id, file3_id],
    description="Complete fine-tuning package",
    create_bundle_torrent=True
)
```

### Query Files

```python
# Get all files for a model
all_files = file_manager.get_model_files(model_id)

# Filter by type
datasets = file_manager.get_model_files(
    model_id,
    file_type=FileType.DATASET
)

# Filter by phase
pretrain_files = file_manager.get_model_files(
    model_id,
    phase=TrainingPhase.PRETRAINING
)

# Get bundles
bundles = file_manager.get_model_bundles(model_id)
```

### File Summary

```python
# Get comprehensive summary
summary = file_manager.list_model_files_summary(model_id)

print(f"Total files: {summary['total_files']}")
print(f"Total size: {summary['total_size_mb']:.2f} MB")
print(f"Seeding files: {summary['seeding_files']}")

# Phase breakdown
for phase, stats in summary['phase_stats'].items():
    print(f"{phase}: {stats['file_count']} files, {stats['total_size_mb']:.2f} MB")
```

## Blockchain Transactions

### Upload File Transaction

Register an uploaded file on the blockchain:

```python
from dessin.file_upload_transactions import UploadFileTransaction

tx = UploadFileTransaction(
    sender=owner_address,
    model_id=model_id,
    file_id=file.file_id,
    file_name=file.file_name,
    file_hash=file.file_hash,  # SHA256 for verification
    file_size=file.file_size,
    file_type="dataset",
    phase="pretraining",
    torrent_hash=file.torrent_hash,
    magnet_link=file.magnet_link,
    description="Pretraining dataset",
    tags=["baseline", "v1"],
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)

# Process transaction
processor.process_transaction(tx)
```

### Create Bundle Transaction

Register a file bundle on the blockchain:

```python
from dessin.file_upload_transactions import CreateFileBundleTransaction

tx = CreateFileBundleTransaction(
    sender=owner_address,
    model_id=model_id,
    bundle_id=bundle.bundle_id,
    bundle_name="fine_tuning_complete",
    phase="fine_tuning",
    file_ids=[file1_id, file2_id, file3_id],
    bundle_torrent_hash=bundle.bundle_torrent_hash,
    bundle_magnet_link=bundle.bundle_magnet_link,
    description="Complete fine-tuning package",
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)
```

### Delete File Transaction

Stop seeding a file:

```python
from dessin.file_upload_transactions import DeleteFileTransaction

tx = DeleteFileTransaction(
    sender=owner_address,
    file_id=file_id,
    reason="File no longer needed",
    fee=0.01,
    timestamp=time.time(),
    public_key=public_key,
    signature=signature,
    tx_id=tx_id
)
```

## BitTorrent Distribution

All files are distributed via BitTorrent for decentralized access:

### Individual Files
- Each file gets its own torrent
- Single-file torrents for efficient sharing
- Automatic seeding after upload

### File Bundles
- Multi-file torrents for related files
- Download entire training phase at once
- More efficient for large collections

### Torrent Metadata
- Torrent hash stored on-chain
- Magnet links for easy sharing
- Seeder/leecher tracking
- Download statistics

## Storage Organization

Files are organized in storage directory:

```
model_files/
├── model_abc123/
│   ├── pretraining/
│   │   ├── dataset.jsonl
│   │   ├── config.json
│   │   └── tokenizer.json
│   ├── fine_tuning/
│   │   ├── domain_dataset.jsonl
│   │   ├── task_dataset.jsonl
│   │   └── checkpoint_layer1.pt
│   └── rlhf/
│       ├── preference_data.jsonl
│       └── reward_model.pt
└── model_xyz789/
    └── ...
```

## File Metadata

Each file includes:

- **file_id**: Unique identifier
- **file_hash**: SHA256 for verification
- **file_size**: Size in bytes
- **torrent_hash**: BitTorrent info hash
- **magnet_link**: Magnet link for downloads
- **description**: Human-readable description
- **tags**: Searchable tags
- **uploaded_at**: Upload timestamp
- **download_count**: Download statistics
- **seeder_count**: Active seeders

## Common Use Cases

### Pretraining Setup

```python
# Upload pretraining dataset
dataset = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="pretrain_data.jsonl",
    file_type=FileType.DATASET,
    phase=TrainingPhase.PRETRAINING,
    description="100GB pretraining corpus",
    tags=["pretrain", "large-scale"]
)

# Upload config
config = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="config.json",
    file_type=FileType.CONFIG,
    phase=TrainingPhase.PRETRAINING,
    description="Training hyperparameters"
)

# Upload tokenizer
tokenizer = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="tokenizer.json",
    file_type=FileType.TOKENIZER,
    phase=TrainingPhase.PRETRAINING,
    description="Custom tokenizer (50k vocab)"
)
```

### Fine-Tuning Setup

```python
# Upload multiple datasets
datasets_dir = "fine_tuning_datasets/"
bundle = file_manager.upload_directory(
    model_id=model_id,
    owner=owner,
    directory_path=datasets_dir,
    file_type=FileType.DATASET,
    phase=TrainingPhase.FINE_TUNING,
    recursive=True,
    create_bundle=True
)

# Upload checkpoint to resume from
checkpoint = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="checkpoint_epoch_10.pt",
    file_type=FileType.CHECKPOINT,
    phase=TrainingPhase.FINE_TUNING,
    description="Checkpoint after pretraining"
)
```

### RLHF Setup

```python
# Upload preference dataset
preferences = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="human_preferences.jsonl",
    file_type=FileType.DATASET,
    phase=TrainingPhase.RLHF,
    description="Human preference comparisons",
    tags=["rlhf", "preferences", "human-labeled"]
)

# Upload reward model
reward_model = file_manager.upload_file(
    model_id=model_id,
    owner=owner,
    file_path="reward_model.pt",
    file_type=FileType.WEIGHTS,
    phase=TrainingPhase.RLHF,
    description="Trained reward model"
)
```

## Best Practices

### File Organization
- Use descriptive file names
- Tag files appropriately
- Group related files in bundles
- Organize by training phase

### BitTorrent Seeding
- Keep seeding important files
- Use bundles for related files
- Monitor seeder counts
- Stop seeding obsolete files

### File Verification
- Always check file hashes
- Verify torrent integrity
- Use blockchain metadata
- Track file versions

### Storage Management
- Monitor storage usage
- Delete obsolete files
- Compress large datasets
- Use efficient formats

## Demo

Run the demo to see the system in action:

```bash
./bin/python e2e/scripts/model_file_upload_demo.py
```

This demonstrates:
- Uploading files of different types
- Organizing by training phase
- Creating file bundles
- BitTorrent distribution
- Blockchain transactions
- File querying and summary

## File Types Reference

```python
class FileType(Enum):
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    VOCABULARY = "vocabulary"
    METADATA = "metadata"
    WEIGHTS = "weights"
    OPTIMIZER_STATE = "optimizer_state"
    TRAINING_ARGS = "training_args"
    CUSTOM = "custom"

class TrainingPhase(Enum):
    PRETRAINING = "pretraining"
    FINE_TUNING = "fine_tuning"
    INSTRUCTION_TUNING = "instruction_tuning"
    RLHF = "rlhf"
    DPO = "dpo"
    CUSTOM = "custom"
```

## Benefits

✓ **Flexible**: Upload any file type
✓ **Organized**: By phase and type
✓ **Decentralized**: BitTorrent distribution
✓ **Verifiable**: On-chain metadata and hashes
✓ **Scalable**: Bundles for large collections
✓ **Efficient**: Torrent-based sharing
✓ **Trackable**: Download and seeder statistics
✓ **Owner-Controlled**: Full control over files

Model owners have complete flexibility in uploading and managing all files needed for their training pipelines!
