"""
Blockchain Transactions for File Uploads
=========================================

Allows model owners to upload files via blockchain transactions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from ..consensus.transactions import BaseTransaction


@dataclass
class UploadFileTransaction(BaseTransaction):
    """Transaction to register an uploaded file"""
    model_id: str
    file_id: str
    file_name: str
    file_hash: str  # SHA256 hash for verification
    file_size: int
    file_type: str  # "dataset", "checkpoint", "config", etc.
    phase: str  # "pretraining", "fine_tuning", etc.
    torrent_hash: Optional[str] = None
    magnet_link: Optional[str] = None
    description: str = ""
    tags: Optional[List[str]] = None
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'upload_file',
            'model_id': self.model_id,
            'file_id': self.file_id,
            'file_name': self.file_name,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'phase': self.phase,
            'torrent_hash': self.torrent_hash,
            'magnet_link': self.magnet_link,
            'description': self.description,
            'tags': self.tags or [],
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


@dataclass
class CreateFileBundleTransaction(BaseTransaction):
    """Transaction to create a bundle of files"""
    model_id: str
    bundle_id: str
    bundle_name: str
    phase: str
    file_ids: List[str]
    bundle_torrent_hash: Optional[str] = None
    bundle_magnet_link: Optional[str] = None
    description: str = ""
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'create_file_bundle',
            'model_id': self.model_id,
            'bundle_id': self.bundle_id,
            'bundle_name': self.bundle_name,
            'phase': self.phase,
            'file_ids': self.file_ids,
            'bundle_torrent_hash': self.bundle_torrent_hash,
            'bundle_magnet_link': self.bundle_magnet_link,
            'description': self.description,
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


@dataclass
class DeleteFileTransaction(BaseTransaction):
    """Transaction to mark a file as deleted (stops seeding)"""
    file_id: str
    reason: str = ""
    
    def get_transaction_data(self) -> Dict[str, Any]:
        return {
            'type': 'delete_file',
            'file_id': self.file_id,
            'reason': self.reason,
            'sender': self.sender,
            'fee': self.fee,
            'timestamp': self.timestamp
        }


class FileUploadTransactionProcessor:
    """Processes file upload transactions"""
    
    def __init__(self, file_manager, model_manager):
        self.file_manager = file_manager
        self.model_manager = model_manager
    
    def process_transaction(self, transaction: BaseTransaction) -> bool:
        """Process a file upload transaction"""
        if isinstance(transaction, UploadFileTransaction):
            return self._process_upload_file(transaction)
        
        elif isinstance(transaction, CreateFileBundleTransaction):
            return self._process_create_bundle(transaction)
        
        elif isinstance(transaction, DeleteFileTransaction):
            return self._process_delete_file(transaction)
        
        return False
    
    def _process_upload_file(self, tx: UploadFileTransaction) -> bool:
        """Process file upload registration"""
        # Verify model exists and sender is owner
        model = self.model_manager.models.get(tx.model_id)
        if not model:
            print(f"Model not found: {tx.model_id}")
            return False
        
        if model.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of {tx.model_id}")
            return False
        
        print(f"✓ Processed UploadFileTransaction")
        print(f"  Model: {tx.model_id}")
        print(f"  File: {tx.file_name}")
        print(f"  Size: {tx.file_size / 1024:.2f} KB")
        print(f"  Type: {tx.file_type}")
        print(f"  Phase: {tx.phase}")
        if tx.torrent_hash:
            print(f"  Torrent: {tx.torrent_hash[:16]}...")
        
        return True
    
    def _process_create_bundle(self, tx: CreateFileBundleTransaction) -> bool:
        """Process bundle creation"""
        model = self.model_manager.models.get(tx.model_id)
        if not model:
            print(f"Model not found: {tx.model_id}")
            return False
        
        if model.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of {tx.model_id}")
            return False
        
        print(f"✓ Processed CreateFileBundleTransaction")
        print(f"  Model: {tx.model_id}")
        print(f"  Bundle: {tx.bundle_name}")
        print(f"  Files: {len(tx.file_ids)}")
        print(f"  Phase: {tx.phase}")
        
        return True
    
    def _process_delete_file(self, tx: DeleteFileTransaction) -> bool:
        """Process file deletion"""
        file = self.file_manager.get_file(tx.file_id)
        if not file:
            print(f"File not found: {tx.file_id}")
            return False
        
        if file.owner != tx.sender:
            print(f"Unauthorized: {tx.sender} is not owner of file {tx.file_id}")
            return False
        
        # Mark as not seeding
        file.is_seeding = False
        
        print(f"✓ Processed DeleteFileTransaction")
        print(f"  File: {file.file_name}")
        print(f"  Reason: {tx.reason}")
        
        return True
