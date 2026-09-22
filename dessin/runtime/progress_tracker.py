"""
Progress tracking and display for DeSSIN model distribution.
Provides console progress bars for downloads, uploads, and gradient updates.
"""

import time
import threading
import sys
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressType(Enum):
    DOWNLOAD = "download"
    UPLOAD = "upload"
    GRADIENT_UPDATE = "gradient_update"
    TORRENT_CREATION = "torrent_creation"
    MODEL_TRAINING = "model_training"


@dataclass
class ProgressInfo:
    """Information about an ongoing progress operation"""
    operation_id: str
    operation_type: ProgressType
    description: str
    total_size: int  # bytes
    current_size: int = 0
    start_time: float = 0.0
    last_update: float = 0.0
    speed_bytes_per_sec: float = 0.0
    eta_seconds: Optional[int] = None
    status: str = "running"  # running, completed, failed, paused
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage (0-100)"""
        if self.total_size <= 0:
            return 0.0
        return min(100.0, (self.current_size / self.total_size) * 100.0)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time if self.start_time > 0 else 0.0


class SimpleProgressBar:
    """Simple progress bar implementation when tqdm is not available"""
    
    def __init__(self, total: int, desc: str = "", unit: str = "B", unit_scale: bool = True):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.unit_scale = unit_scale
        self.current = 0
        self.start_time = time.time()
        self.last_print_time = 0
        self.width = 50  # Progress bar width
        
    def update(self, n: int = 1):
        """Update progress by n units"""
        self.current = min(self.current + n, self.total)
        current_time = time.time()
        
        # Only update display every 0.1 seconds to avoid spam
        if current_time - self.last_print_time >= 0.1:
            self._display()
            self.last_print_time = current_time
    
    def set_current(self, current: int):
        """Set current progress value"""
        self.current = min(current, self.total)
        self._display()
    
    def _display(self):
        """Display the progress bar"""
        if self.total <= 0:
            return
        
        percent = (self.current / self.total) * 100
        filled_width = int(self.width * self.current / self.total)
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        # Calculate speed and ETA
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.current > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0
        
        # Format size
        if self.unit_scale and self.unit == "B":
            current_str = self._format_bytes(self.current)
            total_str = self._format_bytes(self.total)
            speed_str = f"{self._format_bytes(speed)}/s"
        else:
            current_str = f"{self.current}{self.unit}"
            total_str = f"{self.total}{self.unit}"
            speed_str = f"{speed:.1f}{self.unit}/s"
        
        eta_str = self._format_time(eta) if eta > 0 else "∞"
        
        # Print progress line
        line = (f"\r{self.desc}: {percent:6.1f}%|{bar}| "
                f"{current_str}/{total_str} [{speed_str}, ETA: {eta_str}]")
        
        print(line, end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def _format_bytes(self, bytes_val: float) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}PB"
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m{seconds%60:.0f}s"
        else:
            return f"{seconds/3600:.0f}h{(seconds%3600)/60:.0f}m"
    
    def close(self):
        """Close the progress bar"""
        if self.current < self.total:
            print()  # Ensure new line


class ProgressTracker:
    """
    Tracks and displays progress for multiple concurrent operations.
    Supports both tqdm and simple fallback progress bars.
    """
    
    def __init__(self, use_simple_bars: bool = False):
        self.use_simple_bars = use_simple_bars or not TQDM_AVAILABLE
        self.operations: Dict[str, ProgressInfo] = {}
        self.progress_bars: Dict[str, Any] = {}  # tqdm or SimpleProgressBar instances
        self.callbacks: Dict[str, Callable] = {}
        # Use re-entrant lock to avoid deadlocks when methods call each other
        self.lock = threading.RLock()
        
        if not TQDM_AVAILABLE and not use_simple_bars:
            print("⚠️  tqdm not available, using simple progress bars")
            self.use_simple_bars = True
    
    def start_operation(
        self,
        operation_id: str,
        operation_type: ProgressType,
        description: str,
        total_size: int,
        callback: Optional[Callable] = None
    ) -> bool:
        """Start tracking a new operation"""
        with self.lock:
            if operation_id in self.operations:
                return False  # Operation already exists
            
            # Create progress info
            progress_info = ProgressInfo(
                operation_id=operation_id,
                operation_type=operation_type,
                description=description,
                total_size=total_size,
                start_time=time.time()
            )
            
            self.operations[operation_id] = progress_info
            
            if callback:
                self.callbacks[operation_id] = callback
            
            # Create progress bar
            self._create_progress_bar(operation_id, progress_info)
            
            return True
    
    def _create_progress_bar(self, operation_id: str, progress_info: ProgressInfo):
        """Create a progress bar for the operation"""
        try:
            if self.use_simple_bars:
                # Use simple progress bar
                bar = SimpleProgressBar(
                    total=progress_info.total_size,
                    desc=progress_info.description,
                    unit="B",
                    unit_scale=True
                )
            else:
                # Use tqdm progress bar
                bar = tqdm(
                    total=progress_info.total_size,
                    desc=progress_info.description,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=100,
                    leave=True
                )
            
            self.progress_bars[operation_id] = bar
            
        except Exception as e:
            print(f"Error creating progress bar for {operation_id}: {e}")
    
    def update_progress(
        self,
        operation_id: str,
        current_size: int,
        speed_bytes_per_sec: Optional[float] = None
    ) -> bool:
        """Update progress for an operation"""
        with self.lock:
            if operation_id not in self.operations:
                return False
            
            progress_info = self.operations[operation_id]
            old_size = progress_info.current_size
            progress_info.current_size = min(current_size, progress_info.total_size)
            progress_info.last_update = time.time()
            
            if speed_bytes_per_sec is not None:
                progress_info.speed_bytes_per_sec = speed_bytes_per_sec
            else:
                # Calculate speed from progress
                elapsed = progress_info.elapsed_time
                if elapsed > 0:
                    progress_info.speed_bytes_per_sec = progress_info.current_size / elapsed
            
            # Calculate ETA
            if progress_info.speed_bytes_per_sec > 0:
                remaining = progress_info.total_size - progress_info.current_size
                progress_info.eta_seconds = int(remaining / progress_info.speed_bytes_per_sec)
            
            # Update progress bar
            if operation_id in self.progress_bars:
                bar = self.progress_bars[operation_id]
                
                if self.use_simple_bars:
                    bar.set_current(progress_info.current_size)
                else:
                    # Update tqdm bar
                    delta = progress_info.current_size - old_size
                    if delta > 0:
                        bar.update(delta)
                    
                    # Update postfix with speed and ETA
                    if hasattr(bar, 'set_postfix'):
                        postfix = {}
                        if progress_info.speed_bytes_per_sec > 0:
                            speed_mb = progress_info.speed_bytes_per_sec / 1024 / 1024
                            postfix['speed'] = f"{speed_mb:.1f}MB/s"
                        if progress_info.eta_seconds:
                            postfix['eta'] = f"{progress_info.eta_seconds}s"
                        bar.set_postfix(postfix)
            
            # Call callback if provided
            if operation_id in self.callbacks:
                try:
                    self.callbacks[operation_id](progress_info)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
            
            return True
    
    def complete_operation(self, operation_id: str, success: bool = True):
        """Mark an operation as completed"""
        with self.lock:
            if operation_id not in self.operations:
                return
            
            progress_info = self.operations[operation_id]
            progress_info.status = "completed" if success else "failed"
            
            # Close progress bar
            if operation_id in self.progress_bars:
                bar = self.progress_bars[operation_id]
                
                if self.use_simple_bars:
                    if success:
                        bar.set_current(progress_info.total_size)
                    bar.close()
                else:
                    if success and hasattr(bar, 'n'):
                        # Ensure tqdm shows 100%
                        remaining = progress_info.total_size - bar.n
                        if remaining > 0:
                            bar.update(remaining)
                    bar.close()
                
                del self.progress_bars[operation_id]
            
            # Clean up callback
            self.callbacks.pop(operation_id, None)
            
            # Print completion message
            if success:
                elapsed = progress_info.elapsed_time
                total_mb = progress_info.total_size / 1024 / 1024
                avg_speed = total_mb / elapsed if elapsed > 0 else 0
                print(f"✓ {progress_info.description} completed in {elapsed:.1f}s (avg: {avg_speed:.1f} MB/s)")
            else:
                print(f"❌ {progress_info.description} failed")
    
    def fail_operation(self, operation_id: str, error_message: str = ""):
        """Mark an operation as failed"""
        # First, mark status under lock
        with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = "failed"
        # Then, complete outside the inner lock scope to avoid nested lock contention
        self.complete_operation(operation_id, success=False)
        if error_message:
            print(f"   Error: {error_message}")
    
    def pause_operation(self, operation_id: str):
        """Pause an operation"""
        with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = "paused"
    
    def resume_operation(self, operation_id: str):
        """Resume a paused operation"""
        with self.lock:
            if operation_id in self.operations:
                self.operations[operation_id].status = "active"
    
    def get_operation_info(self, operation_id: str) -> Optional[ProgressInfo]:
        """Get progress information for an operation"""
        with self.lock:
            return self.operations.get(operation_id)
    
    def get_all_operations(self) -> Dict[str, ProgressInfo]:
        """Get all current operations"""
        with self.lock:
            return self.operations.copy()
    
    def cleanup_completed(self, max_age_seconds: float = 300):
        """Clean up completed operations older than max_age_seconds"""
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for operation_id, progress_info in self.operations.items():
                if (progress_info.status in ["completed", "failed"] and 
                    current_time - progress_info.last_update > max_age_seconds):
                    to_remove.append(operation_id)
            
            for operation_id in to_remove:
                del self.operations[operation_id]
                # Progress bars should already be closed
    
    def print_status_summary(self):
        """Print a summary of all operations"""
        with self.lock:
            if not self.operations:
                print("No active operations")
                return
            
            print("\n📊 Progress Summary:")
            print("-" * 80)
            
            for operation_id, info in self.operations.items():
                status_icon = {
                    "active": "🔄",
                    "completed": "✓",
                    "failed": "❌",
                    "paused": "⏸️"
                }.get(info.status, "❓")
                
                print(f"{status_icon} {info.description}")
                print(f"   Progress: {info.progress_percent:.1f}% "
                      f"({info.current_size:,} / {info.total_size:,} bytes)")
                
                if info.speed_bytes_per_sec > 0:
                    speed_mb = info.speed_bytes_per_sec / 1024 / 1024
                    print(f"   Speed: {speed_mb:.1f} MB/s")
                
                if info.eta_seconds:
                    print(f"   ETA: {info.eta_seconds}s")
                
                print(f"   Elapsed: {info.elapsed_time:.1f}s")
                print()


# Global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def track_download(operation_id: str, description: str, total_size: int) -> str:
    """Convenience function to start tracking a download"""
    tracker = get_progress_tracker()
    tracker.start_operation(operation_id, ProgressType.DOWNLOAD, description, total_size)
    return operation_id


def track_upload(operation_id: str, description: str, total_size: int) -> str:
    """Convenience function to start tracking an upload"""
    tracker = get_progress_tracker()
    tracker.start_operation(operation_id, ProgressType.UPLOAD, description, total_size)
    return operation_id


def track_gradient_update(operation_id: str, description: str, total_steps: int) -> str:
    """Convenience function to start tracking gradient updates"""
    tracker = get_progress_tracker()
    tracker.start_operation(operation_id, ProgressType.GRADIENT_UPDATE, description, total_steps)
    return operation_id


def update_progress(operation_id: str, current: int, speed: Optional[float] = None):
    """Convenience function to update progress"""
    tracker = get_progress_tracker()
    tracker.update_progress(operation_id, current, speed)


def complete_progress(operation_id: str, success: bool = True):
    """Convenience function to complete progress"""
    tracker = get_progress_tracker()
    tracker.complete_operation(operation_id, success)


def fail_progress(operation_id: str, error_message: str = ""):
    """Convenience function to fail progress"""
    tracker = get_progress_tracker()
    tracker.fail_operation(operation_id, error_message)
