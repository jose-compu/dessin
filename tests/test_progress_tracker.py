"""
Unit tests for progress tracker functionality.
"""

import unittest
import time
import threading
from unittest.mock import patch

from dessin.runtime.progress_tracker import (
    ProgressTracker, ProgressType, ProgressInfo, SimpleProgressBar,
    get_progress_tracker, track_download, update_progress, complete_progress
)


class TestProgressInfo(unittest.TestCase):
    """Test ProgressInfo data class"""
    
    def test_progress_info_creation(self):
        """Test creating ProgressInfo objects"""
        progress = ProgressInfo(
            operation_id="test_op",
            operation_type=ProgressType.DOWNLOAD,
            description="Test download",
            total_size=1024*1024,
            current_size=512*1024,
            start_time=time.time()
        )
        
        self.assertEqual(progress.operation_id, "test_op")
        self.assertEqual(progress.operation_type, ProgressType.DOWNLOAD)
        self.assertEqual(progress.total_size, 1024*1024)
        self.assertEqual(progress.current_size, 512*1024)
        self.assertEqual(progress.progress_percent, 50.0)
        self.assertGreater(progress.elapsed_time, 0)
    
    def test_progress_percent_calculation(self):
        """Test progress percentage calculation"""
        progress = ProgressInfo(
            operation_id="test",
            operation_type=ProgressType.UPLOAD,
            description="Test",
            total_size=1000,
            current_size=250
        )
        
        self.assertEqual(progress.progress_percent, 25.0)
        
        # Test edge cases
        progress.current_size = 0
        self.assertEqual(progress.progress_percent, 0.0)
        
        progress.current_size = 1000
        self.assertEqual(progress.progress_percent, 100.0)
        
        progress.current_size = 1200  # Over 100%
        self.assertEqual(progress.progress_percent, 100.0)


class TestSimpleProgressBar(unittest.TestCase):
    """Test simple progress bar implementation"""
    
    def test_progress_bar_creation(self):
        """Test creating progress bars"""
        bar = SimpleProgressBar(total=1000, desc="Test progress")
        self.assertEqual(bar.total, 1000)
        self.assertEqual(bar.desc, "Test progress")
        self.assertEqual(bar.current, 0)
    
    def test_progress_updates(self):
        """Test updating progress"""
        bar = SimpleProgressBar(total=100, desc="Test")
        
        # Test increment updates
        bar.update(10)
        self.assertEqual(bar.current, 10)
        
        bar.update(20)
        self.assertEqual(bar.current, 30)
        
        # Test set current
        bar.set_current(50)
        self.assertEqual(bar.current, 50)
        
        # Test overflow protection
        bar.set_current(150)
        self.assertEqual(bar.current, 100)
    
    def test_byte_formatting(self):
        """Test byte formatting"""
        bar = SimpleProgressBar(total=1000)
        
        self.assertEqual(bar._format_bytes(1024), "1.0KB")
        self.assertEqual(bar._format_bytes(1024*1024), "1.0MB")
        self.assertEqual(bar._format_bytes(1024*1024*1024), "1.0GB")
        self.assertEqual(bar._format_bytes(500), "500.0B")
    
    def test_time_formatting(self):
        """Test time formatting"""
        bar = SimpleProgressBar(total=1000)
        
        self.assertEqual(bar._format_time(30), "30s")
        self.assertEqual(bar._format_time(90), "2m30s")
        self.assertEqual(bar._format_time(3665), "1h1m")


class TestProgressTracker(unittest.TestCase):
    """Test ProgressTracker functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.tracker = ProgressTracker(use_simple_bars=True)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any operations
        for op_id in list(self.tracker.operations.keys()):
            self.tracker.complete_operation(op_id)
    
    def test_start_operation(self):
        """Test starting progress operations"""
        success = self.tracker.start_operation(
            operation_id="test_op",
            operation_type=ProgressType.DOWNLOAD,
            description="Test download",
            total_size=1000
        )
        
        self.assertTrue(success)
        self.assertIn("test_op", self.tracker.operations)
        self.assertIn("test_op", self.tracker.progress_bars)
        
        # Test duplicate operation
        success = self.tracker.start_operation(
            operation_id="test_op",
            operation_type=ProgressType.UPLOAD,
            description="Duplicate",
            total_size=500
        )
        self.assertFalse(success)
    
    def test_update_progress(self):
        """Test updating progress"""
        self.tracker.start_operation(
            "test_op", ProgressType.DOWNLOAD, "Test", 1000
        )
        
        success = self.tracker.update_progress("test_op", 250, 100.0)
        self.assertTrue(success)
        
        operation = self.tracker.get_operation_info("test_op")
        self.assertEqual(operation.current_size, 250)
        self.assertEqual(operation.speed_bytes_per_sec, 100.0)
        self.assertIsNotNone(operation.eta_seconds)
        
        # Test invalid operation
        success = self.tracker.update_progress("invalid_op", 100)
        self.assertFalse(success)
    
    def test_complete_operation(self):
        """Test completing operations"""
        self.tracker.start_operation(
            "test_op", ProgressType.DOWNLOAD, "Test", 1000
        )
        
        self.tracker.complete_operation("test_op", success=True)
        
        operation = self.tracker.get_operation_info("test_op")
        self.assertEqual(operation.status, "completed")
        self.assertNotIn("test_op", self.tracker.progress_bars)
    
    def test_fail_operation(self):
        """Test failing operations"""
        self.tracker.start_operation(
            "test_op", ProgressType.UPLOAD, "Test", 1000
        )
        
        self.tracker.fail_operation("test_op", "Test error")
        
        operation = self.tracker.get_operation_info("test_op")
        self.assertEqual(operation.status, "failed")
    
    def test_pause_resume_operation(self):
        """Test pausing and resuming operations"""
        self.tracker.start_operation(
            "test_op", ProgressType.GRADIENT_UPDATE, "Test", 100
        )
        
        self.tracker.pause_operation("test_op")
        operation = self.tracker.get_operation_info("test_op")
        self.assertEqual(operation.status, "paused")
        
        self.tracker.resume_operation("test_op")
        operation = self.tracker.get_operation_info("test_op")
        self.assertEqual(operation.status, "active")
    
    def test_cleanup_completed(self):
        """Test cleaning up completed operations"""
        # Create some operations
        self.tracker.start_operation("op1", ProgressType.DOWNLOAD, "Test 1", 100)
        self.tracker.start_operation("op2", ProgressType.UPLOAD, "Test 2", 200)
        
        # Complete them
        self.tracker.complete_operation("op1")
        self.tracker.complete_operation("op2")
        
        # They should still be there initially
        self.assertEqual(len(self.tracker.operations), 2)
        
        # Clean up with 0 age (should remove immediately)
        self.tracker.cleanup_completed(max_age_seconds=0)
        self.assertEqual(len(self.tracker.operations), 0)
    
    def test_get_all_operations(self):
        """Test getting all operations"""
        self.tracker.start_operation("op1", ProgressType.DOWNLOAD, "Test 1", 100)
        self.tracker.start_operation("op2", ProgressType.UPLOAD, "Test 2", 200)
        
        all_ops = self.tracker.get_all_operations()
        self.assertEqual(len(all_ops), 2)
        self.assertIn("op1", all_ops)
        self.assertIn("op2", all_ops)


class TestProgressTrackerConvenience(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test environment"""
        # Reset global tracker
        import dessin.runtime.progress_tracker
        dessin.runtime.progress_tracker._global_tracker = None
    
    def test_get_global_tracker(self):
        """Test getting global tracker instance"""
        tracker1 = get_progress_tracker()
        tracker2 = get_progress_tracker()
        
        # Should be the same instance
        self.assertIs(tracker1, tracker2)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test track_download
        op_id = track_download("test_dl", "Test download", 1000)
        self.assertEqual(op_id, "test_dl")
        
        tracker = get_progress_tracker()
        self.assertIn("test_dl", tracker.operations)
        
        # Test update_progress
        update_progress("test_dl", 500, 100.0)
        operation = tracker.get_operation_info("test_dl")
        self.assertEqual(operation.current_size, 500)
        
        # Test complete_progress
        complete_progress("test_dl", success=True)
        operation = tracker.get_operation_info("test_dl")
        self.assertEqual(operation.status, "completed")


class TestProgressTrackerThreadSafety(unittest.TestCase):
    """Test thread safety of progress tracker"""
    
    def test_concurrent_operations(self):
        """Test concurrent progress operations"""
        tracker = ProgressTracker(use_simple_bars=True)
        
        def worker(worker_id):
            op_id = f"worker_{worker_id}"
            tracker.start_operation(
                op_id, ProgressType.DOWNLOAD, f"Worker {worker_id}", 1000
            )
            
            for i in range(10):
                tracker.update_progress(op_id, i * 100)
                time.sleep(0.01)
            
            tracker.complete_operation(op_id)
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should be completed
        completed_ops = [op for op in tracker.operations.values() 
                        if op.status == "completed"]
        self.assertEqual(len(completed_ops), 5)


if __name__ == "__main__":
    unittest.main()
