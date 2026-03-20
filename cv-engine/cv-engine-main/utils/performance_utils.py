"""
Performance optimization utilities for traffic analysis
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional, Callable
import threading
import queue

class PerformanceOptimizer:
    """Performance optimization utilities for real-time processing"""
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize performance optimizer
        
        Args:
            target_fps: Target frames per second for processing
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0
        self.frame_times = []
        self.performance_stats = {
            'avg_fps': 0,
            'min_fps': float('inf'),
            'max_fps': 0,
            'total_frames': 0,
            'processing_time': 0
        }
    
    def limit_fps(self, current_time: float) -> bool:
        """
        Limit processing to target FPS
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if should process this frame, False otherwise
        """
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False
    
    def update_stats(self, processing_time: float) -> None:
        """
        Update performance statistics
        
        Args:
            processing_time: Time taken to process current frame
        """
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 100:  # Keep only last 100 frames
            self.frame_times.pop(0)
        
        # Update statistics
        self.performance_stats['total_frames'] += 1
        self.performance_stats['processing_time'] += processing_time
        self.performance_stats['avg_fps'] = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        self.performance_stats['min_fps'] = min(self.performance_stats['min_fps'], current_fps)
        self.performance_stats['max_fps'] = max(self.performance_stats['max_fps'], current_fps)
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.frame_times.clear()
        self.performance_stats = {
            'avg_fps': 0,
            'min_fps': float('inf'),
            'max_fps': 0,
            'total_frames': 0,
            'processing_time': 0
        }
    
    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size: Tuple[int, int], 
                              maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize frame with optimized performance
        
        Args:
            frame: Input frame
            target_size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if maintain_aspect_ratio:
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create canvas with target size and place resized frame
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
        else:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def apply_gaussian_blur_optimized(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction
        
        Args:
            frame: Input frame
            kernel_size: Kernel size for blur
            
        Returns:
            Blurred frame
        """
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def convert_color_optimized(frame: np.ndarray, conversion: int = cv2.COLOR_BGR2GRAY) -> np.ndarray:
        """
        Convert color space with optimization
        
        Args:
            frame: Input frame
            conversion: OpenCV color conversion code
            
        Returns:
            Converted frame
        """
        return cv2.cvtColor(frame, conversion)
    
    @staticmethod
    def create_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
        """
        Create background subtractor for motion detection
        
        Returns:
            Background subtractor object
        """
        return cv2.createBackgroundSubtractorMOG2(
            history=500,        # History length
            varThreshold=16,    # Variance threshold
            detectShadows=True  # Detect shadows
        )

class FrameBuffer:
    """Thread-safe frame buffer for multi-threaded processing"""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize frame buffer
        
        Args:
            max_size: Maximum number of frames to buffer
        """
        self.buffer = queue.Queue(maxsize=max_size)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def put_frame(self, frame: np.ndarray) -> None:
        """
        Add frame to buffer
        
        Args:
            frame: Frame to add
        """
        try:
            # Try to add to queue (non-blocking)
            self.buffer.put_nowait(frame)
        except queue.Full:
            # If buffer is full, remove oldest frame
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame)
            except queue.Empty:
                pass
        
        # Update latest frame
        with self.lock:
            self.latest_frame = frame.copy()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get frame from buffer
        
        Returns:
            Frame if available, None otherwise
        """
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            # Return latest frame if buffer is empty
            with self.lock:
                return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def clear(self) -> None:
        """Clear the buffer"""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        
        with self.lock:
            self.latest_frame = None

class AsyncProcessor:
    """Asynchronous frame processor for improved performance"""
    
    def __init__(self, processing_func: Callable, num_workers: int = 2):
        """
        Initialize async processor
        
        Args:
            processing_func: Function to process frames
            num_workers: Number of worker threads
        """
        self.processing_func = processing_func
        self.num_workers = num_workers
        self.input_queue = queue.Queue(maxsize=num_workers * 2)
        self.output_queue = queue.Queue(maxsize=num_workers * 2)
        self.workers = []
        self.running = False
    
    def start(self) -> None:
        """Start worker threads"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_func, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self) -> None:
        """Stop worker threads"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers.clear()
    
    def _worker_func(self, worker_id: int) -> None:
        """
        Worker function for processing frames
        
        Args:
            worker_id: ID of the worker thread
        """
        while self.running:
            try:
                # Get frame from input queue
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                # Process frame
                frame, frame_id = frame_data
                result = self.processing_func(frame)
                
                # Put result in output queue
                self.output_queue.put((result, frame_id))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """
        Submit frame for processing
        
        Args:
            frame: Frame to process
            frame_id: Unique frame identifier
        """
        try:
            self.input_queue.put_nowait((frame, frame_id))
        except queue.Full:
            # Drop frame if queue is full
            pass
    
    def get_result(self) -> Optional[Tuple]:
        """
        Get processing result
        
        Returns:
            Tuple of (result, frame_id) if available, None otherwise
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def clear_queues(self) -> None:
        """Clear input and output queues"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

def benchmark_processing(func: Callable, test_data: list, iterations: int = 10) -> dict:
    """
    Benchmark processing function performance
    
    Args:
        func: Function to benchmark
        test_data: Test data to process
        iterations: Number of iterations to run
        
    Returns:
        Benchmark statistics
    """
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        for data in test_data:
            func(data)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_dev': np.std(times),
        'iterations': iterations,
        'data_size': len(test_data)
    }
