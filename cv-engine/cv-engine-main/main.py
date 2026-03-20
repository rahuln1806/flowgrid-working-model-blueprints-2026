#!/usr/bin/env python3
"""
Main CV Engine Application
Integrates traffic detection with Redis for complete traffic analysis system
"""

import cv2
import time
import argparse
import signal
import sys
from typing import Optional
from pathlib import Path

from inference.detect_enhanced import TrafficDetector
from integration.redis_enhanced import RedisIntegration
from utils.performance_utils import PerformanceOptimizer
from utils.visualization_utils import VisualizationHelper

class TrafficAnalysisEngine:
    """Main traffic analysis engine combining detection and integration"""
    
    def __init__(
        self,
        model_path: str = "runs/detect/traffic_train/weights/best.pt",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        confidence_threshold: float = 0.5,
        target_fps: int = 30,
        enable_redis: bool = True,
        enable_visualization: bool = True
    ):
        """
        Initialize traffic analysis engine
        
        Args:
            model_path: Path to trained model
            redis_host: Redis server host
            redis_port: Redis server port
            confidence_threshold: Detection confidence threshold
            target_fps: Target processing FPS
            enable_redis: Whether to enable Redis integration
            enable_visualization: Whether to show visualization
        """
        self.enable_redis = enable_redis
        self.enable_visualization = enable_visualization
        self.running = False
        
        print("🚦 Initializing Traffic Analysis Engine...")
        
        # Initialize traffic detector
        self.detector = TrafficDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(target_fps=target_fps)
        
        # Initialize Redis integration
        self.redis_client = None
        if enable_redis:
            try:
                self.redis_client = RedisIntegration(
                    host=redis_host,
                    port=redis_port
                )
                if not self.redis_client.is_connected():
                    print("⚠️ Redis connection failed, continuing without Redis")
                    self.enable_redis = False
            except Exception as e:
                print(f"⚠️ Redis initialization failed: {e}")
                self.enable_redis = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("✅ Traffic Analysis Engine initialized successfully!")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _process_detection_data(self, detection_data: dict, frame_count: int) -> dict:
        """
        Process and enhance detection data
        
        Args:
            detection_data: Raw detection data
            frame_count: Current frame count
            
        Returns:
            Enhanced detection data
        """
        # Add metadata
        enhanced_data = detection_data.copy()
        enhanced_data.update({
            "frame_count": frame_count,
            "timestamp": int(time.time()),
            "fps": self.optimizer.performance_stats.get('avg_fps', 0)
        })
        
        # Send to Redis if enabled
        if self.enable_redis and self.redis_client:
            self.redis_client.send_detection_data(enhanced_data)
            
            # Send ambulance alert if detected
            if detection_data.get("ambulance", False):
                self.redis_client.send_alert(
                    "ambulance",
                    f"Ambulance detected in frame {frame_count}",
                    "critical"
                )
        
        return enhanced_data
    
    def _display_frame(self, frame: np.ndarray, detection_data: dict, fps: float):
        """
        Display frame with detection information
        
        Args:
            frame: Input frame
            detection_data: Detection results
            fps: Current FPS
        """
        if not self.enable_visualization:
            return
        
        # Draw statistics
        VisualizationHelper.draw_statistics(frame, detection_data, (10, 30))
        
        # Draw FPS
        VisualizationHelper.draw_fps(frame, fps, (10, frame.shape[0] - 30))
        
        # Show frame
        cv2.imshow('Traffic Analysis Engine', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f"traffic_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved: {filename}")
    
    def run_webcam(self, camera_id: int = 0, save_output: bool = False, output_path: Optional[str] = None):
        """
        Run traffic analysis on webcam feed
        
        Args:
            camera_id: Webcam camera ID
            save_output: Whether to save output video
            output_path: Output video path
        """
        print(f"🎥 Starting webcam analysis (Camera {camera_id})")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Failed to open webcam {camera_id}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📐 Webcam resolution: {width}x{height}")
        print(f"⏱️ Webcam FPS: {fps}")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            if output_path is None:
                output_path = f"traffic_analysis_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        # Setup system status
        if self.enable_redis and self.redis_client:
            self.redis_client.set_system_status("active", {
                "source": "webcam",
                "camera_id": camera_id,
                "resolution": f"{width}x{height}",
                "target_fps": self.optimizer.target_fps
            })
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from webcam")
                    break
                
                current_time = time.time()
                
                # Limit FPS if needed
                if not self.optimizer.limit_fps(current_time):
                    continue
                
                # Run detection
                annotated_frame, detection_data = self.detector.detect_frame(frame)
                
                # Process detection data
                enhanced_data = self._process_detection_data(detection_data, frame_count)
                
                # Calculate FPS
                processing_time = time.time() - current_time
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Update performance stats
                self.optimizer.update_stats(processing_time)
                
                # Display frame
                self._display_frame(annotated_frame, enhanced_data, current_fps)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"📊 Processed {frame_count} frames | Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Analysis stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if self.enable_visualization:
                cv2.destroyAllWindows()
            
            # Update system status
            if self.enable_redis and self.redis_client:
                self.redis_client.set_system_status("inactive")
            
            # Print final statistics
            total_time = time.time() - start_time
            final_stats = self.optimizer.get_stats()
            print(f"\n📊 Final Statistics:")
            print(f"   Total frames: {frame_count}")
            print(f"   Processing time: {total_time:.2f}s")
            print(f"   Average FPS: {final_stats['avg_fps']:.1f}")
            print(f"   Min/Max FPS: {final_stats['min_fps']:.1f}/{final_stats['max_fps']:.1f}")
    
    def run_video(self, video_path: str, save_output: bool = False, output_path: Optional[str] = None):
        """
        Run traffic analysis on video file
        
        Args:
            video_path: Path to video file
            save_output: Whether to save output video
            output_path: Output video path
        """
        print(f"🎬 Starting video analysis: {video_path}")
        
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📐 Video resolution: {width}x{height}")
        print(f"⏱️ Video FPS: {fps}")
        print(f"🎞️ Total frames: {total_frames}")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            if output_path is None:
                output_path = f"traffic_analysis_{Path(video_path).stem}_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        # Setup system status
        if self.enable_redis and self.redis_client:
            self.redis_client.set_system_status("active", {
                "source": "video_file",
                "video_path": video_path,
                "resolution": f"{width}x{height}",
                "total_frames": total_frames
            })
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Run detection
                annotated_frame, detection_data = self.detector.detect_frame(frame)
                
                # Process detection data
                enhanced_data = self._process_detection_data(detection_data, frame_count)
                
                # Calculate FPS
                processing_time = time.time() - current_time
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Update performance stats
                self.optimizer.update_stats(processing_time)
                
                # Display frame
                self._display_frame(annotated_frame, enhanced_data, current_fps)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"📊 Progress: {progress:.1f}% | Frames: {frame_count}/{total_frames} | Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Analysis stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if self.enable_visualization:
                cv2.destroyAllWindows()
            
            # Update system status
            if self.enable_redis and self.redis_client:
                self.redis_client.set_system_status("inactive")
            
            # Print final statistics
            total_time = time.time() - start_time
            final_stats = self.optimizer.get_stats()
            print(f"\n📊 Final Statistics:")
            print(f"   Processed frames: {frame_count}/{total_frames}")
            print(f"   Processing time: {total_time:.2f}s")
            print(f"   Average FPS: {final_stats['avg_fps']:.1f}")
            print(f"   Min/Max FPS: {final_stats['min_fps']:.1f}/{final_stats['max_fps']:.1f}")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Traffic Analysis Engine")
    
    # Input source
    parser.add_argument(
        "--source", 
        type=str, 
        default="0",
        help="Video file path or webcam index (default: 0)"
    )
    
    # Model settings
    parser.add_argument(
        "--model", 
        type=str, 
        default="runs/detect/traffic_train/weights/best.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.5,
        help="Confidence threshold"
    )
    
    # Performance settings
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30,
        help="Target processing FPS"
    )
    
    # Redis settings
    parser.add_argument(
        "--redis-host", 
        type=str, 
        default="localhost",
        help="Redis server host"
    )
    parser.add_argument(
        "--redis-port", 
        type=int, 
        default=6379,
        help="Redis server port"
    )
    parser.add_argument(
        "--no-redis", 
        action="store_true",
        help="Disable Redis integration"
    )
    
    # Output settings
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output video path"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Disable visual display"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize engine
        engine = TrafficAnalysisEngine(
            model_path=args.model,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            confidence_threshold=args.conf,
            target_fps=args.fps,
            enable_redis=not args.no_redis,
            enable_visualization=not args.no_display
        )
        
        # Run analysis
        if args.source.isdigit():
            # Webcam
            engine.run_webcam(
                camera_id=int(args.source),
                save_output=args.save,
                output_path=args.output
            )
        else:
            # Video file
            engine.run_video(
                video_path=args.source,
                save_output=args.save,
                output_path=args.output
            )
        
        return 0
        
    except Exception as e:
        print(f"❌ Traffic analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
