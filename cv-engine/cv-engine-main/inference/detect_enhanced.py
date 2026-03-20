#!/usr/bin/env python3
"""
YOLOv8 Inference Engine for Traffic Analysis
Real-time detection of pedestrians, vehicles, and ambulances
"""

import cv2
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import numpy as np

class TrafficDetector:
    """Traffic detection class using YOLOv8"""
    
    def __init__(
        self,
        model_path: str = "runs/detect/traffic_train/weights/best.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_size: Tuple[int, int] = (640, 480),
        frame_skip: int = 1
    ):
        """
        Initialize traffic detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            target_size: Target frame size for processing
            frame_skip: Process every nth frame for performance
        """
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.frame_count = 0
        
        # Class names and colors
        self.class_names = {0: "pedestrian", 1: "vehicle", 2: "ambulance"}
        self.class_colors = {
            0: (0, 255, 0),    # Green for pedestrian
            1: (255, 0, 0),    # Blue for vehicle
            2: (0, 0, 255)     # Red for ambulance
        }
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Fallback to pretrained model for demo
            print("🔄 Loading pretrained YOLOv8n model as fallback...")
            self.model = YOLO("yolov8n.pt")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size for faster processing"""
        return cv2.resize(frame, self.target_size)
    
    def _filter_classes(self, results) -> List[Dict]:
        """
        Filter detections to only include our target classes
        Maps COCO classes to our traffic classes
        """
        filtered_detections = []
        
        if not results or len(results) == 0:
            return filtered_detections
            
        result = results[0]
        if result.boxes is None:
            return filtered_detections
        
        for box in result.boxes:
            # Get class ID and confidence
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            # Filter by confidence
            if conf < self.conf_threshold:
                continue
            
            # Map COCO classes to our traffic classes
            # COCO: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, 11=stop sign, etc.
            if cls_id == 0:  # person -> pedestrian
                traffic_cls = 0
            elif cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck -> vehicle
                traffic_cls = 1
            else:
                # Skip other classes
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class_id': traffic_cls,
                'class_name': self.class_names[traffic_cls],
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            }
            filtered_detections.append(detection)
        
        return filtered_detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for class
            color = self.class_colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
            
            # Special highlight for ambulance
            if class_id == 2:  # ambulance
                cv2.putText(
                    frame, 
                    "🚑 AMBULANCE DETECTED!", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    3
                )
        
        return frame
    
    def _process_detections(self, detections: List[Dict]) -> Dict:
        """Process detections to get counts and ambulance presence"""
        counts = {"pedestrian": 0, "vehicle": 0, "ambulance": 0}
        ambulance_present = False
        
        for det in detections:
            class_name = det['class_name']
            counts[class_name] += 1
            
            if class_name == "ambulance":
                ambulance_present = True
        
        return {
            "vehicles": counts["vehicle"],
            "pedestrians": counts["pedestrian"],
            "ambulance": ambulance_present,
            "total_objects": sum(counts.values()),
            "detections": detections
        }
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detection_data)
        """
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame, {"vehicles": 0, "pedestrians": 0, "ambulance": False}
        
        # Resize frame for faster processing
        processed_frame = self._resize_frame(frame)
        
        # Run inference
        results = self.model(processed_frame, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Filter and process detections
        detections = self._filter_classes(results)
        detection_data = self._process_detections(detections)
        
        # Scale bounding boxes back to original frame size
        if detections:
            h, w = frame.shape[:2]
            scale_x, scale_y = w / self.target_size[0], h / self.target_size[1]
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                det['bbox'] = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
        
        # Draw detections on original frame
        annotated_frame = self._draw_detections(frame.copy(), detections)
        
        return annotated_frame, detection_data
    
    def detect_video(self, source: str, save_output: bool = False, output_path: str = None):
        """
        Run detection on video file or webcam
        
        Args:
            source: Video file path or webcam index (0, 1, etc.)
            save_output: Whether to save output video
            output_path: Path to save output video
        """
        # Open video capture
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving output
        writer = None
        if save_output:
            if output_path is None:
                output_path = f"traffic_detection_output_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎥 Processing video: {source}")
        print(f"📐 Resolution: {width}x{height}")
        print(f"⏱️ FPS: {fps}")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                annotated_frame, detection_data = self.detect_frame(frame)
                
                # Display detection info on frame
                info_text = f"Vehicles: {detection_data['vehicles']} | Pedestrians: {detection_data['pedestrians']}"
                if detection_data['ambulance']:
                    info_text += " | 🚑 AMBULANCE"
                
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Show frame
                cv2.imshow('Traffic Detection', annotated_frame)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                frame_count += 1
                
                # Calculate and display FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    print(f"⚡ Processing FPS: {current_fps:.1f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"📸 Frame saved: {save_path}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
                print(f"💾 Output video saved: {output_path}")
            cv2.destroyAllWindows()
            
            # Print statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\n📊 Processing Statistics:")
            print(f"   Total frames: {frame_count}")
            print(f"   Processing time: {total_time:.2f}s")
            print(f"   Average FPS: {avg_fps:.1f}")

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description="Run traffic detection on video")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="0",
        help="Video file path or webcam index (default: 0)"
    )
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
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.45,
        help="IoU threshold for NMS"
    )
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
        "--frame-skip", 
        type=int, 
        default=1,
        help="Process every nth frame"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = TrafficDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            frame_skip=args.frame_skip
        )
        
        # Run detection
        detector.detect_video(
            source=args.source,
            save_output=args.save,
            output_path=args.output
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Detection failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
