#!/usr/bin/env python3
"""
YOLOv8 Inference Engine for Traffic Analysis with Roboflow Dataset
Real-time detection of pedestrians and vehicles using Roboflow-trained model
"""

import cv2
import json
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import numpy as np

class RoboflowTrafficDetector:
    """Traffic detection class using Roboflow-trained YOLOv8 model"""
    
    def __init__(
        self,
        model_path: str = "runs/detect/traffic_train/weights/best.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_size: Tuple[int, int] = (640, 480),
        frame_skip: int = 1,
        class_mapping_file: str = "configs/class_mapping.yaml"
    ):
        """
        Initialize traffic detector with Roboflow dataset
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            target_size: Target frame size for processing
            frame_skip: Process every nth frame for performance
            class_mapping_file: Path to class mapping configuration
        """
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.frame_count = 0
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping(class_mapping_file)
        
        # Traffic class names and colors
        self.traffic_class_names = {0: "pedestrian", 1: "vehicle", 2: "ambulance"}
        self.class_colors = {
            0: (0, 255, 0),    # Green for pedestrian
            1: (255, 0, 0),    # Blue for vehicle
            2: (0, 0, 255)     # Red for ambulance
        }
        
        # Load model
        self._load_model()
        
    def _load_class_mapping(self, mapping_file: str) -> Dict[int, int]:
        """Load class mapping from YAML file"""
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = yaml.safe_load(f)
            
            mapping = mapping_data.get('mapping', {})
            dataset_classes = mapping_data.get('dataset_classes', {})
            
            print(f"📋 Loaded class mapping:")
            print(f"   Dataset classes: {dataset_classes}")
            print(f"   Mapping: {mapping}")
            
            return mapping
            
        except Exception as e:
            print(f"⚠️ Could not load class mapping: {e}")
            print("🔄 Using default mapping (0->vehicle, 1->pedestrian)")
            return {0: 1, 1: 0}  # Default mapping
    
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
    
    def _map_dataset_to_traffic_classes(self, detections: List[Dict]) -> List[Dict]:
        """
        Map dataset class IDs to traffic analysis classes
        
        Args:
            detections: List of detection dictionaries with dataset class IDs
            
        Returns:
            List of detections with traffic class IDs
        """
        mapped_detections = []
        
        for det in detections:
            dataset_class_id = det.get('class_id', -1)
            
            # Map to traffic class using mapping
            traffic_class_id = self.class_mapping.get(dataset_class_id, 1)  # Default to vehicle
            
            # Create new detection with traffic class
            mapped_det = det.copy()
            mapped_det['class_id'] = traffic_class_id
            mapped_det['class_name'] = self.traffic_class_names.get(traffic_class_id, 'unknown')
            
            mapped_detections.append(mapped_det)
        
        return mapped_detections
    
    def _filter_classes(self, results) -> List[Dict]:
        """
        Filter detections and map to traffic classes
        
        Args:
            results: YOLOv8 detection results
            
        Returns:
            List of filtered detections with traffic classes
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
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Create detection with dataset class
            detection = {
                'dataset_class_id': cls_id,
                'class_id': cls_id,  # Will be mapped later
                'class_name': f'class_{cls_id}',  # Will be updated
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            }
            filtered_detections.append(detection)
        
        # Map to traffic classes
        mapped_detections = self._map_dataset_to_traffic_classes(filtered_detections)
        
        return mapped_detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for class
            color = self.class_colors.get(class_id, (128, 128, 128))
            
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
        
        return frame
    
    def _process_detections(self, detections: List[Dict]) -> Dict:
        """Process detections to get counts and ambulance presence"""
        counts = {"pedestrian": 0, "vehicle": 0, "ambulance": 0}
        ambulance_present = False
        
        for det in detections:
            class_name = det['class_name']
            if class_name in counts:
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
                cv2.imshow('Traffic Detection (Roboflow)', annotated_frame)
                
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
    parser = argparse.ArgumentParser(description="Run traffic detection with Roboflow dataset")
    
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
    parser.add_argument(
        "--mapping", 
        type=str,
        default="configs/class_mapping.yaml",
        help="Class mapping file path"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = RoboflowTrafficDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            frame_skip=args.frame_skip,
            class_mapping_file=args.mapping
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
