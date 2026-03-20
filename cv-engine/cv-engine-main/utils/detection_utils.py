"""
Detection utility functions for traffic analysis
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from ultralytics import YOLO

class DetectionUtils:
    """Utility class for detection processing and filtering"""
    
    @staticmethod
    def filter_by_confidence(detections: List[Dict], threshold: float) -> List[Dict]:
        """
        Filter detections by confidence threshold
        
        Args:
            detections: List of detection dictionaries
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det.get('confidence', 0) >= threshold]
    
    @staticmethod
    def filter_by_class(detections: List[Dict], class_names: List[str]) -> List[Dict]:
        """
        Filter detections by class names
        
        Args:
            detections: List of detection dictionaries
            class_names: List of class names to keep
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det.get('class_name') in class_names]
    
    @staticmethod
    def parse_bounding_box(bbox: List[int]) -> Tuple[int, int, int, int]:
        """
        Parse bounding box coordinates
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (x1, y1, x2, y2)
        """
        if len(bbox) != 4:
            raise ValueError("Bounding box must have 4 coordinates: [x1, y1, x2, y2]")
        return tuple(bbox)
    
    @staticmethod
    def calculate_bbox_area(bbox: List[int]) -> int:
        """
        Calculate area of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Area of the bounding box
        """
        x1, y1, x2, y2 = DetectionUtils.parse_bounding_box(bbox)
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    @staticmethod
    def calculate_bbox_center(bbox: List[int]) -> Tuple[int, int]:
        """
        Calculate center point of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Center point (x, y)
        """
        x1, y1, x2, y2 = DetectionUtils.parse_bounding_box(bbox)
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = DetectionUtils.parse_bounding_box(bbox1)
        x1_2, y1_2, x2_2, y2_2 = DetectionUtils.parse_bounding_box(bbox2)
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        
        # Calculate union
        area1 = DetectionUtils.calculate_bbox_area(bbox1)
        area2 = DetectionUtils.calculate_bbox_area(bbox2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    @staticmethod
    def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort detections by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        suppressed = []
        while sorted_detections:
            # Keep the detection with highest confidence
            current = sorted_detections.pop(0)
            suppressed.append(current)
            
            # Remove detections that overlap significantly with current
            remaining = []
            for det in sorted_detections:
                if (det.get('class_name') == current.get('class_name') and
                    DetectionUtils.calculate_iou(det['bbox'], current['bbox']) > iou_threshold):
                    # Skip this detection (suppressed)
                    continue
                remaining.append(det)
            
            sorted_detections = remaining
        
        return suppressed
    
    @staticmethod
    def format_detection_output(detections: List[Dict]) -> Dict[str, Any]:
        """
        Format detection results into structured JSON output
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Formatted detection data
        """
        counts = {"pedestrian": 0, "vehicle": 0, "ambulance": 0}
        ambulance_present = False
        
        for det in detections:
            class_name = det.get('class_name', 'unknown')
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
    
    @staticmethod
    def map_coco_to_traffic_classes(coco_class_id: int) -> int:
        """
        Map COCO class IDs to traffic analysis classes
        
        Args:
            coco_class_id: COCO class ID
            
        Returns:
            Traffic class ID (0=pedestrian, 1=vehicle, 2=ambulance)
        """
        # COCO classes relevant to traffic
        # 0: person -> pedestrian
        # 2: car, 3: motorcycle, 5: bus, 7: truck -> vehicle
        if coco_class_id == 0:
            return 0  # pedestrian
        elif coco_class_id in [2, 3, 5, 7]:
            return 1  # vehicle
        else:
            return -1  # not relevant
    
    @staticmethod
    def validate_detection_format(detection: Dict) -> bool:
        """
        Validate detection dictionary format
        
        Args:
            detection: Detection dictionary
            
        Returns:
            True if format is valid, False otherwise
        """
        required_keys = ['class_id', 'class_name', 'confidence', 'bbox']
        
        for key in required_keys:
            if key not in detection:
                return False
        
        # Validate bbox format
        bbox = detection['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        
        # Validate confidence
        confidence = detection['confidence']
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        return True
    
    @staticmethod
    def map_dataset_to_traffic_classes(dataset_class_id: int, class_mapping: dict) -> int:
        """
        Map dataset class IDs to traffic analysis classes using class mapping
        
        Args:
            dataset_class_id: Dataset class ID
            class_mapping: Class mapping dictionary
            
        Returns:
            Traffic class ID (0=pedestrian, 1=vehicle, 2=ambulance)
        """
        return class_mapping.get(dataset_class_id, 1)  # Default to vehicle


    @staticmethod
    def save_detections_to_json(detections: List[Dict], filename: str) -> None:
        """
        Save detection results to JSON file
        
        Args:
            detections: List of detection dictionaries
            filename: Output filename
        """
        formatted_data = DetectionUtils.format_detection_output(detections)
        
        with open(filename, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"📄 Detection results saved to: {filename}")
    
    @staticmethod
    def load_detections_from_json(filename: str) -> Dict[str, Any]:
        """
        Load detection results from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Detection data dictionary
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data
