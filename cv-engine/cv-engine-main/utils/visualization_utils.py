"""
Visualization utilities for traffic analysis
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import colorsys

class VisualizationHelper:
    """Helper class for visualization and drawing functions"""
    
    # Color palette for different classes
    COLORS = {
        'pedestrian': (0, 255, 0),      # Green
        'vehicle': (255, 0, 0),          # Blue
        'ambulance': (0, 0, 255),       # Red
        'default': (128, 128, 128)      # Gray
    }
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    FONT_COLOR = (255, 255, 255)  # White
    
    @staticmethod
    def draw_bbox(frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int], 
                  thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Frame with bounding box drawn
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], font_scale: float = None,
                   thickness: int = None) -> np.ndarray:
        """
        Draw text label on frame
        
        Args:
            frame: Input frame
            text: Text to draw
            position: Position (x, y)
            color: BGR color tuple
            font_scale: Font scale (uses default if None)
            thickness: Font thickness (uses default if None)
            
        Returns:
            Frame with label drawn
        """
        if font_scale is None:
            font_scale = VisualizationHelper.FONT_SCALE
        if thickness is None:
            thickness = VisualizationHelper.FONT_THICKNESS
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, VisualizationHelper.FONT, font_scale, thickness
        )
        
        # Background rectangle
        cv2.rectangle(
            frame,
            (position[0], position[1] - text_height - baseline - 5),
            (position[0] + text_width, position[1] + baseline),
            color,
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            text,
            position,
            VisualizationHelper.FONT,
            font_scale,
            VisualizationHelper.FONT_COLOR,
            thickness
        )
        
        return frame
    
    @staticmethod
    def draw_detection(frame: np.ndarray, detection: Dict, 
                      show_confidence: bool = True, 
                      highlight_ambulance: bool = True) -> np.ndarray:
        """
        Draw a single detection on frame
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            show_confidence: Whether to show confidence score
            highlight_ambulance: Whether to highlight ambulance detections
            
        Returns:
            Frame with detection drawn
        """
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get color for class
        color = VisualizationHelper.COLORS.get(class_name, VisualizationHelper.COLORS['default'])
        
        # Highlight ambulance with thicker border
        thickness = 4 if (highlight_ambulance and class_name == 'ambulance') else 2
        
        # Draw bounding box
        VisualizationHelper.draw_bbox(frame, bbox, color, thickness)
        
        # Prepare label text
        label = class_name
        if show_confidence:
            label += f": {confidence:.2f}"
        
        # Draw label
        label_position = (bbox[0], bbox[1] - 5)
        VisualizationHelper.draw_label(frame, label, label_position, color)
        
        # Special ambulance alert
        if highlight_ambulance and class_name == 'ambulance':
            VisualizationHelper.draw_ambulance_alert(frame)
        
        return frame
    
    @staticmethod
    def draw_ambulance_alert(frame: np.ndarray, position: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """
        Draw ambulance alert on frame
        
        Args:
            frame: Input frame
            position: Alert position (x, y)
            
        Returns:
            Frame with ambulance alert
        """
        alert_text = "🚑 AMBULANCE DETECTED!"
        
        # Draw alert background
        (text_width, text_height), _ = cv2.getTextSize(
            alert_text, VisualizationHelper.FONT, 1.0, 3
        )
        
        # Pulsing red background
        cv2.rectangle(
            frame,
            (position[0] - 10, position[1] - text_height - 10),
            (position[0] + text_width + 10, position[1] + 10),
            (0, 0, 255),
            -1
        )
        
        # Alert text
        cv2.putText(
            frame,
            alert_text,
            position,
            VisualizationHelper.FONT,
            1.0,
            (255, 255, 255),
            3
        )
        
        return frame
    
    @staticmethod
    def draw_statistics(frame: np.ndarray, stats: Dict, 
                       position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw detection statistics on frame
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            position: Statistics position (x, y)
            
        Returns:
            Frame with statistics drawn
        """
        # Prepare statistics text
        lines = [
            f"Vehicles: {stats.get('vehicles', 0)}",
            f"Pedestrians: {stats.get('pedestrians', 0)}",
        ]
        
        if stats.get('ambulance', False):
            lines.append("🚑 AMBULANCE: YES")
        
        # Draw background for statistics
        line_height = 25
        max_width = max(len(line) for line in lines) * 10
        
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - line_height + 5),
            (position[0] + max_width, position[1] + len(lines) * line_height),
            (0, 0, 0),
            -1
        )
        
        # Draw each line
        for i, line in enumerate(lines):
            y_pos = position[1] + i * line_height
            cv2.putText(
                frame,
                line,
                (position[0], y_pos),
                VisualizationHelper.FONT,
                VisualizationHelper.FONT_SCALE,
                VisualizationHelper.FONT_COLOR,
                VisualizationHelper.FONT_THICKNESS
            )
        
        return frame
    
    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float, 
                 position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input frame
            fps: Current FPS
            position: FPS position (x, y)
            
        Returns:
            Frame with FPS drawn
        """
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw background
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, VisualizationHelper.FONT, VisualizationHelper.FONT_SCALE, 
            VisualizationHelper.FONT_THICKNESS
        )
        
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            position,
            VisualizationHelper.FONT,
            VisualizationHelper.FONT_SCALE,
            (0, 255, 0),  # Green color for FPS
            VisualizationHelper.FONT_THICKNESS
        )
        
        return frame
    
    @staticmethod
    def draw_grid(frame: np.ndarray, grid_size: int = 50, 
                  color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
        """
        Draw grid on frame for reference
        
        Args:
            frame: Input frame
            grid_size: Grid cell size in pixels
            color: Grid color (BGR)
            
        Returns:
            Frame with grid drawn
        """
        height, width = frame.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), color, 1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), color, 1)
        
        return frame
    
    @staticmethod
    def create_heatmap(frame_shape: Tuple[int, int], detections: List[Dict], 
                      grid_size: int = 50) -> np.ndarray:
        """
        Create heatmap from detection positions
        
        Args:
            frame_shape: Shape of original frame (height, width)
            detections: List of detection dictionaries
            grid_size: Grid cell size for heatmap
            
        Returns:
            Heatmap as numpy array
        """
        height, width = frame_shape
        
        # Create grid
        grid_h = height // grid_size
        grid_w = width // grid_size
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Accumulate detection positions
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Convert to grid coordinates
            grid_x = min(center_x // grid_size, grid_w - 1)
            grid_y = min(center_y // grid_size, grid_h - 1)
            
            heatmap[grid_y, grid_x] += 1
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Resize to original frame size
        heatmap_colored = cv2.resize(heatmap_colored, (width, height))
        
        return heatmap_colored
    
    @staticmethod
    def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on frame
        
        Args:
            frame: Original frame
            heatmap: Heatmap to overlay
            alpha: Transparency factor (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    
    @staticmethod
    def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                       color: Tuple[int, int, int] = (0, 255, 255),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw object trajectory on frame
        
        Args:
            frame: Input frame
            trajectory: List of (x, y) positions
            color: Trajectory color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with trajectory drawn
        """
        if len(trajectory) < 2:
            return frame
        
        # Draw trajectory lines
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)
        
        # Draw points
        for point in trajectory:
            cv2.circle(frame, point, 3, color, -1)
        
        return frame
    
    @staticmethod
    def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors for visualization
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.8
            value = 0.9
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    @staticmethod
    def create_info_panel(frame: np.ndarray, info: Dict, 
                         position: Tuple[int, int] = (10, 10),
                         panel_width: int = 300) -> np.ndarray:
        """
        Create information panel on frame
        
        Args:
            frame: Input frame
            info: Information dictionary
            position: Panel position (x, y)
            panel_width: Panel width in pixels
            
        Returns:
            Frame with info panel
        """
        # Prepare info lines
        lines = []
        for key, value in info.items():
            lines.append(f"{key}: {value}")
        
        # Calculate panel dimensions
        line_height = 25
        panel_height = len(lines) * line_height + 20
        
        # Draw panel background
        x, y = position
        cv2.rectangle(
            frame,
            (x, y),
            (x + panel_width, y + panel_height),
            (0, 0, 0),
            -1
        )
        
        # Draw panel border
        cv2.rectangle(
            frame,
            (x, y),
            (x + panel_width, y + panel_height),
            (255, 255, 255),
            2
        )
        
        # Draw info text
        for i, line in enumerate(lines):
            text_y = y + 20 + i * line_height
            cv2.putText(
                frame,
                line,
                (x + 10, text_y),
                VisualizationHelper.FONT,
                VisualizationHelper.FONT_SCALE,
                VisualizationHelper.FONT_COLOR,
                VisualizationHelper.FONT_THICKNESS
            )
        
        return frame
