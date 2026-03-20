# 🚦 Computer Vision Engine for Traffic Analysis

A production-ready computer vision system using YOLOv8 for real-time traffic analysis, detecting vehicles, pedestrians, and ambulances with Redis integration.

## 🎯 Features

- **Real-time Detection**: Detects vehicles, pedestrians, and ambulances in real-time
- **YOLOv8 Powered**: Uses state-of-the-art YOLOv8 model for accurate detection
- **Performance Optimized**: Frame resizing, skipping, and multi-threading support
- **Redis Integration**: Real-time data sharing and alert system
- **Modular Design**: Clean, extensible architecture
- **Ambulance Alerts**: Special detection and alerting for emergency vehicles
- **Structured Output**: JSON format for easy integration

## 🏗️ Project Structure

```
cv-engine/
├── data/                   # Dataset directory
│   ├── images/            # Training/validation images
│   └── labels/            # Annotation files
├── configs/
│   └── data.yaml          # Dataset configuration
├── models/                # Trained models
├── training/
│   ├── train.py           # Basic training script
│   └── train_enhanced.py  # Enhanced training with CLI
├── inference/
│   ├── detect.py          # Basic detection script
│   └── detect_enhanced.py # Enhanced detection with features
├── integration/
│   ├── redis_sender.py    # Basic Redis client
│   └── redis_enhanced.py  # Enhanced Redis integration
├── utils/
│   ├── __init__.py
│   ├── detection_utils.py # Detection processing utilities
│   ├── performance_utils.py # Performance optimization
│   └── visualization_utils.py # Visualization helpers
├── requirements.txt       # Basic dependencies
├── requirements_enhanced.txt # Full dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd cv-engine

# Install dependencies
pip install -r requirements_enhanced.txt

# For basic functionality
pip install -r requirements.txt
```

### 2. Training (Optional)

If you have a custom dataset:

```bash
# Basic training
python training/train.py

# Enhanced training with options
python training/train_enhanced.py \
    --data configs/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --img-size 640 \
    --batch-size 16
```

### 3. Run Detection

```bash
# Basic detection (webcam)
python inference/detect.py

# Enhanced detection with options
python inference/detect_enhanced.py \
    --source 0 \
    --model runs/detect/traffic_train/weights/best.pt \
    --conf 0.5 \
    --save \
    --output traffic_output.mp4

# Detection on video file
python inference/detect_enhanced.py \
    --source video.mp4 \
    --conf 0.6 \
    --frame-skip 2
```

### 4. Redis Integration

```bash
# Test Redis connection
python integration/redis_enhanced.py --test

# Run with Redis integration in your code
from integration.redis_enhanced import RedisIntegration

redis_client = RedisIntegration()
redis_client.send_detection_data({
    "vehicles": 5,
    "pedestrians": 2,
    "ambulance": True
})
```

## 📊 Detection Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0        | pedestrian | People walking, standing |
| 1        | vehicle    | Cars, trucks, motorcycles, buses |
| 2        | ambulance  | Emergency vehicles (special detection) |

## 🔧 Configuration

### Dataset Configuration (`configs/data.yaml`)

```yaml
train: ../data/dataset/images/train
val: ../data/dataset/images/val
nc: 3
names: ["pedestrian", "vehicle", "ambulance"]
```

### Model Parameters

- **Model Size**: YOLOv8n (nano) for speed, can use YOLOv8s/m/l/x for accuracy
- **Input Size**: 640x640 (configurable)
- **Confidence Threshold**: 0.5 (default, adjustable)
- **IoU Threshold**: 0.45 (default)

## 📱 Usage Examples

### Basic Detection

```python
from inference.detect_enhanced import TrafficDetector

# Initialize detector
detector = TrafficDetector(
    model_path="runs/detect/traffic_train/weights/best.pt",
    confidence_threshold=0.6
)

# Process single frame
annotated_frame, detection_data = detector.detect_frame(frame)
print(detection_data)
# Output: {"vehicles": 3, "pedestrians": 1, "ambulance": False}
```

### Redis Integration

```python
from integration.redis_enhanced import RedisIntegration

# Initialize Redis client
redis_client = RedisIntegration(host="localhost", port=6379)

# Send detection data
detection_data = {
    "vehicles": 5,
    "pedestrians": 2,
    "ambulance": True,
    "total_objects": 7
}

redis_client.send_detection_data(detection_data)

# Send ambulance alert
if detection_data["ambulance"]:
    redis_client.send_alert(
        "ambulance", 
        "Ambulance detected - Priority traffic management needed",
        "critical"
    )
```

### Performance Optimization

```python
from utils.performance_utils import PerformanceOptimizer, FrameBuffer

# Initialize performance optimizer
optimizer = PerformanceOptimizer(target_fps=30)

# Use frame buffer for smooth processing
frame_buffer = FrameBuffer(max_size=10)

# Process with FPS limiting
current_time = time.time()
if optimizer.limit_fps(current_time):
    # Process frame
    result = process_frame(frame)
    optimizer.update_stats(time.time() - current_time)
```

## 🎨 Visualization Features

- **Color-coded detections**: Green for pedestrians, Blue for vehicles, Red for ambulances
- **Confidence scores**: Displayed on each bounding box
- **Real-time statistics**: Vehicle and pedestrian counts
- **Ambulance alerts**: Special highlighting and notifications
- **FPS counter**: Performance monitoring
- **Heatmaps**: Traffic density visualization

## 📈 Performance Tips

1. **Use YOLOv8n** for real-time applications
2. **Resize frames** to 640x480 for faster processing
3. **Enable frame skipping** for high-FPS sources
4. **Use GPU** if available (CUDA)
5. **Optimize Redis** connection pooling for high-throughput

## 🚨 Ambulance Detection

The system includes special handling for ambulance detection:

- **Visual Alerts**: Red highlighting and warning messages
- **Priority Events**: Special Redis alerts for ambulance detection
- **Integration Ready**: Easy integration with traffic management systems

```python
# Check for ambulance in detection results
if detection_data["ambulance"]:
    print("🚑 Ambulance detected - Priority traffic management activated")
    # Trigger traffic light changes, clear lanes, etc.
```

## 🔍 Output Format

### Detection Data

```json
{
  "vehicles": 10,
  "pedestrians": 4,
  "ambulance": true,
  "total_objects": 14,
  "detections": [
    {
      "class_id": 0,
      "class_name": "pedestrian",
      "confidence": 0.85,
      "bbox": [100, 150, 120, 200]
    }
  ]
}
```

### Redis Data Structure

```json
{
  "timestamp": 1672531200,
  "data": {
    "vehicles": 5,
    "pedestrians": 2,
    "ambulance": true,
    "total_objects": 7
  }
}
```

## 🛠️ Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

### Adding New Classes

1. Update `configs/data.yaml` with new class names
2. Update class mappings in `utils/detection_utils.py`
3. Add colors in `utils/visualization_utils.py`
4. Retrain the model

## 📋 Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics YOLOv8
- Redis (optional, for integration)
- CUDA GPU (optional, for speed)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct or use pretrained YOLOv8n
2. **Redis connection failed**: Check Redis server is running and accessible
3. **Low FPS**: Try reducing input size or enabling frame skipping
4. **No detections**: Check confidence threshold and ensure proper lighting

### Debug Mode

```bash
# Enable verbose logging
python inference/detect_enhanced.py --source 0 --conf 0.3 --verbose
```

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue on the repository

---

**Built with ❤️ for intelligent traffic management**
