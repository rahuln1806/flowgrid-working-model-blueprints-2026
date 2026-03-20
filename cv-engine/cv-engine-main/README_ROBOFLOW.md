# 🚦 Roboflow Traffic Analysis CV Engine

A complete computer vision system using YOLOv8 with Roboflow dataset for real-time traffic analysis, detecting pedestrians and vehicles.

## 🎯 Roboflow Integration Features

- **Roboflow Dataset**: Uses traffic dataset from Roboflow with 2 classes
- **Class Mapping**: Maps Roboflow classes to traffic analysis classes
- **Real-time Detection**: Detects pedestrians and vehicles in real-time
- **YOLOv8 Powered**: Uses state-of-the-art YOLOv8 model for accurate detection
- **Performance Optimized**: Frame resizing, skipping, and multi-threading support
- **Redis Integration**: Real-time data sharing and alert system
- **Modular Design**: Clean, extensible architecture

## 🏗️ Project Structure with Roboflow

```
cv-engine/
├── traffic-9/                    # Downloaded Roboflow dataset
│   ├── train/
│   │   ├── images/              # Training images
│   │   └── labels/              # Training labels
│   ├── valid/
│   │   ├── images/              # Validation images
│   │   └── labels/              # Validation labels
│   └── data.yaml                # Roboflow dataset config
├── configs/
│   ├── data.yaml                # Updated dataset configuration
│   └── class_mapping.yaml       # Class mapping configuration
├── training/
│   └── train_roboflow.py         # Roboflow-specific training
├── inference/
│   └── detect_roboflow.py        # Roboflow-specific detection
├── setup_dataset.py             # Roboflow setup script
├── test_roboflow_setup.py       # Setup verification script
└── README_ROBOFLOW.md            # This file
```

## 🚀 Quick Start with Roboflow

### 1. Setup Dataset (Already Done)

The Roboflow dataset has been downloaded and configured:

```bash
# Verify setup
python test_roboflow_setup.py
```

### 2. Train Model

```bash
# Train on Roboflow dataset
python training/train_roboflow.py

# With custom parameters
python training/train_roboflow.py \
    --model yolov8n.pt \
    --epochs 50 \
    --img-size 640 \
    --batch-size 8
```

### 3. Run Detection

```bash
# Test with webcam
python inference/detect_roboflow.py --source 0

# Test with video file
python inference/detect_roboflow.py --source video.mp4 --save

# With custom confidence
python inference/detect_roboflow.py --source 0 --conf 0.7
```

## 📊 Roboflow Dataset Classes

| Dataset Class ID | Dataset Name | Traffic Class | Description |
|------------------|--------------|---------------|-------------|
| 0                | non-motor-vehicle | vehicle | Bicycles, motorcycles, etc. |
| 1                | person | pedestrian | People walking, standing |

### Class Mapping

The system maps Roboflow classes to traffic analysis classes:

```yaml
mapping:
  0: 1  # non-motor-vehicle -> vehicle
  1: 0  # person -> pedestrian
```

## 🔧 Configuration Files

### Dataset Configuration (`configs/data.yaml`)

```yaml
names:
- non-motor-vehicle
- person
nc: 2
train: traffic-9/train/images
val: traffic-9/valid/images
```

### Class Mapping (`configs/class_mapping.yaml`)

```yaml
dataset_classes:
  0: non-motor-vehicle
  1: person
traffic_classes:
  0: pedestrian
  1: vehicle
  2: ambulance
mapping:
  0: 1  # non-motor-vehicle -> vehicle
  1: 0  # person -> pedestrian
```

## 📱 Usage Examples

### Basic Detection with Roboflow Model

```python
from inference.detect_roboflow import RoboflowTrafficDetector

# Initialize detector
detector = RoboflowTrafficDetector(
    model_path="runs/detect/roboflow_traffic/weights/best.pt",
    confidence_threshold=0.6
)

# Process single frame
annotated_frame, detection_data = detector.detect_frame(frame)
print(detection_data)
# Output: {"vehicles": 3, "pedestrians": 1, "ambulance": False}
```

### Training with Roboflow Dataset

```python
from training.train_roboflow import train_roboflow_model

# Train model
results = train_roboflow_model(
    epochs=100,
    img_size=640,
    batch_size=16
)
```

## 🎨 Detection Output

### Detection Data Format

```json
{
  "vehicles": 5,
  "pedestrians": 2,
  "ambulance": false,
  "total_objects": 7,
  "detections": [
    {
      "dataset_class_id": 0,
      "class_id": 1,
      "class_name": "vehicle",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 250]
    },
    {
      "dataset_class_id": 1,
      "class_id": 0,
      "class_name": "pedestrian",
      "confidence": 0.92,
      "bbox": [300, 400, 350, 500]
    }
  ]
}
```

### Visual Output

- **Green boxes**: Pedestrians
- **Blue boxes**: Vehicles (non-motor-vehicles from Roboflow)
- **Confidence scores**: Displayed on each bounding box
- **Real-time statistics**: Vehicle and pedestrian counts

## 🔍 Testing and Verification

### Run Setup Test

```bash
python test_roboflow_setup.py
```

This test verifies:
- ✅ Dataset directories exist
- ✅ Configuration files are correct
- ✅ Images can be loaded
- ✅ Model can be loaded
- ✅ Inference setup works

### Manual Testing

```bash
# Test with a sample image (if available)
python inference/detect_roboflow.py --source path/to/image.jpg

# Test with webcam
python inference/detect_roboflow.py --source 0 --conf 0.5
```

## 📈 Performance Tips

1. **Use YOLOv8n** for real-time applications
2. **Adjust confidence threshold** based on your use case
3. **Enable frame skipping** for high-FPS sources
4. **Use GPU** if available (CUDA)
5. **Resize input frames** for faster processing

## 🔄 Extending the Dataset

### Adding More Classes

1. **Upload to Roboflow**: Add more classes to your Roboflow project
2. **Download new version**: Update the version number in setup
3. **Update class mapping**: Modify `configs/class_mapping.yaml`
4. **Retrain model**: Run training with new data

### Custom Class Mapping

Edit `configs/class_mapping.yaml`:

```yaml
# Example: Adding ambulance class
dataset_classes:
  0: non-motor-vehicle
  1: person
  2: ambulance
traffic_classes:
  0: pedestrian
  1: vehicle
  2: ambulance
mapping:
  0: 1  # non-motor-vehicle -> vehicle
  1: 0  # person -> pedestrian
  2: 2  # ambulance -> ambulance
```

## 🚨 Important Notes

- **Dataset Classes**: The current Roboflow dataset only has 2 classes (non-motor-vehicle, person)
- **Ambulance Detection**: Not available in current dataset (mapped to vehicle by default)
- **Class Mapping**: Essential for converting Roboflow classes to traffic classes
- **Model Training**: Required before using custom detection

## 🆘 Troubleshooting

### Common Issues

1. **No images found**: Check dataset paths in `configs/data.yaml`
2. **Class mapping errors**: Verify `configs/class_mapping.yaml` is correct
3. **Model loading fails**: Ensure training completed successfully
4. **Poor detection**: Adjust confidence threshold or retrain with more data

### Debug Commands

```bash
# Check dataset structure
ls -la traffic-9/train/images/
ls -la traffic-9/valid/images/

# Verify configuration
cat configs/data.yaml
cat configs/class_mapping.yaml

# Test setup
python test_roboflow_setup.py
```

## 📞 Support

For issues specific to Roboflow integration:
1. Check the troubleshooting section
2. Verify dataset setup with test script
3. Review configuration files
4. Check Roboflow project settings

---

**Built with ❤️ for intelligent traffic management using Roboflow datasets**
