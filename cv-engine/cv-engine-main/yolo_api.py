from flask import Flask, request, jsonify
from ultralytics import YOLO
app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    file.save("temp.jpg")
    results = model("temp.jpg")
    detections=[]
    for r in results:
        for b in r.boxes:
            detections.append({"class":int(b.cls[0]),"confidence":float(b.conf[0])})
    return jsonify({"detections":detections})

app.run(port=5001)

    