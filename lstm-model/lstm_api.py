from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    traffic = data.get("traffic",[10,20,30])
    return jsonify({"predicted_density": sum(traffic)/len(traffic)})

app.run(port=5002)
