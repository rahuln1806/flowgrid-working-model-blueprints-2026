from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/simulate")
def simulate():
    return jsonify({"signals":[{"id":"INT-01","state":"green"}]})

app.run(port=5003)
