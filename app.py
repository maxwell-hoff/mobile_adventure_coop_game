from flask import Flask, render_template, jsonify
import yaml
import os

app = Flask(__name__)

with open(os.path.join("data", "world.yaml"), "r", encoding="utf-8") as f:
    world_data = yaml.safe_load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/map_data")
def map_data():
    return jsonify(world_data)

if __name__ == "__main__":
    app.run(debug=True)
