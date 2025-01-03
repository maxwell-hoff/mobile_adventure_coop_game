from flask import Flask, render_template, jsonify
import yaml
import os

app = Flask(__name__)

# Load world and pieces data
with open(os.path.join("data", "world.yaml"), "r", encoding="utf-8") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r", encoding="utf-8") as f:
    pieces_data = yaml.safe_load(f)

# Combine the data
game_data = {
    "world": world_data,
    "pieces": pieces_data
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/map_data")
def map_data():
    return jsonify(game_data)

if __name__ == "__main__":
    app.run(debug=True)
