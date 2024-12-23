from flask import Flask, render_template, jsonify
import yaml
import os

app = Flask(__name__)

# Load YAML once on startup
with open(os.path.join('data', 'world.yaml'), 'r', encoding='utf-8') as f:
    world_data = yaml.safe_load(f)

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/api/map_data')
def map_data():
    """Return the entire YAML as JSON."""
    return jsonify(world_data)

if __name__ == '__main__':
    app.run(debug=True)
