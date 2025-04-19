from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import yaml
import os
import random
import sqlite3
from passlib.hash import bcrypt
import redis
import json
from game_logic import GameState, Piece, HexCoordinate

# If you want to run MCTS or PPO, import from your training script:
from modeling.rl_training import HexPuzzleEnv, mcts_policy, make_env_fn
# If you want PPO:
from sb3_contrib import MaskablePPO
import numpy as np


app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_SOMETHING_SECRET"

# Redis connection with retry logic
def get_redis_client():
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")
        return None

redis_client = get_redis_client()

if not redis_client:
    print("Warning: Failed to connect to Redis. Some features may not work properly.")

# Load world and pieces data (unchanged)
with open(os.path.join("data", "world.yaml"), "r", encoding="utf-8") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r", encoding="utf-8") as f:
    pieces_data = yaml.safe_load(f)

# Build a list of valid classes (excluding "Bloodwarden" and "Priest")
all_classes = [
    piece
    for piece in pieces_data['classes'].keys()
    if piece not in ("BloodWarden", "Priest")
]

game_data = {
    "world": world_data,
    "pieces": pieces_data
}

# (Optional) If you want PPO, load it if you have a saved model:
ppo_model = None
MODEL_PATH = "ppo_model.zip"  # or wherever you stored it
if os.path.exists(MODEL_PATH):
    # Potentially wrap in a dummy env if needed for maskable usage:
    # scenario_0 = world_data["regions"][0]["puzzleScenarios"][0]
    # env_0 = make_env_fn(scenario_0, randomize=False)()
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # vec_env = DummyVecEnv([lambda: env_0])
    # ppo_model = MaskablePPO.load(MODEL_PATH, vec_env)
    ppo_model = MaskablePPO.load(MODEL_PATH)
    print("Loaded PPO model from", MODEL_PATH)
else:
    print("No PPO model found. PPO approach may fail if called.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/map_data")
def map_data():
    # Query characters from the database:
    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, char_name, char_class, regionLocation, detailLocation, level FROM characters")
        rows = cur.fetchall()

    characters = []
    for r in rows:
        characters.append({
            "id": r[0],
            "name": r[1],
            "char_class": r[2],
            "regionLocation": r[3],  # stored as "q,r"
            "detailLocation": r[4],  # stored as "q,r"
            "level": r[5]
        })

    game_data = {
        "world": world_data,
        "pieces": pieces_data,
        "characters": characters
    }
    return jsonify(game_data)


# ---------------------------------------
# 1. Initialize & Migrate SQLite Database
# ---------------------------------------
def init_db():
    with sqlite3.connect("userData.db") as conn:
        # Ensure 'users' table exists
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            passwordHash TEXT,
            userClass TEXT,
            currentLocation TEXT
        );
        """)
        # Create the 'characters' table with two location fields.
        conn.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            char_name TEXT,
            char_class TEXT,
            regionLocation TEXT, 
            detailLocation TEXT, 
            level INTEGER DEFAULT 1,
            UNIQUE(user_id, char_name),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
        conn.commit()

init_db()

# -------------------------
# Sign Up
# -------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    if not data:
        return jsonify(error="No sign-up data provided"), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify(error="Username and password are required."), 400

    # Check if user exists
    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        existing = cur.fetchone()
        if existing:
            return jsonify(error="Username already exists."), 400

        # Hash password
        hashed_pw = bcrypt.hash(password)
        # Insert user
        cur.execute("""INSERT INTO users (username, passwordHash) VALUES (?, ?)""",
                    (username, hashed_pw))
        conn.commit()

    return jsonify(success=True, message="User registered successfully"), 200

# -------------------------
# Login
# -------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify(error="No login data provided"), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify(error="Username and password are required."), 400

    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, passwordHash
            FROM users
            WHERE username = ?
        """, (username,))
        row = cur.fetchone()

    if not row:
        return jsonify(error="Invalid credentials (no user)."), 400

    user_id, stored_hash = row

    # verify password
    if not bcrypt.verify(password, stored_hash):
        return jsonify(error="Invalid credentials (bad password)."), 400

    # store user_id in session
    session["user_id"] = user_id
    return jsonify(success=True, message="Logged in successfully."), 200

# -------------------------
# Logout (optional)
# -------------------------
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify(success=True, message="Logged out.")

# -------------------------
# Get Available Classes
# (We exclude Bloodwarden/Priest)
# -------------------------
@app.route("/get_classes", methods=["GET"])
def get_classes():
    # Return the classes we built earlier
    return jsonify(all_classes), 200

# -------------------------
# Get Characters for the Logged-In User
# -------------------------
@app.route("/get_characters", methods=["GET"])
def get_characters():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="Not logged in"), 401

    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, char_name, char_class, regionLocation, detailLocation, level
            FROM characters
            WHERE user_id = ?
        """, (user_id,))
        rows = cur.fetchall()

    # Convert to list of dicts
    char_list = []
    for r in rows:
        char_list.append({
            "id": r[0],
            "name": r[1],
            "char_class": r[2],
            "location": r[3],
            "level": r[4],
        })

    return jsonify(char_list), 200

# -------------------------
# Create a New Character
# -------------------------
@app.route("/create_character", methods=["POST"])
def create_character():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="Not logged in"), 401

    data = request.get_json()
    if not data:
        return jsonify(error="No character data provided"), 400

    char_name = data.get("char_name")
    char_class = data.get("char_class")
    location = "regionId=1|q=0|r=0"  
    region = "regionId=1"
    detail = "q=0|r=0"

    if not char_name or not char_class:
        return jsonify(error="Name and class are required for new character."), 400

    # Check if class is valid
    if char_class not in all_classes:
        return jsonify(error=f"Invalid class. Must be one of {all_classes}."), 400

    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO characters (user_id, char_name, char_class, regionLocation, detailLocation, level)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (user_id, char_name, char_class, region, detail))
            conn.commit()
        except sqlite3.IntegrityError:
            # likely UNIQUE constraint for (user_id, char_name)
            return jsonify(error=f"Character name '{char_name}' already exists for this account."), 400

    return jsonify(success=True, message="Character created successfully."), 200

# -------------------------
# (Optional) "Begin Game" or "Load Character"
# This is where you'd handle continuing to the next screen, etc.
# -------------------------
@app.route("/load_character/<int:char_id>", methods=["GET"])
def load_character(char_id):
    """ Example route if you wanted to load the puzzle for a specific character. """
    # You can store chosen char_id in session, or return details to front-end
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error="Not logged in"), 401

    with sqlite3.connect("userData.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, char_name, char_class, location, level
            FROM characters
            WHERE id = ? AND user_id = ?
        """, (char_id, user_id))
        row = cur.fetchone()

    if not row:
        return jsonify(error="Character not found or not yours."), 400

    # In a real game, you'd do more logic. We'll just return the data:
    return jsonify({
        "char_id": row[0],
        "name": row[1],
        "char_class": row[2],
        "location": row[3],
        "level": row[4]
    }), 200


# ---------------------------------------
# GET ENEMY ACTION (MCTS OR PPO)
# ---------------------------------------
@app.route("/api/enemy_action", methods=["POST"])
def enemy_action():
    if not redis_client:
        return jsonify({"error": "Redis connection failed"}), 500

    from modeling.rl_training import mcts_tree
    mcts_tree.clear()

    data = request.get_json()
    print("[DEBUG] Received /api/enemy_action data:", data)

    scenario_in = data.get("scenario")
    approach = data.get("approach", "mcts")

    if not scenario_in:
        return jsonify({"error": "No scenario provided"}), 400

    env = HexPuzzleEnv(puzzle_scenario=scenario_in, max_turns=10, randomize_positions=False)
    env.sync_with_puzzle_scenario(scenario_in, turn_side="enemy")

    valid_actions = env.build_action_list()
    print("[DEBUG] Number of valid_actions:", len(valid_actions))
    print("[DEBUG] valid_actions detail:", valid_actions)

    if not valid_actions:
        print("[DEBUG] No valid actions => returning error msg.")
        return jsonify({"error": "No valid actions for enemy side."}), 200

    if approach == "mcts":
        action_idx = mcts_policy(env, max_iterations=50)
        print("[DEBUG] MCTS chosen action_idx:", action_idx)
    elif approach == "ppo" and ppo_model is not None:
        obs = env.get_obs()
        action, _ = ppo_model.predict(obs, deterministic=True)
        action_idx = int(action)
        print("[DEBUG] PPO chosen action_idx:", action_idx)
        # Validate action index is within valid range
        if action_idx >= len(valid_actions):
            print(f"[WARNING] PPO predicted invalid action {action_idx}, falling back to random action")
            action_idx = random.randint(0, len(valid_actions)-1)
    else:
        action_idx = random.randint(0, len(valid_actions)-1)
        print("[DEBUG] Randomly chosen action_idx:", action_idx)

    # Double check action index is valid
    if action_idx < 0 or action_idx >= len(valid_actions):
        print(f"[ERROR] Invalid action index {action_idx} (valid range: 0-{len(valid_actions)-1})")
        return jsonify({"error": "Invalid action index"}), 500

    obs2, reward, done, truncated, info = env.step(action_idx)
    (pidx, chosen_subaction) = valid_actions[action_idx]
    piece_label = env.all_pieces[pidx].get("label", "?")

    print("[DEBUG] Step outcome => reward:", reward, "done:", done, 
          "truncated:", truncated, "info:", info)
    print("[DEBUG] chosen_subaction =>", chosen_subaction)

    result = {
        "piece_label": piece_label,
        "sub_action": chosen_subaction
    }

    try:
        action_record = {
            "type": "enemy_action",
            "data": result
        }
        redis_client.lpush("game_actions", json.dumps(action_record))
        redis_client.ltrim("game_actions", 0, 1000)
    except redis.RedisError as e:
        print(f"Redis error when storing action: {e}")
        return jsonify({"error": "Failed to store action"}), 500

    return jsonify(result), 200

# (Optional) Add a route to retrieve actions for polling
@app.route("/api/get_actions", methods=["GET"])
def get_actions():
    if not redis_client:
        return jsonify({"error": "Redis connection failed"}), 500

    try:
        actions_raw = redis_client.lrange("game_actions", 0, -1)
        actions_list = [json.loads(a) for a in actions_raw]
        return jsonify(actions_list), 200
    except redis.RedisError as e:
        print(f"Redis error when retrieving actions: {e}")
        return jsonify({"error": "Failed to retrieve actions"}), 500
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return jsonify({"error": "Failed to parse actions"}), 500

@app.route("/api/validate_move", methods=["POST"])
def validate_move():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create game state from request data
    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    # Get move parameters
    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    target_q = data["target_q"]
    target_r = data["target_r"]
    max_range = data["max_range"]

    # Validate move
    is_valid, message = game_state.validate_move(piece, target_q, target_r, max_range)
    
    return jsonify({
        "valid": is_valid,
        "message": message
    })

@app.route("/api/apply_trap_effect", methods=["POST"])
def apply_trap_effect():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create game state from request data
    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    # Get piece and trap
    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    trap = data["trap"]

    # Apply trap effect
    message = game_state.apply_trap_effect(piece, trap)
    
    return jsonify({
        "message": message,
        "piece_state": {
            "immobilized": piece.immobilized,
            "immobilized_turns": piece.immobilized_turns
        }
    })

@app.route("/api/validate_attack", methods=["POST"])
def validate_attack():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    target_q = data["target_q"]
    target_r = data["target_r"]
    max_range = data["max_range"]
    max_targets = data.get("max_targets", 1)

    is_valid, message = game_state.validate_attack(piece, target_q, target_r, max_range, max_targets)
    
    return jsonify({
        "valid": is_valid,
        "message": message
    })

@app.route("/api/validate_swap_position", methods=["POST"])
def validate_swap_position():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    target_q = data["target_q"]
    target_r = data["target_r"]
    max_range = data["max_range"]
    ally_only = data.get("ally_only", False)

    is_valid, message = game_state.validate_swap_position(piece, target_q, target_r, max_range, ally_only)
    
    return jsonify({
        "valid": is_valid,
        "message": message
    })

@app.route("/api/validate_pull_push", methods=["POST"])
def validate_pull_push():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    target_q = data["target_q"]
    target_r = data["target_r"]
    max_range = data["max_range"]
    distance = data["distance"]
    is_pull = data["is_pull"]

    is_valid, message = game_state.validate_pull_push(piece, target_q, target_r, max_range, distance, is_pull)
    
    return jsonify({
        "valid": is_valid,
        "message": message
    })

@app.route("/api/validate_aoe", methods=["POST"])
def validate_aoe():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    center_q = data["center_q"]
    center_r = data["center_r"]
    max_range = data["max_range"]
    radius = data["radius"]

    is_valid, message = game_state.validate_aoe(piece, center_q, center_r, max_range, radius)
    
    return jsonify({
        "valid": is_valid,
        "message": message
    })

@app.route("/api/apply_damage", methods=["POST"])
def apply_damage():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    piece = next(p for p in pieces if p.class_ == data["piece_class"] and p.label == data["piece_label"])
    damage = data["damage"]

    message = game_state.apply_damage(piece, damage)
    
    return jsonify({
        "message": message,
        "piece_state": {
            "health": piece.health,
            "dead": piece.dead
        }
    })

# Update the end_turn endpoint to handle cooldowns and effects
@app.route("/api/end_turn", methods=["POST"])
def end_turn():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pieces = [
        Piece(
            class_=p["class"],
            label=p["label"],
            q=p["q"],
            r=p["r"],
            side=p.get("side", "player")
        ) for p in data["pieces"]
    ]
    game_state = GameState(pieces, data["blocked_hexes"])

    # Handle all turn-end effects
    messages = []
    messages.extend(game_state.decrement_immobilized_turns())
    messages.extend(game_state.decrement_cooldowns())
    messages.extend(game_state.decrement_effects())
    
    return jsonify({
        "messages": messages,
        "updated_pieces": [
            {
                "class": p.class_,
                "label": p.label,
                "immobilized": p.immobilized,
                "immobilized_turns": p.immobilized_turns,
                "health": p.health,
                "dead": p.dead,
                "cooldowns": p.cooldowns,
                "active_effects": p.active_effects
            } for p in pieces
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
