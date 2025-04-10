import os
import json
import redis
import yaml
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hex_puzzle_env import HexPuzzleEnv
from hex_puzzle import HexPuzzle
from hex_puzzle_utils import load_scenario, load_piece_classes, load_region_data
import random

app = Flask(__name__)
CORS(app)

# Redis connection with retry logic
def get_redis_client():
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
    except redis.ConnectionError as e:
        print(f"Redis connection error: {e}")
        return None

redis_client = get_redis_client()

# ... existing code ...

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
    else:
        action_idx = random.randint(0, len(valid_actions)-1)
        print("[DEBUG] Randomly chosen action_idx:", action_idx)

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

if __name__ == "__main__":
    app.run(debug=True)

# ... existing code ... 