from flask import Flask, render_template, jsonify, request
import yaml
import os

# If you want to run MCTS or PPO, import from your training script:
from modeling.rl_training import HexPuzzleEnv, mcts_policy, make_env_fn
# If you want PPO:
from sb3_contrib import MaskablePPO
import numpy as np

app = Flask(__name__)

# Load world and pieces data (unchanged)
with open(os.path.join("data", "world.yaml"), "r", encoding="utf-8") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r", encoding="utf-8") as f:
    pieces_data = yaml.safe_load(f)

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
    return jsonify(game_data)

# ---------------------------------------
# NEW ROUTE: GET ENEMY ACTION (MCTS OR PPO)
# ---------------------------------------
@app.route("/api/enemy_action", methods=["POST"])
def enemy_action():
    from modeling.rl_training import mcts_tree
    global mcts_tree
    mcts_tree.clear()
    
    data = request.get_json()
    scenario_in = data.get("scenario")
    approach = data.get("approach", "mcts")

    if not scenario_in:
        return jsonify({"error": "No scenario provided"}), 400

    # Build an environment from the scenario:
    env = HexPuzzleEnv(puzzle_scenario=scenario_in, max_turns=10, randomize_positions=False)
    
    # This call used to do env.reset() => randomizing + resetting
    # Instead, we now do:
    env.sync_with_puzzle_scenario(scenario_in, turn_side="enemy")

    # Now build the action list for 'enemy' side
    valid_actions = env.build_action_list()
    if not valid_actions:
        return jsonify({"error": "No valid actions for enemy side."}), 200

    # MCTS or PPO
    if approach == "mcts":
        action_idx = mcts_policy(env, max_iterations=50)
    else:
        if not ppo_model:
            return jsonify({"error": "No PPO model loaded"}), 500
        obs = env.get_obs()
        action_idx, _ = ppo_model.predict(obs, deterministic=True)

    if action_idx < 0 or action_idx >= len(valid_actions):
        return jsonify({"error": f"Chosen action_idx {action_idx} out of range"}), 200

    pidx, sub_action = valid_actions[action_idx]
    piece = env.all_pieces[pidx]

    # Return the minimal info needed:
    resp = {
        "piece_label": piece.get("label", "?"),
        "action_idx": action_idx,
        "pidx": pidx,
        "sub_action": sub_action
    }
    return jsonify(resp), 200

if __name__ == "__main__":
    app.run(debug=True)
