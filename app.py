from flask import Flask, render_template, jsonify, request
import yaml
import os
import random

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
    mcts_tree.clear()

    # Grab the incoming JSON
    data = request.get_json()
    print("[DEBUG] Received /api/enemy_action data:", data)  # <--- add debug print

    scenario_in = data.get("scenario")
    approach = data.get("approach", "mcts")

    if not scenario_in:
        return jsonify({"error": "No scenario provided"}), 400

    # Build an environment
    env = HexPuzzleEnv(puzzle_scenario=scenario_in, max_turns=10, randomize_positions=False)
    env.sync_with_puzzle_scenario(scenario_in, turn_side="enemy")

    # Check valid actions
    valid_actions = env.build_action_list()
    print("[DEBUG] Number of valid_actions:", len(valid_actions))         # how many are valid
    print("[DEBUG] valid_actions detail:", valid_actions)                # see them all

    if not valid_actions:
        print("[DEBUG] No valid actions => returning error msg.")
        return jsonify({"error": "No valid actions for enemy side."}), 200

    # Decide which approach
    if approach == "mcts":
        action_idx = mcts_policy(env, max_iterations=50)
        print("[DEBUG] MCTS chosen action_idx:", action_idx)
    elif approach == "ppo" and ppo_model is not None:
        # 1) Convert env state to observation
        obs = env.get_obs()

        # 2) If you are using ActionMasker, you typically just do:
        #    action, _ = ppo_model.predict(obs, deterministic=True)
        #    and rely on the mask automatically blocking invalid actions
        action, _ = ppo_model.predict(obs, deterministic=True)

        # But note that `action` is already an integer index in the discrete action space
        action_idx = int(action)
        print("[DEBUG] PPO chosen action_idx:", action_idx)

        # *** Important *** 
        # If the PPO model can pick an action that is out-of-range or invalid,
        # you may need to clamp it or verify it. Typically if you used
        # MaskablePPO + ActionMasker, that won't happen.
    else:
        # fallback = random
        action_idx = random.randint(0, len(valid_actions)-1)
        print("[DEBUG] Randomly chosen action_idx:", action_idx)

    # Step
    obs2, reward, done, truncated, info = env.step(action_idx)
    (pidx, chosen_subaction) = valid_actions[action_idx]
    piece_label = env.all_pieces[pidx].get("label", "?")

    print("[DEBUG] Step outcome => reward:", reward, "done:", done, 
          "truncated:", truncated, "info:", info)
    print("[DEBUG] chosen_subaction =>", chosen_subaction)

    # Return the result
    return jsonify({
        "piece_label": piece_label,
        "sub_action": chosen_subaction
    }), 200



if __name__ == "__main__":
    app.run(debug=True)
