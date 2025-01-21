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
    data = request.get_json()
    scenario_in = data.get("scenario")
    approach = data.get("approach", "mcts")

    if not scenario_in:
        return jsonify({"error": "No scenario provided"}), 400

    # 1) Rebuild environment from scenario
    env = HexPuzzleEnv(puzzle_scenario=scenario_in, max_turns=10, randomize_positions=False)
    env.sync_with_puzzle_scenario(scenario_in, turn_side="enemy")

    # 2) While env.turn_side == "enemy" and we have valid actions, pick one
    #    This means *all* enemy moves happen in one request.
    actions_taken = []
    while env.turn_side == "enemy":
        valid_actions = env.build_action_list()
        if not valid_actions:
            break

        if approach == "mcts":
            action_idx = mcts_policy(env, max_iterations=50)
        else:
            # PPO or random, etc.
            action_idx = random.randint(0, len(valid_actions)-1)

        # Step
        obs2, reward, done, truncated, _ = env.step(action_idx)
        (pidx, chosen_subaction) = valid_actions[action_idx]
        piece_label = env.all_pieces[pidx].get("label", "?")

        actions_taken.append({
            "piece_label": piece_label,
            "sub_action": chosen_subaction
        })

        if done:
            break

    # 3) Return list of all sub_actions so frontend can apply each one in order
    #    If you only want a single sub_action per request, skip this approach
    if not actions_taken:
        return jsonify({"error":"No valid actions for enemy side."}), 200
    
    return jsonify({"actions": actions_taken}), 200


if __name__ == "__main__":
    app.run(debug=True)
