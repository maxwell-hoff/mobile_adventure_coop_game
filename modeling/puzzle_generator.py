__doc__ = """
file: puzzle_generator.py
author: Max Hoff
date: 20250209
description: A starting point for generating candidate puzzles and evaluating them
using your RL environment (from rl_training.py). The idea is to generate
a candidate puzzle scenario, run several simulated episodes with it,
and then use the average outcome (e.g. average reward or turns taken) as a
proxy for difficulty.
"""

import random
import yaml
import copy

# Import your RL environment (and optionally your agent maker)
# from rl_training import HexPuzzleEnv, make_env_fn

# Optionally, if you have a pre-trained PPO model, you can import/load it here:
from sb3_contrib import MaskablePPO

# Load the pre-trained model from file
model = MaskablePPO.load("ppo_model")

# --- Option 1: Do not create a dummy environment here ---
# If you are only using the model for reference or predictions (and it works without setting an env),
# you can simply skip setting an environment.
#
# from stable_baselines3.common.vec_env import DummyVecEnv
# from rl_training import make_env_fn
#
# # This line caused the error because scenario_dict was not defined.
# # env = DummyVecEnv([make_env_fn(scenario_dict, randomize_positions=False)])
# # model.set_env(env)

# --- Option 2: (Alternative) Define a dummy scenario and create an environment ---
# Uncomment the following lines if your model needs an environment for proper inference.
# from stable_baselines3.common.vec_env import DummyVecEnv
# from rl_training import make_env_fn
#
# # Generate a dummy candidate scenario to serve as our "scenario_dict"
# def generate_candidate_puzzle(
#     sub_grid_radius=None,
#     num_blocked=None,
#     pieces_config=None,
#     difficulty=1
# ):
#     if sub_grid_radius is None:
#         sub_grid_radius = random.choice([3, 4]) if difficulty >= 2 else 3
#     all_hexes = []
#     for q in range(-sub_grid_radius, sub_grid_radius + 1):
#         for r in range(-sub_grid_radius, sub_grid_radius + 1):
#             if abs(q + r) <= sub_grid_radius:
#                 all_hexes.append({'q': q, 'r': r})
#     if num_blocked is None:
#         num_blocked = random.randint(1, sub_grid_radius) if difficulty >= 2 else 1
#     blocked_hexes = random.sample(all_hexes, min(num_blocked, len(all_hexes)))
#     if pieces_config is None:
#         pieces_config = [
#             { "class": "Warlock",  "label": "W", "color": "#556b2f", "side": "player", "q": -1, "r": 0 },
#             { "class": "Guardian", "label": "G", "color": "#dc143c", "side": "enemy",  "q": 1,  "r": 0 },
#             { "class": "Hunter",   "label": "H", "color": "#dc143c", "side": "enemy",  "q": 0,  "r": 1 }
#         ]
#     scenario = {
#         "name": f"Puzzle Scenario (difficulty {difficulty})",
#         "subGridRadius": sub_grid_radius,
#         "blockedHexes": blocked_hexes,
#         "pieces": pieces_config,
#         "difficulty": difficulty
#     }
#     return scenario
#
# dummy_scenario = generate_candidate_puzzle(difficulty=1)
# env = DummyVecEnv([make_env_fn(dummy_scenario, randomize_positions=False)])
# model.set_env(env)

#########################################
# 1. Candidate Puzzle Generation
#########################################

def generate_candidate_puzzle(
    sub_grid_radius=None,
    num_blocked=None,
    pieces_config=None,
    difficulty=1,
    player_min_pieces=3,
    enemy_min_pieces=3
):
    """
    Generate a candidate puzzle scenario as a dictionary.
    
    This function now ensures that:
      - The player side always has at least one Priest.
      - The enemy side always has at least one Priest and one BloodWarden.
      - The total number of pieces for each side meets the given minimum.
    
    Other parameters (like subGridRadius and blockedHexes) are generated as before.
    """
    # Choose grid size based on difficulty if not provided.
    if sub_grid_radius is None:
        sub_grid_radius = random.choice([3, 4]) if difficulty >= 2 else 3

    # Build a list of all hex coordinates for the subgrid.
    all_hexes = []
    for q in range(-sub_grid_radius, sub_grid_radius + 1):
        for r in range(-sub_grid_radius, sub_grid_radius + 1):
            if abs(q + r) <= sub_grid_radius:
                all_hexes.append({'q': q, 'r': r})
    
    # Decide how many hexes to block.
    if num_blocked is None:
        num_blocked = random.randint(1, sub_grid_radius) if difficulty >= 2 else 1
    blocked_hexes = random.sample(all_hexes, min(num_blocked, len(all_hexes)))
    
    # If no pieces_config is provided, build one using our minimums.
    if pieces_config is None:
        # --- Build player pieces ---
        player_pieces = []
        # Always include a Priest.
        player_pieces.append({
            "class": "Priest",
            "label": "P",
            "color": "#556b2f",
            "side": "player",
            "q": -1,
            "r": 0
        })
        # Fill up to player_min_pieces using other classes.
        while len(player_pieces) < player_min_pieces:
            cls = random.choice(["Warlock", "Sorcerer"])
            player_pieces.append({
                "class": cls,
                "label": cls[0],
                "color": "#556b2f",
                "side": "player",
                "q": -1,  # initial placeholder
                "r": 0
            })
        
        # --- Build enemy pieces ---
        enemy_pieces = []
        # Always include a Priest and a BloodWarden.
        enemy_pieces.append({
            "class": "Priest",
            "label": "P",
            "color": "#dc143c",
            "side": "enemy",
            "q": 1,
            "r": 0
        })
        enemy_pieces.append({
            "class": "BloodWarden",
            "label": "BW",
            "color": "#dc143c",
            "side": "enemy",
            "q": 2,
            "r": 0
        })
        # Fill up to enemy_min_pieces using other enemy classes.
        while len(enemy_pieces) < enemy_min_pieces:
            cls = random.choice(["Guardian", "Hunter"])
            enemy_pieces.append({
                "class": cls,
                "label": cls[0],
                "color": "#dc143c",
                "side": "enemy",
                "q": 1,  # placeholder
                "r": 0
            })
        pieces_config = player_pieces + enemy_pieces

    # Optionally, if difficulty is higher, adjust the enemy positions a little.
    if difficulty >= 2:
        for piece in pieces_config:
            if piece["side"] == "enemy":
                piece["q"] += random.choice([-1, 0, 1])
                piece["r"] += random.choice([-1, 0, 1])
    
    scenario = {
        "name": f"Puzzle Scenario (difficulty {difficulty})",
        "subGridRadius": sub_grid_radius,
        "blockedHexes": blocked_hexes,
        "pieces": pieces_config,
        "difficulty": difficulty
    }
    return scenario

#########################################
# 2. Puzzle Evaluation via RL Simulation
#########################################

def evaluate_puzzle(scenario, num_trials=5, approach="random", agent=None):
    """
    Evaluate the candidate puzzle by simulating episodes.
    
    This function now creates the environment with the same settings as used during training:
      - randomize_radius=True
      - randomize_blocked=True
      - randomize_pieces=True
      - player_min_pieces=3, player_max_pieces=4
      - enemy_min_pieces=3, enemy_max_pieces=5
    
    This ensures that the observation produced (of shape (18,)) matches what the PPO model expects.
    """
    total_reward = 0.0
    total_turns = 0

    from rl_training import HexPuzzleEnv  # import the environment
    for _ in range(num_trials):
        env = HexPuzzleEnv(
            puzzle_scenario=copy.deepcopy(scenario),
            max_turns=10,
            randomize_positions=True,
            randomize_radius=True,      # match training setting
            randomize_blocked=True,     # match training setting
            randomize_pieces=True,      # match training setting
            player_min_pieces=3,
            player_max_pieces=4,
            enemy_min_pieces=3,
            enemy_max_pieces=5
        )
        obs, _ = env.reset()
        done = False
        while not done:
            valid_actions = env.build_action_list()
            if not valid_actions:
                break
            if approach == "random":
                action_idx = random.randint(0, len(valid_actions)-1)
            elif approach == "ppo" and agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
                action_idx = int(action)
            elif approach == "mcts":
                action_idx = random.randint(0, len(valid_actions)-1)
            else:
                action_idx = random.randint(0, len(valid_actions)-1)
            obs, reward, done, truncated, _ = env.step(action_idx)
        final_reward = env.current_episode[-1].get("reward", 0.0)
        total_reward += final_reward
        total_turns += env.turn_number
    avg_reward = total_reward / num_trials
    avg_turns = total_turns / num_trials
    return {"avg_reward": avg_reward, "avg_turns": avg_turns}


#########################################
# 3. Putting It Together: Generate & Evaluate
#########################################

def main():
    num_candidates = 10
    candidate_puzzles = []
    
    for i in range(num_candidates):
        difficulty = random.randint(1, 10)
        candidate = generate_candidate_puzzle(difficulty=difficulty)
        evaluation = evaluate_puzzle(candidate, num_trials=5, approach="ppo", agent=model)
        candidate["evaluation"] = evaluation
        
        candidate_puzzles.append(candidate)
        print(f"Candidate {i+1} (difficulty {difficulty}) evaluated as: {evaluation}")
    
    with open("data/generated_puzzles.yaml", "w") as f:
        yaml.dump(candidate_puzzles, f, sort_keys=False)
    print("Saved candidate puzzles to generated_puzzles.yaml")

if __name__ == "__main__":
    main()
