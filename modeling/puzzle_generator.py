__doc__ = """
file: puzzle_generator.py
author: Max Hoff
date: 20250209
description: A starting point for generating candidate puzzles and evaluating them
using your RL environment (from rl_training.py). The idea is to generate
a candidate puzzle scenario, run several simulated episodes with it,
and then use the average outcome (e.g. average reward or turns taken) as a
proxy for difficulty. You can later select or adjust puzzles based on that.
"""

import random
import yaml
import copy

# Import your RL environment (and optionally your agent maker)
# from rl_training import HexPuzzleEnv, make_env_fn

# Optionally, if you have a pre-trained PPO model, you can import/load it here:
from sb3_contrib import MaskablePPO

# Load the pre-trained model from file
model = MaskablePPO.load("ppo_model.zip")

# If your model requires a specific environment (e.g., wrapped with ActionMasker),
# you might need to provide a dummy or the actual environment.
# For example:
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_training import make_env_fn

# Create an environment instance
env = DummyVecEnv([make_env_fn(scenario_dict, randomize_positions=False)])
model.set_env(env)


#########################################
# 1. Candidate Puzzle Generation
#########################################

def generate_candidate_puzzle(
    sub_grid_radius=None,
    num_blocked=None,
    pieces_config=None,
    difficulty=1
):
    """
    Generate a candidate puzzle scenario as a dictionary.
    
    The candidate includes:
      - subGridRadius: The radius of the puzzle grid.
      - blockedHexes: A list of hexes (dicts with q and r) that are blocked.
      - pieces: A list of piece dictionaries.
      - difficulty: A parameter that might affect other values.
    
    You can make this more elaborate by adding more parameters or changing
    how pieces are placed based on the difficulty.
    """
    # If no radius is given, choose one based on difficulty (for example, harder puzzles may use a larger grid)
    if sub_grid_radius is None:
        sub_grid_radius = random.choice([3, 4]) if difficulty >= 2 else 3

    # Build all hex coordinates for the subgrid.
    all_hexes = []
    for q in range(-sub_grid_radius, sub_grid_radius + 1):
        for r in range(-sub_grid_radius, sub_grid_radius + 1):
            if abs(q + r) <= sub_grid_radius:
                all_hexes.append({'q': q, 'r': r})
    
    # If no blocked hexes count is provided, decide based on difficulty.
    if num_blocked is None:
        # For example: higher difficulty might mean more obstacles (or fewer, depending on your design)
        num_blocked = random.randint(1, sub_grid_radius) if difficulty >= 2 else 1
    # Randomly sample some hexes to be blocked.
    blocked_hexes = random.sample(all_hexes, min(num_blocked, len(all_hexes)))

    # Define a basic pieces configuration if none is provided.
    if pieces_config is None:
        # For example, one player piece and a few enemy pieces.
        pieces_config = [
            { "class": "Warlock",  "label": "W", "color": "#556b2f", "side": "player", "q": -1, "r": 0 },
            { "class": "Guardian", "label": "G", "color": "#dc143c", "side": "enemy",  "q": 1,  "r": 0 },
            { "class": "Hunter",   "label": "H", "color": "#dc143c", "side": "enemy",  "q": 0,  "r": 1 }
        ]
    
    # Optionally adjust enemy positions based on difficulty.
    if difficulty >= 2:
        for piece in pieces_config:
            if piece["side"] == "enemy":
                piece["q"] += random.choice([-1, 0, 1])
                piece["r"] += random.choice([-1, 0, 1])
    
    # Assemble the candidate puzzle scenario.
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
    Evaluate the candidate puzzle by simulating a number of episodes.
    
    Parameters:
      - scenario: The candidate puzzle scenario (a dict).
      - num_trials: How many episodes to simulate.
      - approach: Which action policy to use: "random", "mcts", or "ppo".
      - agent: If using PPO (or another agent), pass it here.
      
    Returns:
      A metric that can serve as a proxy for difficulty (for example, the average reward).
      (Depending on your design, a more negative reward or more turns taken might mean a higher challenge.)
    """
    total_reward = 0.0
    total_turns = 0

    # For each trial, we create a new environment instance.
    for _ in range(num_trials):
        env = HexPuzzleEnv(
            puzzle_scenario=copy.deepcopy(scenario),
            max_turns=10,  # you may want to set this based on difficulty
            randomize_positions=False
        )
        obs, _ = env.reset()
        done = False
        while not done:
            valid_actions = env.build_action_list()
            if not valid_actions:
                break
            # Select an action based on the chosen approach.
            if approach == "random":
                action_idx = random.randint(0, len(valid_actions)-1)
            elif approach == "ppo" and agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
                action_idx = int(action)
            elif approach == "mcts":
                # For now, we use a simple random choice as a placeholder.
                action_idx = random.randint(0, len(valid_actions)-1)
            else:
                action_idx = random.randint(0, len(valid_actions)-1)
            obs, reward, done, truncated, _ = env.step(action_idx)
        # Use the final episode reward and/or number of turns as an evaluation metric.
        # (For example, a puzzle that ends with a low reward or takes many turns may be harder.)
        final_reward = env.current_episode[-1].get("reward", 0.0)
        total_reward += final_reward
        total_turns += env.turn_number
    avg_reward = total_reward / num_trials
    avg_turns = total_turns / num_trials
    # You could return a tuple or a dict with multiple metrics.
    return {"avg_reward": avg_reward, "avg_turns": avg_turns}

#########################################
# 3. Putting It Together: Generate & Evaluate
#########################################

def main():
    num_candidates = 10
    candidate_puzzles = []
    
    for i in range(num_candidates):
        # You can vary difficulty here (or even cycle through different values)
        difficulty = random.randint(1, 3)
        candidate = generate_candidate_puzzle(difficulty=difficulty)
        
        # Evaluate the candidate puzzle.
        # You can choose "random" for a baseline evaluation or "ppo" if you have an agent.
        evaluation = evaluate_puzzle(candidate, num_trials=5, approach="random")
        candidate["evaluation"] = evaluation
        
        candidate_puzzles.append(candidate)
        print(f"Candidate {i+1} (difficulty {difficulty}) evaluated as: {evaluation}")
    
    # Write the candidates out to a YAML file.
    with open("generated_puzzles.yaml", "w") as f:
        yaml.dump(candidate_puzzles, f, sort_keys=False)
    print("Saved candidate puzzles to generated_puzzles.yaml")

if __name__ == "__main__":
    main()
