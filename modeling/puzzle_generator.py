"""
file: puzzle_generator.py
author: Max Hoff
description:
  Generates candidate puzzles for a Hex-based scenario, then evaluates them
  by having a PPO RL agent self-play. The puzzle's 'difficulty' is determined by
  how often the player side (always going first) wins within a 10-turn limit,
  compared to enemy wins or timeouts. Puzzles with zero player wins are considered
  unwinnable and are filtered out.
"""

import random
import yaml
import copy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your RL environment and an env-maker function
from rl_training import HexPuzzleEnv, make_env_fn

# Load a pre-trained PPO model from file
model = MaskablePPO.load("ppo_model")

#########################################
# 1. Candidate Puzzle Generation
#########################################

def generate_candidate_puzzle(
    sub_grid_radius=None,
    num_blocked=None,
    pieces_config=None,
):
    """
    Generate a candidate puzzle scenario as a dictionary.

    - By default, picks a sub-grid radius randomly (3 or 4).
    - Randomly blocks a few hexes.
    - Ensures both sides have exactly 2 pieces: 1 Priest + 1 other class.
    - Positions are randomly assigned to non-blocked hexes.
    """
    # Choose random sub-grid radius if not provided
    if sub_grid_radius is None:
        sub_grid_radius = random.choice([3, 4])  # or fix to 3 if you prefer

    # Build a list of all hex coordinates for the subgrid
    all_hexes = []
    for q in range(-sub_grid_radius, sub_grid_radius + 1):
        for r in range(-sub_grid_radius, sub_grid_radius + 1):
            if abs(q + r) <= sub_grid_radius:
                all_hexes.append({'q': q, 'r': r})

    # Decide how many hexes to block; pick randomly but not too large
    if num_blocked is None:
        # just pick some small random count. Adjust as you wish
        max_to_block = max(1, sub_grid_radius)  # e.g., up to radius
        num_blocked = random.randint(1, max_to_block)
    blocked_hexes = random.sample(all_hexes, min(num_blocked, len(all_hexes)))

    # Build a set of blocked positions (as tuples)
    def hex_tuple(h):
        return (h['q'], h['r'])
    blocked_set = {hex_tuple(h) for h in blocked_hexes}

    # Build a list of available hex positions (as tuples) excluding blocked ones
    available_hexes = [hex_tuple(h) for h in all_hexes if hex_tuple(h) not in blocked_set]

    # If no pieces_config is provided, build one with exactly 2 pieces per side:
    #   1 Priest + 1 random class
    if pieces_config is None:
        # Some possible non-Priest classes:
        possible_classes = ["Warlock", "Sorcerer", "Guardian", "BloodWarden", "Hunter"]

        # --- Build player pieces ---
        player_priest = {
            "class": "Priest",
            "label": "P",
            "color": "#556b2f",
            "side": "player",
            "q": None,
            "r": None
        }
        player_other_class = random.choice(possible_classes)
        player_other = {
            "class": player_other_class,
            "label": player_other_class[0],  # e.g. 'W', 'S', 'G', 'B', 'H'
            "color": "#556b2f",
            "side": "player",
            "q": None,
            "r": None
        }
        player_pieces = [player_priest, player_other]

        # --- Build enemy pieces ---
        enemy_priest = {
            "class": "Priest",
            "label": "P",
            "color": "#dc143c",
            "side": "enemy",
            "q": None,
            "r": None
        }
        enemy_other_class = random.choice(possible_classes)
        enemy_other = {
            "class": enemy_other_class,
            "label": enemy_other_class[0],  # e.g. 'W', 'S', 'G', 'B', 'H'
            "color": "#dc143c",
            "side": "enemy",
            "q": None,
            "r": None
        }
        enemy_pieces = [enemy_priest, enemy_other]

        pieces_config = player_pieces + enemy_pieces

    # Make sure we have enough hexes to place the 4 pieces
    total_pieces = len(pieces_config)
    if total_pieces > len(available_hexes):
        raise ValueError("Not enough available hexes to place all pieces.")

    # Randomly assign unique positions from available_hexes
    assigned_positions = random.sample(available_hexes, total_pieces)
    for i, pos in enumerate(assigned_positions):
        pieces_config[i]["q"] = pos[0]
        pieces_config[i]["r"] = pos[1]

    scenario = {
        "name": "Puzzle Scenario",
        "subGridRadius": sub_grid_radius,
        "blockedHexes": blocked_hexes,
        "pieces": pieces_config
    }
    return scenario


#########################################
# 2. Puzzle Evaluation via RL Simulations
#########################################

def evaluate_puzzle(scenario, num_simulations=5):
    """
    Evaluate the puzzle by self-play using the PPO model.
    Runs `num_simulations` episodes. Each episode:
      - Player side moves first (env defaults).
      - The same PPO model is used for both player & enemy turns.
      - If final reward >= +30 and final turn_side == 'player', that's a PLAYER WIN.
      - If final reward >= +30 and final turn_side == 'enemy', that's an ENEMY WIN.
      - If final reward <= -30, whichever side had the final turn was the winner
        (meaning the other side got wiped). So if final turn_side='player' and rew <= -30,
        the enemy actually won, etc.
      - If final reward == -20 or we hit turn limit => that's a 'draw/timeout'.

    Returns a dict with tallies of wins/losses/draws and a 'puzzle_difficulty' measure
    you can define as you like. Here, we'll define:
      puzzle_difficulty = (enemy_wins - player_wins), for example.
    """
    player_wins = 0
    enemy_wins = 0
    draws = 0

    for _ in range(num_simulations):
        env = HexPuzzleEnv(
            puzzle_scenario=copy.deepcopy(scenario),
            max_turns=10,
            randomize_positions=False
        )
        obs, _ = env.reset()
        done = False

        while not done:
            # PPO picks an action for the current side (player or enemy)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)

        # Evaluate final outcome
        final_step = env.current_episode[-1]
        final_rew = final_step["reward"]
        final_side = final_step["turn_side"]  # side that took the last turn

        if final_rew >= 30:
            # The side that moved last is the winner
            if final_side == "player":
                player_wins += 1
            else:
                enemy_wins += 1
        elif final_rew <= -30:
            # If final_side = 'player' and reward <= -30 => the *enemy* effectively won
            if final_side == "player":
                enemy_wins += 1
            else:
                player_wins += 1
        else:
            # Usually -20 => timed out => draw
            draws += 1

    # You can define "puzzle_difficulty" however you like
    # e.g. a puzzle is "hard" if the player rarely wins
    # We'll do a simple numeric measure: enemy_wins - player_wins
    puzzle_difficulty = enemy_wins - player_wins

    return {
        "player_wins": player_wins,
        "enemy_wins": enemy_wins,
        "draws": draws,
        "puzzle_difficulty": puzzle_difficulty
    }


#########################################
# 3. Putting It Together: Generate & Evaluate
#########################################

def main():
    num_candidates = 10
    num_simulations_per_puzzle = 1000  # You can configure

    candidate_puzzles = []

    for i in range(num_candidates):
        candidate = generate_candidate_puzzle()
        evaluation = evaluate_puzzle(candidate, num_simulations=num_simulations_per_puzzle)

        # Filter out if unwinnable (player_wins == 0)
        if evaluation["player_wins"] == 0:
            # skip unwinnable puzzles
            print(f"Candidate {i+1} is unwinnable (player never wins). Excluding.")
            continue

        # Attach evaluation data
        candidate["evaluation"] = evaluation
        candidate_puzzles.append(candidate)

        print(f"Candidate {i+1} => W:{evaluation['player_wins']}  "
              f"L:{evaluation['enemy_wins']}  D:{evaluation['draws']}  "
              f"Difficulty:{evaluation['puzzle_difficulty']}")

    # Save the valid puzzles to YAML
    if candidate_puzzles:
        with open("data/generated_puzzles.yaml", "w") as f:
            yaml.dump(candidate_puzzles, f, sort_keys=False)
        print("Saved candidate puzzles to generated_puzzles.yaml")
    else:
        print("No valid (winnable) puzzles were generated.")

if __name__ == "__main__":
    main()
