__doc__ = """
file: puzzle_generator.py
description:
  Generates candidate puzzles for a Hex-based scenario, then evaluates them
  via a PPO RL agent self-play. Each side:
    - Must have exactly 1 Priest
    - Has a random number of additional pieces from [Warlock, Sorcerer, Guardian, Hunter, BloodWarden].
  We skip any puzzle where the player can't ever win, or if the player's win
  rate exceeds a certain threshold (puzzle is too easy).
"""

import random
import yaml
import copy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_training import HexPuzzleEnv

# Load your existing PPO model
model = MaskablePPO.load("ppo_model")


############################################
# 1. Build side pieces (always 1 Priest + random others)
############################################

def build_side_pieces_with_priest(
    side: str,
    min_total_pieces: int = 2,
    max_total_pieces: int = 5
):
    """
    Create a random number (between min_total_pieces and max_total_pieces)
    of total pieces for the given side. Exactly one of these pieces is a Priest.
    The rest are chosen from [Warlock, Sorcerer, Guardian, Hunter, BloodWarden].
    
    Example:
      min_total_pieces=2, max_total_pieces=5
      - side might end up with anywhere between 2..5 total pieces
      - the first is always a Priest
      - the remainder (total_pieces - 1) are random picks from the "extra_classes".
    """
    # Always add one Priest
    color = "#556b2f" if side == "player" else "#dc143c"
    pieces = [{
        "class": "Priest",
        "label": "P",
        "color": color,
        "side": side,
        "q": None,
        "r": None
    }]

    # The other classes that either side can have
    extra_classes = ["Warlock", "Sorcerer", "Guardian", "Hunter", "BloodWarden"]

    # Decide total piece count for this side
    total_count = random.randint(min_total_pieces, max_total_pieces)
    # We already have 1 Priest, so we fill the rest with random picks
    extras_needed = total_count - 1

    for _ in range(extras_needed):
        chosen_class = random.choice(extra_classes)
        # We'll label them by their first letter (like 'W', 'S', 'G', 'H') 
        # except "BloodWarden" => maybe "B" or "BW"? We'll just do "B" for short.
        if chosen_class == "BloodWarden":
            label = "B"
        else:
            label = chosen_class[0].upper()

        piece_info = {
            "class": chosen_class,
            "label": label,
            "color": color,
            "side": side,
            "q": None,
            "r": None
        }
        pieces.append(piece_info)

    return pieces


############################################
# 2. Candidate puzzle generation
############################################

def generate_candidate_puzzle(
    sub_grid_radius=None,
    num_blocked=None,
    player_min_total=2,
    player_max_total=5,
    enemy_min_total=2,
    enemy_max_total=5
):
    """
    Generate a puzzle scenario:
      - sub_grid_radius is random [3 or 4] if not specified
      - block some random hexes
      - For the player side: exactly 1 Priest, plus random others => total in [player_min_total..player_max_total]
      - For the enemy side: exactly 1 Priest, plus random others => total in [enemy_min_total..enemy_max_total]
      - Place them on non-blocked hexes.
    """
    # 1) sub-grid radius
    if sub_grid_radius is None:
        sub_grid_radius = random.choice([3, 4])

    # gather all hex coords in sub-grid
    all_hexes = []
    for q in range(-sub_grid_radius, sub_grid_radius + 1):
        for r in range(-sub_grid_radius, sub_grid_radius + 1):
            if abs(q + r) <= sub_grid_radius:
                all_hexes.append({"q": q, "r": r})

    # 2) random blocked
    if num_blocked is None:
        max_to_block = max(1, sub_grid_radius)
        num_blocked = random.randint(1, max_to_block)
    blocked_hexes = random.sample(all_hexes, min(num_blocked, len(all_hexes)))
    blocked_set = {(h["q"], h["r"]) for h in blocked_hexes}
    available_hexes = [
        (h["q"], h["r"]) for h in all_hexes 
        if (h["q"], h["r"]) not in blocked_set
    ]

    # 3) build random piece sets
    player_pieces = build_side_pieces_with_priest(
        side="player",
        min_total_pieces=player_min_total,
        max_total_pieces=player_max_total
    )
    enemy_pieces = build_side_pieces_with_priest(
        side="enemy",
        min_total_pieces=enemy_min_total,
        max_total_pieces=enemy_max_total
    )

    all_pieces = player_pieces + enemy_pieces

    # ensure enough available hexes
    if len(all_pieces) > len(available_hexes):
        raise ValueError("Not enough unblocked hexes to place all pieces.")

    # 4) assign random unique positions
    assigned_positions = random.sample(available_hexes, len(all_pieces))
    for i, (qx, rx) in enumerate(assigned_positions):
        all_pieces[i]["q"] = qx
        all_pieces[i]["r"] = rx

    # final scenario
    scenario = {
        "name": "Puzzle Scenario",
        "subGridRadius": sub_grid_radius,
        "blockedHexes": blocked_hexes,
        "pieces": all_pieces
    }
    return scenario


############################################
# 3. Evaluate puzzle by RL self-play
############################################

def evaluate_puzzle(scenario, num_simulations=50):
    """
    Let the PPO model self-play `num_simulations` times. 
    Tally how often the player wins vs. enemy wins vs. draws.
    Returns a dict with that info and a puzzle_difficulty measure 
    (which you can define however you like).
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)

        final_data = env.current_episode[-1]
        final_side = final_data["turn_side"]
        final_rew = final_data["reward"]

        # standard "who won" logic:
        if final_rew >= 30:
            if final_side == "player":
                player_wins += 1
            else:
                enemy_wins += 1
        elif final_rew <= -30:
            # if side=player but rew <= -30 => enemy effectively won
            if final_side == "player":
                enemy_wins += 1
            else:
                player_wins += 1
        else:
            # typically -20 => time limit => draw
            draws += 1

    puzzle_difficulty = enemy_wins - player_wins
    return {
        "player_wins": player_wins,
        "enemy_wins": enemy_wins,
        "draws": draws,
        "puzzle_difficulty": puzzle_difficulty
    }


############################################
# 4. Putting it all together
############################################

def main():
    num_candidates = 10  # how many scenarios to generate
    num_simulations_per_puzzle = 1000

    # If the puzzle is too easy, skip it if player win rate is above this fraction
    max_player_win_fraction = 0.01

    candidate_puzzles = []

    for i in range(num_candidates):
        try:
            candidate = generate_candidate_puzzle(
                sub_grid_radius=None,
                num_blocked=None,
                player_min_total=2,
                player_max_total=5, 
                enemy_min_total=2,
                enemy_max_total=5
            )
        except ValueError as err:
            # e.g. "Not enough unblocked hexes"
            print(f"Candidate {i+1}: Skipping => {err}")
            continue

        # Evaluate
        evaluation = evaluate_puzzle(candidate, num_simulations=num_simulations_per_puzzle)
        pw = evaluation["player_wins"]
        ew = evaluation["enemy_wins"]
        dr = evaluation["draws"]
        diff = evaluation["puzzle_difficulty"]

        # Filter 1: If the puzzle is unwinnable (player never wins), skip
        if pw == 0:
            print(f"Candidate {i+1}: Player never wins => skipping.")
            continue

        # Filter 2: If puzzle is "too easy" (win rate above threshold), skip
        player_win_rate = pw / num_simulations_per_puzzle
        if player_win_rate > max_player_win_fraction:
            print(f"Candidate {i+1}: Player wins {player_win_rate:.2%} > {max_player_win_fraction:.2%}, skipping.")
            continue

        # Accept puzzle
        candidate["evaluation"] = evaluation
        candidate_puzzles.append(candidate)
        print(
            f"Candidate {i+1} => P-Wins:{pw}, E-Wins:{ew}, Draws:{dr}, Diff:{diff}, "
            f"PlayerWinRate:{player_win_rate:.2%}"
        )

    # Save results
    if candidate_puzzles:
        with open("data/generated_puzzles.yaml", "w") as f:
            yaml.dump(candidate_puzzles, f, sort_keys=False)
        print("Saved candidate puzzles to generated_puzzles.yaml")
    else:
        print("No valid puzzles generated under these constraints.")


if __name__ == "__main__":
    main()