"""
puzzle_generator.v2.py

An expanded, backward-based puzzle generator that creates "mate in N" style scenarios.
1) We define a final 'checkmate' state in which the enemy's Priest is certainly lost.
2) We do a BFS/DFS backward up to `mate_in_n` full moves (2 * mate_in_n half-moves).
3) We optionally prune states using the PPO model to see if the enemy can escape.

We then pick one of these initial states, format it like in the original puzzle scripts,
and save it to a YAML file (data/generated_puzzles_v2.yaml).

**Disclaimer**: The code still includes simplified stubs. For real usage, you'd need
comprehensive logic to enumerate moves, handle special abilities, blocked hexes, etc.
"""

import random
import copy
from collections import deque
import yaml
from sb3_contrib import MaskablePPO

from rl_training import HexPuzzleEnv

# Load your PPO model for optional pruning
model = MaskablePPO.load("ppo_model")


##########################################################
# 1. Basic utility for hashing states, checking if Priest is dead
##########################################################

def state_to_key(pieces, turn_side):
    """
    Convert a list of piece dicts + turn_side into a hashable key
    for BFS/DFS visited sets.
    """
    # Sort pieces by (side, class, q, r) ignoring any 'dead' pieces
    living = [p for p in pieces if not p.get("dead", False)]
    living_sorted = sorted(living, key=lambda p: (p["side"], p["class"], p["q"], p["r"]))
    piece_tuples = tuple(
        (p["side"], p["class"], p["q"], p["r"])
        for p in living_sorted
    )
    return (turn_side, piece_tuples)

def is_enemy_priest_dead(pieces):
    """
    Return True if there is no living 'Priest' on 'enemy' side.
    """
    for p in pieces:
        if p["side"] == "enemy" and p["class"] == "Priest" and not p.get("dead", False):
            return False
    return True


##########################################################
# 2. Core backward search
##########################################################

def backward_generate_mate_in_n(final_state, mate_in_n=2, use_ppo_pruning=False):
    """
    Attempt to generate puzzle states leading to `final_state` in `mate_in_n` full moves.

    final_state = (pieces_list, turn_side)
      - The side to move in final_state is presumably 'player' if the
        player's move is delivering checkmate, or it might be 'enemy' if
        you define your final scenario differently.

    We'll do BFS up to 2*mate_in_n half-moves of depth.
    We'll store each discovered predecessor in a queue along with how many
    half-moves we've gone so far.

    At each step, we:
      1) find all possible predecessors
      2) check the "forced mate" property, optionally using the PPO model if `use_ppo_pruning` is True

    Return a list of possible starting states that guarantee arrival at 'final_state'.
    """

    # max_depth in half-moves
    max_depth = 2 * mate_in_n

    visited = set()
    queue = deque()
    queue.append((final_state, 0))  # ( (pieces, turn_side), half_move_depth )
    results = []

    while queue:
        (current_state, depth) = queue.popleft()
        (pieces, turn_side) = current_state

        skey = state_to_key(pieces, turn_side)
        if skey in visited:
            continue
        visited.add(skey)

        # If we've reached the BFS frontier => record this as a potential "initial" puzzle
        if depth >= max_depth:
            results.append(current_state)
            continue

        # Attempt to find all possible predecessor states that lead to 'current_state' in one half-move
        # i.e. invert the environment's step logic. We'll call a stub function below.
        predecessors = find_predecessor_states(current_state)

        for prev_state in predecessors:
            # check if from prev_state, the next side can't avoid going into current_state or an equally losing line
            # if use_ppo_pruning is True, we ask the PPO model to see if there's a possible move that escapes
            if check_forced_mate_property(prev_state, current_state, use_ppo_pruning):
                # That means it's forced => we accept
                prev_depth = depth + 1
                queue.append((prev_state, prev_depth))

    return results


def find_predecessor_states(next_state):
    """
    Return a list of states (pieces, side_to_move) that, with one action, become 'next_state'.

    **STUB**: A real implementation enumerates all possible moves for side_to_move,
    applies them to a hypothetical 'prev_state', and checks if it yields `next_state`.
    That can be large. We'll just return an empty list here.
    """
    return []


def check_forced_mate_property(prev_state, next_state, use_ppo_pruning):
    """
    Return True if from 'prev_state' the side to move cannot avoid going to 'next_state'
    or an equally lost scenario. In other words, for every possible move the current side
    can do, they'd end up eventually forced to 'next_state'.

    If use_ppo_pruning is True, we'll do an approximate check:
      - We'll step forward from prev_state with the PPO controlling the side to move
      - If the PPO finds a path that avoids next_state or leads to a non-lost outcome,
        we consider it "not forced."
    **STUB** for demonstration only.
    """
    if not use_ppo_pruning:
        return True  # naive approach => always forced

    # approximate approach => do 1-step forward from prev_state with PPO
    # if the resulting state is not next_state or doesn't quickly lead to a losing line,
    # we declare it's not forced.

    # This is simplified. In real code you'd:
    # 1) Create env from prev_state
    # 2) Step once using PPO
    # 3) Compare resulting state to next_state
    # etc.
    # We'll just randomly return True/False half the time for demonstration
    return (random.random() < 0.5)


##########################################################
# 3. Building a final scenario & BFS
##########################################################

def build_final_state_example():
    """
    Build a minimal final scenario:
      - e.g. player's Warlock can kill the enemy Priest immediately, no escape.
    We'll say it's player's turn, so the final state's sideToMove = 'player'.
    """
    warlock = {
        "class": "Warlock",
        "label": "W",
        "color": "#556b2f",
        "side": "player",
        "q": 0,
        "r": 0,
        "dead": False
    }
    epriest = {
        "class": "Priest",
        "label": "P",
        "color": "#dc143c",
        "side": "enemy",
        "q": 1,
        "r": 0,
        "dead": False
    }
    pieces = [warlock, epriest]
    turn_side = "player"

    return (pieces, turn_side)


##########################################################
# 4. Forward-check / environment integration
##########################################################

def scenario_dict_from_state(state):
    """
    Convert a (pieces, turn_side) into a puzzle scenario dict
    that your environment can load. We'll do a simple subGridRadius=3
    with no blocked hexes for demonstration. You can adjust or randomize.
    """
    (pieces, turn_side) = state
    scenario = {
        "name": "MateInPuzzle",
        "subGridRadius": 3,
        "blockedHexes": [],
        "pieces": copy.deepcopy(pieces)
    }
    # We won't store 'turn_side' in the scenario directly. If your env
    # needs that, consider adding a 'turn_side' field. Or your env might
    # detect who moves first automatically (the player).
    return scenario


def main():
    # Let the user specify how many moves to mate
    mate_in_n = 3  # e.g. mate in 3 full moves
    use_ppo_pruning = True  # if True, we do approximate checks with PPO

    # 1) Build an example final "checkmate" scenario
    final_state = build_final_state_example()

    # 2) Do the backward BFS up to 2*mate_in_n half-moves
    initial_candidates = backward_generate_mate_in_n(
        final_state,
        mate_in_n=mate_in_n,
        use_ppo_pruning=use_ppo_pruning
    )

    if not initial_candidates:
        print("No states found that lead to final checkmate under these constraints.")
        return

    # For demonstration, let's just pick the first candidate
    chosen_state = initial_candidates[0]
    scenario = scenario_dict_from_state(chosen_state)

    # Now we can store that scenario to a YAML file,
    # in the same format you originally used:
    puzzle_data = [scenario]  # or you might store multiple

    output_file = "data/generated_puzzles_v2.yaml"
    with open(output_file, "w") as f:
        yaml.dump(puzzle_data, f, sort_keys=False)

    print(f"Done! Wrote {len(puzzle_data)} puzzle(s) to {output_file}.")


if __name__ == "__main__":
    main()