"""
puzzle_generator.v2.py

A 'non-conceptual' example that tries to:
  1) Generate many random small states (with Warlock, Priest, Guardian).
  2) Check if there's a forced "mate in 2" scenario (i.e., the player can guarantee
     killing the enemy Priest by the end of the player's second turn).
  3) If found, output one puzzle scenario to data/generated_puzzles_v2.yaml.

We do a forward-based brute force search for "mate in 2":
   - Depth = 4 half-moves: Player => Enemy => Player => Enemy.
   - If in all possible lines, the enemy Priest ends up dead by or before that 4th half-move,
     we say it's forced mate in 2.

This code is purely illustrative but should be able to produce a puzzle or two
if you run it enough times, given the small movement logic.

Requires:
  - Python 3.8+
  - pyyaml
"""

import random
import yaml
import copy
import math

# For convenience, we'll store the result in a single puzzle YAML.
OUTPUT_FILE = "data/generated_puzzles_v2.yaml"

############################################################
# 1) Basic environment constants
############################################################

# We'll fix subGridRadius = 2 or 3 for a small board
POSSIBLE_RADII = [2, 3]

# We'll define a very small set of classes: Warlock, Priest, Guardian.
# Each has minimal movement or attack rules for demonstration.
# In a real scenario, you'd integrate your own class logic or re-use rl_training.

CLASSES = ["Warlock", "Priest", "Guardian"]

# For each class, define a (move_range, can_attack_range).
# Warlock: can move 1, can kill an adjacent piece within 2. (Simplified)
# Priest: can move 1, no attack
# Guardian: can move 1, can kill adjacent piece (range=1)
CLASS_DATA = {
    "Warlock":   {"move_range": 1, "attack_range": 2},
    "Priest":    {"move_range": 1, "attack_range": 0},  # can't attack
    "Guardian":  {"move_range": 1, "attack_range": 1}
}

# We want each side to have exactly 2 or 3 pieces, including exactly 1 Priest.
PLAYER_COLOR = "#556b2f"
ENEMY_COLOR  = "#dc143c"

# We'll try up to this many random boards
MAX_RANDOM_BOARDS = 500

############################################################
# 2) Utility: axial distance, coordinate generation
############################################################

def hex_distance(q1, r1, q2, r2):
    """
    Standard axial distance on a hex grid: cube coords but we can do a quick formula
    """
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2

def all_hexes_in_radius(radius):
    """
    Return a list of all axial (q,r) within subGridRadius
    """
    coords = []
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            if abs(q+r) <= radius:
                coords.append((q, r))
    return coords

############################################################
# 3) Generating random puzzle states
############################################################

def generate_random_state():
    """
    1) Pick a subgrid radius from POSSIBLE_RADII
    2) Randomly block some hexes
    3) Build a few pieces for player side (1 Priest + 1-2 others from Warlock/Guardian),
       and similarly for enemy
    4) Assign them to distinct unblocked hexes
    """
    radius = random.choice(POSSIBLE_RADII)
    coords = all_hexes_in_radius(radius)
    random.shuffle(coords)

    # Possibly block up to radius hexes
    block_count = random.randint(0, radius)
    blocked = set(coords[:block_count])
    free_spots = coords[block_count:]

    # Build side pieces
    player_pieces = build_side_pieces("player")
    enemy_pieces  = build_side_pieces("enemy")
    all_pieces = player_pieces + enemy_pieces

    if len(all_pieces) > len(free_spots):
        # Not enough free spaces => fail
        raise ValueError("Not enough free spots to place pieces.")

    # Assign random positions from free_spots
    random.shuffle(free_spots)
    for i, piece in enumerate(all_pieces):
        piece["q"], piece["r"] = free_spots[i]

    # Build final puzzle state representation
    # We'll store sideToMove='player' at the beginning
    # We'll store (pieces, sideToMove, radius, blockedList).
    blocked_list = [{"q": q, "r": r} for (q, r) in blocked]
    state = {
        "pieces": all_pieces,
        "sideToMove": "player",
        "radius": radius,
        "blockedHexes": blocked_list
    }
    return state

def build_side_pieces(side):
    """
    Guarantee exactly 1 Priest, plus 1 or 2 from (Warlock, Guardian).
    """
    color = PLAYER_COLOR if side=="player" else ENEMY_COLOR
    # We'll do either 2 total pieces or 3 total pieces
    total_count = random.choice([2,3]) 
    # 1 piece is Priest
    pieces = [{
        "class": "Priest",
        "label": "P",
        "color": color,
        "side": side,
        "q": None,
        "r": None,
        "dead": False
    }]
    # The remainder is chosen from [Warlock, Guardian]
    other_classes = ["Warlock", "Guardian"]
    for _ in range(total_count - 1):
        c = random.choice(other_classes)
        label = c[0].upper()  # 'W' or 'G'
        pinfo = {
            "class": c,
            "label": label,
            "color": color,
            "side": side,
            "q": None,
            "r": None,
            "dead": False
        }
        pieces.append(pinfo)
    return pieces


############################################################
# 4) "Mate in 2" check via forward brute force
############################################################

def is_forced_mate_in_2(state):
    """
    Return True if from this state, the player can guarantee the enemy Priest
    is killed by the end of the player's second turn (4 half-moves):
       Move #1: Player
       Move #2: Enemy
       Move #3: Player
       Move #4: Enemy
    AND that the enemy cannot avoid losing the Priest.

    We'll do a small minimax-style search that enumerates all possible moves for each side.
    If we find any line in which the enemy Priest survives => NOT forced mate.
    If in all lines the Priest dies by or before the 4th half-move => forced mate.

    For performance, we keep track if the Priest is already dead earlier => no need to keep going.
    We also track if the player's own Priest is wiped out => might consider a draw or some outcome,
    but let's keep it simple: we only care about killing the enemy Priest in all lines.
    """
    # We'll do a recursive function that enumerates up to depth=4.
    # If at any time the enemy Priest is dead => that branch is good.
    # If we reach depth=4 and the enemy Priest is alive => that branch is a fail => not forced.

    # If there's ANY branch that results in the Priest surviving => not forced mate.
    # If in EVERY branch, the Priest is killed => forced mate.

    # We'll define a helper
    return check_line(state, depth=0)

def check_line(state, depth):
    # If enemy Priest is dead => success
    if priest_is_dead(state["pieces"], side="enemy"):
        return True  # branch is good

    if depth >= 4:
        # Reached the end => Priest not dead => fail
        return False

    side = state["sideToMove"]
    # gather all moves for side
    moves = enumerate_moves(state)
    if not moves:
        # no moves => we skip to next side, but is that a good or bad outcome?
        # If the priest isn't dead, we keep going but side can't move
        next_state = do_side_switch(state)
        return check_line(next_state, depth+1)

    # We do different logic if side == 'player' or side == 'enemy':
    # - If side == 'player', we want to see if there's ANY move that leads to forced mate
    #   (player only needs one winning line).
    # - If side == 'enemy', we want to see if they can find ANY move that escapes
    #   (enemy only needs one move that leads to the Priest surviving => entire scenario fails).

    if side == "player":
        # We require that the Player can pick at least ONE move such that
        # from that resulting state,  everything leads to a kill of the Priest eventually.
        # If ALL moves fail, we fail. But player can choose a good one => so we only need one success.
        success_found = False
        for m in moves:
            new_st = apply_move(state, m)
            # next half-move
            if check_line(new_st, depth+1):
                success_found = True
                break
        return success_found

    else:
        # side == 'enemy'
        # If the enemy can find ANY move that leads to the Priest surviving => scenario is not forced.
        # So if we find a single move => in that branch the Priest lives => return False for the entire scenario.
        for m in moves:
            new_st = apply_move(state, m)
            if check_line(new_st, depth+1):
                # That means in that line, the Priest was eventually killed => we keep going
                # Actually wait, we want to see if there's a line that leads to survival => if so, return False
                # So if check_line returned True => means that line kills the Priest => not what the enemy wants
                continue
            else:
                # We found a line where the Priest survives => forced mate fails
                return False
        # If we never found any line that leads to survival => means in all lines the Priest dies => forced
        return True

############################################################
# 5) Move enumeration and application
############################################################

def enumerate_moves(state):
    """
    Return a list of possible moves for the sideToMove in the given state.
    We'll store moves as (piece_label, "move", target_q, target_r) or
    (piece_label, "attack", target_q, target_r).
    Simplified ignoring blocked hexes or advanced lines-of-sight, etc.
    """
    side = state["sideToMove"]
    pieces = state["pieces"]
    radius = state["radius"]
    blocked = {(b["q"], b["r"]) for b in state["blockedHexes"]}

    # collect living pieces for this side
    side_pieces = [p for p in pieces if p["side"]==side and not p["dead"]]
    moves = []
    for sp in side_pieces:
        cdata = CLASS_DATA[sp["class"]]
        # move_range = cdata["move_range"]
        # attack_range = cdata["attack_range"]

        # We do a minimal approach: you can move to an adjacent hex or attack an enemy piece in range.
        # 1) possible move
        all_hex_neighbors = hex_neighbors(sp["q"], sp["r"])
        for (nq, nr) in all_hex_neighbors:
            if (nq,nr) not in blocked and not piece_at(pieces, nq, nr):
                # a valid move
                moves.append((sp["label"], "move", nq, nr))

        # 2) possible attack
        # We'll allow an attack if distance <= attack_range and there's an enemy piece there
        attack_range = cdata["attack_range"]
        # gather all enemy positions that are within 'attack_range'
        for e in pieces:
            if e["side"] != side and not e["dead"]:
                dist = hex_distance(sp["q"], sp["r"], e["q"], e["r"])
                if dist <= attack_range:
                    moves.append((sp["label"], "attack", e["q"], e["r"]))

    return moves

def apply_move(state, move):
    """
    Given a state, apply one move. Return a new state with side switched.
    move = (piece_label, action, tq, tr)
    """
    piece_label, action, tq, tr = move
    new_state = copy.deepcopy(state)
    # find the piece
    side = new_state["sideToMove"]
    pieces = new_state["pieces"]
    p = next((p for p in pieces if p["label"]==piece_label and p["side"]==side and not p["dead"]), None)
    if not p:
        # invalid move => just skip?
        return do_side_switch(new_state)
    
    if action=="move":
        p["q"], p["r"] = tq, tr
    elif action=="attack":
        # kill the piece at (tq,tr) if found
        vic = next((x for x in pieces if x["q"]==tq and x["r"]==tr and x["side"]!=side and not x["dead"]), None)
        if vic:
            vic["dead"] = True
    # switch side
    return do_side_switch(new_state)

def do_side_switch(state):
    new_st = copy.deepcopy(state)
    if state["sideToMove"]=="player":
        new_st["sideToMove"] = "enemy"
    else:
        new_st["sideToMove"] = "player"
    return new_st

def hex_neighbors(q, r):
    """Return up to 6 neighbors (pointy top axial)."""
    return [
        (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
    ]

def piece_at(pieces, q, r):
    return next((p for p in pieces if p["q"]==q and p["r"]==r and not p["dead"]), None)

def priest_is_dead(pieces, side):
    """
    Check if the Priest on the given side is dead or not present.
    """
    for p in pieces:
        if p["side"]==side and p["class"]=="Priest" and not p["dead"]:
            return False
    return True

############################################################
# 6) Main script
############################################################

def main():
    found_any = False
    for attempt in range(MAX_RANDOM_BOARDS):
        try:
            state = generate_random_state()
        except ValueError:
            # not enough free spots => skip
            continue

        # Now check if it's forced mate in 2
        if is_forced_mate_in_2(state):
            # We found a puzzle
            puzzle_scenario = build_yaml_scenario(state)
            with open(OUTPUT_FILE, "w") as f:
                yaml.dump([puzzle_scenario], f, sort_keys=False)
            print(f"Success on attempt {attempt+1} => wrote puzzle to {OUTPUT_FILE}:\n{puzzle_scenario}")
            found_any = True
            break

    if not found_any:
        print(f"No forced mate in 2 puzzle found after {MAX_RANDOM_BOARDS} attempts.")


def build_yaml_scenario(state):
    """
    Convert our internal 'state' into the format used by your puzzle scripts.
    {
      name: "Puzzle Scenario",
      subGridRadius: int,
      blockedHexes: list of {q:..., r:...},
      pieces: [ {class, label, color, side, q, r}, ... ],
    }
    """
    puzzle = {
        "name": "MateIn2Puzzle",
        "subGridRadius": state["radius"],
        "blockedHexes": state["blockedHexes"],
        "pieces": []
    }
    # copy pieces
    for p in state["pieces"]:
        if not p["dead"]:
            puzzle["pieces"].append({
                "class": p["class"],
                "label": p["label"],
                "color": p["color"],
                "side": p["side"],
                "q": p["q"],
                "r": p["r"]
            })
    return puzzle

if __name__=="__main__":
    main()