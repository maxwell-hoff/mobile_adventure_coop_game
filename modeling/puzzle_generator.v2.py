"""
puzzle_generator.v2.py

- Generates random puzzle scenarios subject to user-specified CLI arguments.
- Ensures each side has at least 1 Priest. Enemy side can also have BloodWarden, but not player side.
- Uses pieces.yaml for classes and abilities (range, requires_los, etc.).
- Each turn, *every living piece* on the active side picks exactly 1 action (including "pass").
- We apply those actions in ascending label order to form the new state for that turn.
- We do a forward-based brute force to see if it's forced "mate in 2" (4 half-turns):
  - If in *any* line the enemy Priest survives, it's not forced mate.
  - If in *all* lines the Priest is killed, we have forced mate.
  - Then we also check how many distinct winning lines the player's side has:
    * If more than 1 => we discard the puzzle (i.e., too many solutions).
    * If exactly 1 => we accept it.

**What's simplified?**
1) We do not handle advanced multi-target or partial-turn synergy in depth. 
   Each piece picks exactly 1 action from pieces.yaml (like "move," "single_target_attack," etc.), 
   but we do not re-check after each piece's move for new blockages or position changes within the same turn. 
2) We assume the turn's actions apply in a fixed label-sorted order, ignoring some real-time intricacies. 
3) We do not handle cast_speed, multi-target combos, or advanced synergy in a fully robust manner. 
   This code is purely an illustrative skeleton, not a production-grade solution.

Usage (example):
  python puzzle_generator.v2.py --randomize-radius --radius-min 2 --radius-max 5 
                                --randomize-blocked --min-blocked 1 --max-blocked 5
                                --randomize-pieces
                                --player-min-pieces 3 --player-max-pieces 4 
                                --enemy-min-pieces 3 --enemy-max-pieces 5
"""

import argparse
import random
import yaml
import copy
import math
import sys
from itertools import combinations, product

with open("data/pieces.yaml", "r", encoding="utf-8") as f:
    pieces_data = yaml.safe_load(f)

OUTPUT_FILE = "data/generated_puzzles_v2.yaml"

################################################################################
# 1) Parse CLI
################################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hex Puzzle Generator v2 - Mate in 2 (multi-piece turn).")
    parser.add_argument("--randomize", action="store_true", help="Randomize piece positions each reset.")
    parser.add_argument("--approach", choices=["ppo", "tree", "mcts"], default="ppo", 
                        help="(Not fully used, included for compatibility with rl_training.)")
    parser.add_argument("--randomize-radius", action="store_true", help="Randomize puzzle radius.")
    parser.add_argument("--radius-min", type=int, default=2)
    parser.add_argument("--radius-max", type=int, default=5)
    parser.add_argument("--randomize-blocked", action="store_true", help="Randomize blocked hexes.")
    parser.add_argument("--min-blocked", type=int, default=1)
    parser.add_argument("--max-blocked", type=int, default=5)
    parser.add_argument("--randomize-pieces", action="store_true", 
                        help="Randomize piece composition for each side.")
    parser.add_argument("--player-min-pieces", type=int, default=3)
    parser.add_argument("--player-max-pieces", type=int, default=4)
    parser.add_argument("--enemy-min-pieces", type=int, default=3)
    parser.add_argument("--enemy-max-pieces", type=int, default=5)
    return parser.parse_args()

################################################################################
# 2) Scenario generation (similar to before)
################################################################################

def hex_distance(q1, r1, q2, r2):
    return (abs(q1-q2) + abs(r1-r2) + abs((q1+r1)-(q2+r2)))//2

def all_hexes_in_radius(radius):
    coords=[]
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            if abs(q+r)<=radius:
                coords.append((q,r))
    return coords

def generate_random_scenario(args):
    # pick radius
    if args.randomize_radius:
        radius = random.randint(args.radius_min, args.radius_max)
    else:
        radius = random.choice([3,4])

    all_coords = all_hexes_in_radius(radius)
    random.shuffle(all_coords)

    if args.randomize_blocked:
        block_count = random.randint(args.min_blocked, min(args.max_blocked, len(all_coords)))
    else:
        block_count = 0

    blocked = set(all_coords[:block_count])
    free_spots = all_coords[block_count:]
    blocked_hexes = [{"q":q, "r":r} for (q,r) in blocked]

    player_pieces = build_side_pieces("player", args.player_min_pieces, args.player_max_pieces)
    enemy_pieces  = build_side_pieces("enemy",  args.enemy_min_pieces, args.enemy_max_pieces)
    all_pieces = player_pieces + enemy_pieces

    if len(all_pieces)>len(free_spots):
        raise ValueError("Not enough free spots to place pieces on board.")

    random.shuffle(free_spots)
    for i, pc in enumerate(all_pieces):
        q,r = free_spots[i]
        pc["q"], pc["r"] = q, r

    scenario = {
        "name": "Puzzle Scenario",
        "subGridRadius": radius,
        "blockedHexes": blocked_hexes,
        "pieces": all_pieces
    }
    return scenario

def build_side_pieces(side, min_total, max_total):
    color = "#556b2f" if side=="player" else "#dc143c"
    total_count = random.randint(min_total, max_total)
    # Always 1 Priest
    pieces = [{
        "class": "Priest",
        "label": "P",
        "color": color,
        "side": side,
        "q": None,
        "r": None,
        "dead": False
    }]
    valid_classes = list(pieces_data["classes"].keys())
    if side=="player":
        valid_classes = [c for c in valid_classes if c!="BloodWarden"]
    valid_classes = [c for c in valid_classes if c!="Priest"]

    needed_others = total_count-1
    for _ in range(needed_others):
        c = random.choice(valid_classes)
        label = pieces_data["classes"][c].get("label", c[0].upper()) or c[0].upper()
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

################################################################################
# 3) The "mate in 2" check with multi-piece turns & "unique solution" check
################################################################################

def is_forced_mate_in_2(scenario):
    """
    We'll do a forward-based brute force for up to 4 half-turns:
      P-turn #1 => E-turn #1 => P-turn #2 => E-turn #2
    Now each 'turn' means *all living pieces* on that side choose exactly 1 action
    from the enumerations, combined into one multi-action set, applied in ascending label order.

    If any line => enemy Priest alive => not forced
    If all lines => Priest is dead => forced
    Then we count how many distinct lines lead to a kill. If >1 => we discard.
    We'll do this by enumerating all lines, see if *all* lines kill the Priest => forced
    and also track how many lines (the player's lines) are "winning."

    Implementation detail: We'll gather *all possible player-turn combos* for the player's turn,
    and *all possible enemy-turn combos* for the enemy's turn, recursively.

    Then "unique solution" => means there's exactly 1 distinct set of *player-turn combos*
    that kills the Priest, with no alternative path for the player side. 
    We'll do a small approach: we track the number of distinct winning lines for the player's
    first move. If that is >1 => discard. Similarly for the player's second move.
    """
    state = {
        "pieces": copy.deepcopy(scenario["pieces"]),
        "blockedHexes": copy.deepcopy(scenario["blockedHexes"]),
        "radius": scenario["subGridRadius"],
        "sideToMove": "player"
    }
    # Clear any dead flags
    for p in state["pieces"]:
        p["dead"] = False

    # We'll gather *all lines* up to depth=4.
    # We'll keep track if *any* line => Priest lives => not forced
    # We'll also track how many distinct "player solutions" exist. If >1 => discard puzzle.

    # We'll store a structure: "lines" = list of (playerCombosUsed, finalState, priestDead)
    lines = []
    enumerate_lines(state, depth=0, partialPlayerMoves=[], lines=lines)

    # If we find any line with priestDead=False => not forced => return False immediately
    if any(line["priestDead"]==False for line in lines):
        return False  # means scenario not forced

    if not lines:
        return False  # no lines => weird => not forced

    # Now all lines => PriestDead => forced. Next we check how many distinct ways the player had.
    # We'll look at the unique sets of "partialPlayerMoves" among lines that lead to a kill.
    # If that is more than 1 => discard => not "unique solution."

    # "partialPlayerMoves" is a list: [ (turn=1, comboUsed), (turn=3, comboUsed) ] i.e. the player's half-turn indices.
    # We'll group lines by (comboUsed on player's first turn, comboUsed on player's second turn).
    winning_combos = set()
    for line in lines:
        # line["partialPlayerMoves"] might be something like:
        # [ { 'turn': 0, 'combo': 'some representation' }, { 'turn': 2, 'combo': 'some other representation' } ]
        # We'll produce a tuple of the combos for both player turns:
        # If the player only acted on turn 0,2 we gather that. 
        # We'll do a small approach: sorted by turn
        pm_sorted = sorted(line["partialPlayerMoves"], key=lambda x: x["turn"])
        combo_list = tuple(item["combo"] for item in pm_sorted)
        winning_combos.add(combo_list)

    # If we have more than 1 distinct combo list => multiple solutions => discard => return False
    if len(winning_combos)>1:
        return False

    return True  # exactly 1 solution, forced in all lines => success

def enumerate_lines(state, depth, partialPlayerMoves, lines):
    """
    Recursively enumerate all lines up to depth=4 half-turns.
    For each turn:
      - gather combos for the current side => each piece picks 1 action
      - for each combo => apply in label order => produce nextState => recursively continue
    If at any point we have depth=4 or the Priest is dead, we store the result in lines[].
    """
    # If enemy Priest is already dead
    if is_priest_dead(state["pieces"], "enemy"):
        lines.append({
            "partialPlayerMoves": copy.deepcopy(partialPlayerMoves),
            "priestDead": True
        })
        return

    if depth>=4:
        # we ended => Priest not dead => store line
        lines.append({
            "partialPlayerMoves": copy.deepcopy(partialPlayerMoves),
            "priestDead": False
        })
        return

    side = state["sideToMove"]
    combos = build_all_turn_combos(state, side)

    if not combos:
        # no combos => skip side
        newState = switch_side(state)
        enumerate_lines(newState, depth+1, partialPlayerMoves, lines)
        return

    for combo in combos:
        # Apply them in ascending label order
        st2 = apply_turn_combo(state, combo)
        # If side == "player", record which combo we used
        # We'll store a small string representation for "combo"
        # e.g. "W=move(1,0),P=attack(2,2)..." etc.
        usedComboStr = combo_to_string(combo)

        newPartial = partialPlayerMoves
        if side=="player":
            newPartial = copy.deepcopy(partialPlayerMoves)
            newPartial.append({
                "turn": depth,
                "combo": usedComboStr
            })

        enumerate_lines(st2, depth+1, newPartial, lines)

def build_all_turn_combos(state, side):
    """
    Build a list of all possible "turn combos" for the given side:
      - Each living piece picks exactly 1 action from enumerate_piece_actions(...)
      - We collect them into a single set: { piece_label -> chosen_move }.
    We'll do a cartesian product of each piece's possible moves.
    If e.g. side has 2 pieces => each has maybe 3 moves => total combos=3*3=9.

    We skip combos that are obviously contradictory (two pieces moving to the same hex?), 
    though the code below is still simplified: it doesn't re-check if piece A's move blocks piece B.
    We just let them do that and rely on the final apply in label order to see the net effect.
    For large # of pieces, this can blow up combinatorially.
    """
    side_pieces = [p for p in state["pieces"] if p["side"]==side and not p["dead"]]
    if not side_pieces:
        return []  # no combos

    # For each piece, gather possible "single" moves
    all_piece_moves = []
    for piece in side_pieces:
        pmoves = gather_piece_actions(state, piece)
        if not pmoves:
            # if no moves => we add a 'pass'
            pmoves = [(piece["label"], "pass", 0, 0)]
        all_piece_moves.append(pmoves)

    # Now do a cartesian product
    combos = product(*all_piece_moves)
    # combos is an iterator of tuples. Each tuple = (move_for_piece1, move_for_piece2, ...)

    # We'll store them as a list of combos, each combo is e.g. [ (label, atype, x, y), ... ]
    final_combos = []
    for ctuple in combos:
        final_combos.append(list(ctuple))
    return final_combos

def gather_piece_actions(state, piece):
    """
    Return a list of possible single actions for this piece (excluding "pass" because we add it in build_all_turn_combos).
    This is basically the same logic as 'enumerate_moves' but only for one piece. 
    """
    side = piece["side"]
    pieces = state["pieces"]
    blocked_hexes = {(b["q"], b["r"]) for b in state["blockedHexes"]}
    radius = state["radius"]

    actions=[]
    cls = piece["class"]
    if cls not in pieces_data["classes"]:
        return actions

    c_actions = pieces_data["classes"][cls]["actions"]
    # move
    if "move" in c_actions:
        rng = c_actions["move"].get("range",1)
        for (q,r) in all_hexes_in_radius(radius):
            dist = hex_distance(piece["q"], piece["r"], q, r)
            if dist<=rng and not is_occupied_or_blocked(q,r, pieces, blocked_hexes):
                if (q,r)!=(piece["q"], piece["r"]):
                    actions.append((piece["label"], "move", q, r))

    # single-target / multi-target / aoe / swap
    # We'll do a simpler approach: basically the same snippet from 'enumerate_moves' but restricted to one piece
    for aname, adata in c_actions.items():
        if aname=="move": 
            continue
        if "action_type" not in adata:
            continue
        atype = adata["action_type"]
        rng   = adata.get("range",0)
        requires_los = adata.get("requires_los",False)
        ally_only    = adata.get("ally_only",False)
        rad_aoe      = adata.get("radius",0)

        if atype=="single_target_attack":
            enemies = [e for e in pieces if e["side"]!=side and not e.get("dead",False)]
            for e in enemies:
                dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                if dist<=rng:
                    if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_hexes, pieces):
                        actions.append((piece["label"], "single_target_attack", e["q"], e["r"]))

        elif atype=="multi_target_attack":
            # skipping full combos
            pass

        elif atype=="aoe":
            # if there's at least 1 enemy in range, we allow an "aoe" action
            enemies_in_range=[]
            for e in pieces:
                if e["side"]!=side and not e["dead"]:
                    d = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if d<=rad_aoe:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_hexes, pieces):
                            enemies_in_range.append(e)
            if enemies_in_range:
                actions.append((piece["label"], "aoe", 0, 0))

        elif atype=="swap_position":
            # ...
            # We'll do the same approach
            if ally_only:
                possible_targets=[pp for pp in pieces if pp["side"]==side and not pp["dead"] and pp!=piece]
            else:
                possible_targets=[pp for pp in pieces if not pp["dead"] and pp!=piece]
            for t in possible_targets:
                dist=hex_distance(piece["q"], piece["r"], t["q"], t["r"])
                if dist<=rng:
                    if (not requires_los) or line_of_sight(piece["q"], piece["r"], t["q"], t["r"], blocked_hexes, pieces):
                        actions.append((piece["label"], "swap_position", t["q"], t["r"]))

    return actions

def apply_turn_combo(state, combo):
    """
    We have a list of moves: [ (label, atype, x, y), (label2, atype2, x2,y2), ... ]
    We apply them in ascending piece label order, to produce a new state.
    """
    new_st = copy.deepcopy(state)
    side = new_st["sideToMove"]
    # Sort combo by label to fix the application order
    sorted_combo = sorted(combo, key=lambda mv: mv[0])

    for move in sorted_combo:
        piece_label, atype, tq, tr = move
        apply_single_move(new_st, piece_label, atype, tq, tr)

    # now switch side
    new_st["sideToMove"] = "enemy" if side=="player" else "player"
    return new_st

def apply_single_move(state, piece_label, atype, tq, tr):
    """
    Like apply_move, but for a single piece in the context of a multi-move turn.
    We do not re-check occupancy or adjacency after each piece's move.
    """
    side = state["sideToMove"]
    piece = next((p for p in state["pieces"] if p["label"]==piece_label and p["side"]==side and not p["dead"]), None)
    if not piece:
        return

    if atype=="move":
        piece["q"], piece["r"] = tq, tr
    elif atype=="pass":
        pass
    elif atype=="single_target_attack":
        victim = next((v for v in state["pieces"] if v["q"]==tq and v["r"]==tr and v["side"]!=side and not v["dead"]), None)
        if victim:
            victim["dead"]=True
    elif atype=="aoe":
        cdata = pieces_data["classes"][piece["class"]]["actions"]
        for aname, adesc in cdata.items():
            if adesc.get("action_type","")=="aoe":
                rad = adesc.get("radius",1)
                enemies = [e for e in state["pieces"] if e["side"]!=side and not e["dead"]]
                for e in enemies:
                    dist=hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist<=rad:
                        e["dead"]=True
    elif atype=="swap_position":
        occupant = next((v for v in state["pieces"] if v["q"]==tq and v["r"]==tr and not v["dead"]), None)
        if occupant:
            old_q, old_r = piece["q"], piece["r"]
            piece["q"], piece["r"] = occupant["q"], occupant["r"]
            occupant["q"], occupant["r"] = old_q, old_r

def switch_side(state):
    new_st = copy.deepcopy(state)
    if new_st["sideToMove"]=="player":
        new_st["sideToMove"]="enemy"
    else:
        new_st["sideToMove"]="player"
    return new_st

def is_priest_dead(pieces, side="enemy"):
    for p in pieces:
        if p["side"]==side and p["class"]=="Priest" and not p.get("dead",False):
            return False
    return True

def combo_to_string(combo):
    """
    Convert a list of moves for one turn 
    e.g. [("P","move",1,0),("W","pass",0,0)] => "P=move(1,0),W=pass"
    """
    parts=[]
    for (label,atype,tq,tr) in combo:
        if atype=="move":
            parts.append(f"{label}=move({tq},{tr})")
        elif atype=="pass":
            parts.append(f"{label}=pass")
        elif atype=="single_target_attack":
            parts.append(f"{label}=atk({tq},{tr})")
        elif atype=="aoe":
            parts.append(f"{label}=aoe")
        elif atype=="swap_position":
            parts.append(f"{label}=swap({tq},{tr})")
        else:
            parts.append(f"{label}={atype}({tq},{tr})")
    return "|".join(parts)

################################################################################
# 6) Main driver: generate random scenarios, check forced mate, check uniqueness
################################################################################

def main():
    args = parse_arguments()
    MAX_TRIES=2000
    found_any=False

    for attempt in range(MAX_TRIES):
        try:
            scenario=generate_random_scenario(args)
        except ValueError:
            continue

        # Check forced mate in 2 w single-solution
        if is_forced_mate_in_2(scenario):
            puzzle_list=[scenario]
            with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
                yaml.dump(puzzle_list,f,sort_keys=False)
            print(f"Success on attempt {attempt+1} => wrote puzzle to {OUTPUT_FILE}.")
            print(scenario)
            found_any=True
            break

    if not found_any:
        print(f"No forced mate-in-2 puzzle (unique solution) found after {MAX_TRIES} attempts.")


if __name__=="__main__":
    main()
