"""
puzzle_generator.v2.py

An extended puzzle generator that tries to handle:
  - Per-turn BFS of partial moves (i.e. piece-by-piece synergy) 
    so if piece A's move changes the board, piece B sees that new board during the same turn.
  - Multi-target attacks for classes that have them.
  - Basic "cast speed" or delayed effect for something like BloodWarden (though simplified).
  - We do a forward search up to 4 half-turns ("mate in 2"): Player->Enemy->Player->Enemy.
  - If every line kills the enemy Priest => forced mate.
  - If the player has more than 1 distinct winning route => discard puzzle.

**WARNING**: This code can easily blow up in branching factor for large # of pieces.
It's purely illustrative, not production-ready for big boards.

Usage:
  python puzzle_generator.v2.py --randomize-radius --radius-min=2 --radius-max=5 ...
"""

import argparse
import random
import yaml
import copy
import math
import sys
from itertools import combinations, permutations


with open("data/pieces.yaml","r", encoding="utf-8") as f:
    pieces_data = yaml.safe_load(f)

OUTPUT_FILE = "data/generated_puzzles_v2.yaml"

################################################################################
# 1) Parse arguments
################################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hex puzzle generator v2 with partial-turn BFS.")
    parser.add_argument("--randomize", action="store_true", help="Randomize piece positions each reset.")
    parser.add_argument("--approach", choices=["ppo","tree","mcts"], default="ppo",
                        help="(Included for consistency; not fully used here.)")
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
# 2) Basic helpers: distance, LOS, occupancy
################################################################################

def hex_distance(q1,r1,q2,r2):
    return (abs(q1-q2)+abs(r1-r2)+abs((q1+r1)-(q2+r2)))//2

def all_hexes_in_radius(radius):
    coords=[]
    for q in range(-radius,radius+1):
        for r in range(-radius,radius+1):
            if abs(q+r)<=radius:
                coords.append((q,r))
    return coords

def is_occupied_or_blocked(q,r,pieces, blocked_set):
    if (q,r) in blocked_set:
        return True
    for p in pieces:
        if not p.get("dead",False) and (p["q"],p["r"])==(q,r):
            return True
    return False

def line_of_sight(q1,r1,q2,r2, blocked_set, all_pieces):
    if (q1==q2) and (r1==r2):
        return True
    N = max(abs(q2-q1), abs(r2-r1), abs((q1+r1)-(q2+r2)))
    if N==0:
        return True
    s1=-q1-r1
    s2=-q2-r2
    line_hexes=[]
    for i in range(N+1):
        t = i/N
        qf=q1+(q2-q1)*t
        rf=r1+(r2-r1)*t
        sf=s1+(s2-s1)*t
        rq=round(qf)
        rr=round(rf)
        rs=round(sf)
        # fix rounding
        qdiff=abs(rq-qf)
        rdiff=abs(rr-rf)
        sdiff=abs(rs-sf)
        if qdiff>rdiff and qdiff>sdiff:
            rq=-rr-rs
        elif rdiff>sdiff:
            rr=-rq-rs
        line_hexes.append((rq,rr))
    # skip first,last
    for (hq,hr) in line_hexes[1:-1]:
        if (hq,hr) in blocked_set:
            return False
        occupant= next((pp for pp in all_pieces if not pp.get("dead",False) and (pp["q"],pp["r"])==(hq,hr)), None)
        if occupant:
            return False
    return True

def is_priest_dead(pieces, side="enemy"):
    for p in pieces:
        if p["side"]==side and p["class"]=="Priest" and not p.get("dead",False):
            return False
    return True

################################################################################
# 3) Scenario generation: at least 1 Priest per side, BloodWarden only for enemy
################################################################################

def generate_random_scenario(args):
    # pick radius
    if args.randomize_radius:
        radius = random.randint(args.radius_min, args.radius_max)
    else:
        radius = random.choice([3,4])

    coords = all_hexes_in_radius(radius)
    random.shuffle(coords)

    block_count=0
    if args.randomize_blocked:
        block_count= random.randint(args.min_blocked, min(args.max_blocked, len(coords)))
    blocked = set(coords[:block_count])
    free_spots = coords[block_count:]
    blocked_hexes = [{"q":q,"r":r} for (q,r) in blocked]

    player_pieces = build_side_pieces("player", args.player_min_pieces, args.player_max_pieces)
    enemy_pieces  = build_side_pieces("enemy",  args.enemy_min_pieces,  args.enemy_max_pieces)
    all_pieces = player_pieces+enemy_pieces

    if len(all_pieces)>len(free_spots):
        raise ValueError("Not enough free spots to place pieces.")

    random.shuffle(free_spots)
    for i,pc in enumerate(all_pieces):
        pc["q"], pc["r"] = free_spots[i]

    scenario={
        "name":"Puzzle Scenario",
        "subGridRadius": radius,
        "blockedHexes": blocked_hexes,
        "pieces": all_pieces
    }
    return scenario

def build_side_pieces(side, min_total, max_total):
    color = "#556b2f" if side=="player" else "#dc143c"
    count = random.randint(min_total, max_total)
    # always 1 Priest
    pieces=[{
        "class":"Priest",
        "label":"P",
        "color": color,
        "side": side,
        "q":None,
        "r":None,
        "dead":False
    }]
    valid_classes = list(pieces_data["classes"].keys())
    if side=="player":
        valid_classes = [c for c in valid_classes if c!="BloodWarden"]
    valid_classes = [c for c in valid_classes if c!="Priest"]

    needed = count-1
    for _ in range(needed):
        c = random.choice(valid_classes)
        lbl = pieces_data["classes"][c].get("label", c[0].upper()) or c[0].upper()
        piece={
            "class": c,
            "label": lbl,
            "color": color,
            "side": side,
            "q":None,"r":None,
            "dead":False
        }
        pieces.append(piece)
    return pieces

################################################################################
# 4) The multi-step BFS for each half-turn
################################################################################

def is_forced_mate_in_2(scenario):
    """
    We do 4 half-turns: depth=0..3. 
    Each half-turn => BFS over partial moves for each living piece in all permutations 
    (piece A moves, then piece B moves, etc.). This addresses partial-turn synergy.

    If in all final lines the enemy's Priest is dead => forced mate. 
    Then we check how many distinct winning combos the player has => if >1 => discard puzzle => return False
    """

    state = {
        "pieces": copy.deepcopy(scenario["pieces"]),
        "blockedHexes": {(bh["q"], bh["r"]) for bh in scenario["blockedHexes"]},
        "radius": scenario["subGridRadius"],
        "sideToMove": "player"
    }
    for p in state["pieces"]:
        p["dead"]=False

    lines = []
    enumerate_lines(state, depth=0, partialPlayerCombos=[], lines=lines)

    # if any line => priestDead==False => not forced
    if any(l["priestDead"]==False for l in lines):
        return False
    if not lines:
        return False

    # forced => must check uniqueness
    # gather distinct sets of (player combos on turn 0, turn 2) among lines
    winning_combos=set()
    for l in lines:
        # l["partialPlayerCombos"] => e.g. [ { "turn":0, "comboStr":"(some moves)"}, { "turn":2, "comboStr":"(some moves)"}]
        # we only care about turn=0 and turn=2 combos
        c0 = next((x for x in l["partialPlayerCombos"] if x["turn"]==0), None)
        c2 = next((x for x in l["partialPlayerCombos"] if x["turn"]==2), None)
        pair=( c0["comboStr"] if c0 else "", c2["comboStr"] if c2 else "" )
        winning_combos.add(pair)
    if len(winning_combos)>1:
        return False

    return True

def enumerate_lines(state, depth, partialPlayerCombos, lines):
    """
    BFS for depth up to 4. If enemy Priest is dead => store line.
    Otherwise, build all possible *full-turn sequences* for the side's pieces in permutations.
    Each piece acts one at a time, in any order.
    Then apply the resulting final board => next depth => recursive.

    partialPlayerCombos => track which combos the player used at turn=0 or 2 for uniqueness checking.
    """
    if is_priest_dead(state["pieces"], "enemy"):
        lines.append({
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": True
        })
        return

    if depth>=4:
        lines.append({
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": False
        })
        return

    side = state["sideToMove"]
    living = [p for p in state["pieces"] if p["side"]==side and not p.get("dead",False)]
    if not living:
        # no pieces => skip turn
        st2 = switch_side(state)
        enumerate_lines(st2, depth+1, partialPlayerCombos, lines)
        return

    # gather all permutations of living pieces => for each piece, gather possible single actions => BFS
    # e.g. if we have 2 living pieces => permutations are [ (pA, pB), (pB, pA) ]
    # we then do partial BFS: first piece picks an action => apply => second piece picks action => apply => ...
    all_perms= permutations(living, r=len(living))
    for perm in all_perms:
        # for each permutation, we do a BFS over "index in perm"
        # carrying along a state as we apply each piece's move.
        partial_results = []
        initial_state = copy.deepcopy(state)
        partial_BFS_states = [(initial_state, [])]  # list of (currentState, listOfMovesUsed)
        for piece in perm:
            new_partial = []
            for (stSoFar, movesUsedSoFar) in partial_BFS_states:
                piece_actions = gather_single_piece_actions(stSoFar, piece)
                if not piece_actions:
                    # means piece can do no action => pass
                    newState = apply_single_action(stSoFar, piece, ("pass",0,0))
                    newMoves = movesUsedSoFar + [f"{piece['label']}=pass"]
                    new_partial.append((newState, newMoves))
                else:
                    for (atype,tq,tr) in piece_actions:
                        st3 = apply_single_action(stSoFar, piece, (atype,tq,tr))
                        moveStr = move_to_string(piece, atype, tq, tr)
                        newMoves = movesUsedSoFar + [moveStr]
                        new_partial.append((st3, newMoves))
            partial_BFS_states = new_partial

        # after we finish that BFS => we have a set of final states for the turn
        for (finalTurnState, moveList) in partial_BFS_states:
            # Now we switch side => produce next state => recursion
            st2 = switch_side(finalTurnState)

            # if side=player => store the combo in partial combos
            # we'll build a single string for the entire sequence
            comboStr = ";".join(moveList)
            newPartialPC = partialPlayerCombos
            if side=="player":
                newPartialPC = copy.deepcopy(partialPlayerCombos)
                newPartialPC.append({
                    "turn": depth,
                    "comboStr": comboStr
                })
            enumerate_lines(st2, depth+1, newPartialPC, lines)

def gather_single_piece_actions(state, piece):
    """
    Single piece's possible moves given the current state. 
    We exclude "pass" because we'll handle that if no actions exist. 
    This is like the old enumerate, but for only one piece.
    """
    side=piece["side"]
    blocked_set = state["blockedHexes"]
    pieces = state["pieces"]
    radius = state["radius"]

    cls = piece["class"]
    if cls not in pieces_data["classes"]:
        return []
    c_actions = pieces_data["classes"][cls]["actions"]

    results=[]
    # move
    if "move" in c_actions:
        rng = c_actions["move"].get("range",1)
        coords= all_hexes_in_radius(radius)
        for (q,r) in coords:
            dist= hex_distance(piece["q"], piece["r"], q,r)
            if dist<=rng and not is_occupied_or_blocked(q,r, pieces, blocked_set):
                if (q,r)!=(piece["q"],piece["r"]):
                    results.append(("move", q, r))

    # single_target_attack, multi_target, aoe, swap, etc.
    for aname, adesc in c_actions.items():
        if aname=="move":
            continue
        if "action_type" not in adesc:
            continue
        atype= adesc["action_type"]
        rng = adesc.get("range",0)
        requires_los = adesc.get("requires_los", False)
        ally_only    = adesc.get("ally_only", False)
        rad_aoe      = adesc.get("radius", 0)
        if atype=="single_target_attack":
            enemies=[e for e in pieces if e["side"]!=side and not e.get("dead",False)]
            for e in enemies:
                d=hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                if d<=rng:
                    if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"],e["r"], blocked_set, pieces):
                        results.append(("single_target_attack", e["q"], e["r"]))
        elif atype=="multi_target_attack":
            # Attempt a partial approach. We'll actually produce a single move for each sub-combo.
            # Then in apply_single_action, we kill them all. 
            max_n = adesc.get("max_num_targets",1)
            enemies_in_range=[]
            for e in pieces:
                if e["side"]!=side and not e.get("dead",False):
                    dist=hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist<=rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_set, pieces):
                            enemies_in_range.append(e)
            # produce combos up to max_n
            combos=[]
            for size in range(1, max_n+1):
                for cset in combinations(enemies_in_range,size):
                    combos.append(list(cset))
            # store each as a single action ("multi_target_attack", [list_of_coords], 0)
            for cset in combos:
                # we store them as a list of (q,r) so apply_single_action can handle them
                coords = [(t["q"], t["r"]) for t in cset]
                results.append(("multi_target_attack", coords,0))

        elif atype=="aoe":
            # if there's at least 1 enemy in range => produce an "aoe"
            enemies_in_range=[]
            for e in pieces:
                if e["side"]!=side and not e.get("dead",False):
                    d=hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if d<=rad_aoe:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_set, pieces):
                            enemies_in_range.append(e)
            if enemies_in_range:
                results.append(("aoe",0,0))

        elif atype=="swap_position":
            dist_allowed=rng
            if ally_only:
                possible = [pp for pp in pieces if pp["side"]==side and not pp.get("dead",False) and pp!=piece]
            else:
                possible = [pp for pp in pieces if not pp.get("dead",False) and pp!=piece]
            for t in possible:
                d=hex_distance(piece["q"], piece["r"], t["q"], t["r"])
                if d<=dist_allowed:
                    if (not requires_los) or line_of_sight(piece["q"], piece["r"], t["q"], t["r"], blocked_set, pieces):
                        results.append(("swap_position", t["q"], t["r"]))
    return results

def apply_single_action(state, piece, action_tuple):
    """
    Apply a single piece's action in-place. 
    action_tuple => (atype, x, y) or 
                    for multi_target_attack => (atype, [(x1,y1),(x2,y2),...], 0)
    """
    new_st = copy.deepcopy(state)
    # find the piece by label
    side=piece["side"]
    label=piece["label"]
    p2 = next((pp for pp in new_st["pieces"] if pp["side"]==side and pp["label"]==label and not pp.get("dead",False)), None)
    if not p2:
        return new_st

    atype, x, y = action_tuple
    if atype=="move":
        p2["q"], p2["r"] = x,y
    elif atype=="pass":
        pass
    elif atype=="single_target_attack":
        victim= next((v for v in new_st["pieces"] if v["side"]!=side and not v.get("dead",False) and (v["q"],v["r"])==(x,y)), None)
        if victim:
            victim["dead"]=True
    elif atype=="multi_target_attack":
        # x is a list of coords
        coordsList = x
        for (vq,vr) in coordsList:
            vic= next((vv for vv in new_st["pieces"] if vv["side"]!=side and not vv.get("dead",False) and (vv["q"],vv["r"])==(vq,vr)), None)
            if vic:
                vic["dead"]=True
    elif atype=="aoe":
        # find the radius from the piece's data
        c_actions = pieces_data["classes"][p2["class"]]["actions"]
        for aname, desc in c_actions.items():
            if desc.get("action_type","")== "aoe":
                rad=desc.get("radius",1)
                # kill enemies in range
                enemies=[e for e in new_st["pieces"] if e["side"]!=side and not e.get("dead",False)]
                for e in enemies:
                    dist=hex_distance(p2["q"], p2["r"], e["q"], e["r"])
                    if dist<=rad:
                        e["dead"]=True
    elif atype=="swap_position":
        occupant= next((oc for oc in new_st["pieces"] if not oc.get("dead",False) and (oc["q"], oc["r"])==(x,y)), None)
        if occupant:
            oldQ, oldR= p2["q"], p2["r"]
            p2["q"], p2["r"] = occupant["q"], occupant["r"]
            occupant["q"], occupant["r"] = oldQ, oldR

    return new_st

def switch_side(state):
    ns=copy.deepcopy(state)
    ns["sideToMove"] = "enemy" if state["sideToMove"]=="player" else "player"
    return ns

def move_to_string(piece, atype, tq, tr):
    lbl= piece["label"]
    if atype=="move":
        return f"{lbl}=move({tq},{tr})"
    elif atype=="pass":
        return f"{lbl}=pass"
    elif atype=="single_target_attack":
        return f"{lbl}=atk({tq},{tr})"
    elif atype=="multi_target_attack":
        # tq is a list of coords
        coordsStr="|".join([f"({xx},{yy})" for (xx,yy) in tq])
        return f"{lbl}=multiatk[{coordsStr}]"
    elif atype=="aoe":
        return f"{lbl}=aoe"
    elif atype=="swap_position":
        return f"{lbl}=swap({tq},{tr})"
    return f"{lbl}={atype}({tq},{tr})"

################################################################################
# 5) Main: generate, test, save
################################################################################

def main():
    args = parse_arguments()
    MAX_TRIES=2000
    found_any=False
    for attempt in range(MAX_TRIES):
        try:
            scenario= generate_random_scenario(args)
        except ValueError:
            continue
        if is_forced_mate_in_2(scenario):
            # success => save
            with open(OUTPUT_FILE,"w",encoding="utf-8") as f:
                yaml.dump([scenario], f, sort_keys=False)
            print(f"Success on attempt {attempt+1}, puzzle => {OUTPUT_FILE}")
            print(scenario)
            found_any=True
            break
    if not found_any:
        print("No forced mate-in-2 puzzle (with unique solution) found after all tries.")


if __name__=="__main__":
    main()
