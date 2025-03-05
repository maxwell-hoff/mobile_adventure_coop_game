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
import logging
import time
from itertools import combinations, permutations

# Add TRACE level (even more detailed than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('puzzle_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# Load piece data
logger.info("Loading piece data from pieces.yaml")
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
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--trace", action="store_true", help="Enable extremely verbose trace logging (hex calculations, etc).")
    parser.add_argument("--max-attempts", type=int, default=2000, 
                        help="Maximum number of puzzle generation attempts.")
    parser.add_argument("--min-difficulty", type=int, default=1, choices=[1, 2, 3],
                        help="Minimum puzzle difficulty (1=Easy, 2=Medium, 3=Hard)")
    parser.add_argument("--log-file-only", action="store_true", 
                        help="Write logs only to file, not to console")
    args = parser.parse_args()
    
    # Set up handlers based on log-file-only flag
    if args.log_file_only:
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add file handler only
        logger.addHandler(logging.FileHandler('puzzle_generator.log'))
    
    # Set log level based on debug/trace flags
    if args.trace:
        logger.setLevel(TRACE_LEVEL)
        logger.info("Trace logging enabled (very verbose)")
    elif args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    logger.info(f"Arguments parsed: {vars(args)}")
    return args

################################################################################
# 2) Basic helpers: distance, LOS, occupancy
################################################################################

def hex_distance(q1,r1,q2,r2):
    """Calculate hex distance between two points using axial coordinates."""
    distance = (abs(q1-q2)+abs(r1-r2)+abs((q1+r1)-(q2+r2)))//2
    logger.trace(f"Hex distance from ({q1},{r1}) to ({q2},{r2}) = {distance}")
    return distance

def all_hexes_in_radius(radius):
    """Generate all valid hex coordinates within given radius."""
    logger.debug(f"Generating all hexes within radius {radius}")
    coords=[]
    for q in range(-radius,radius+1):
        for r in range(-radius,radius+1):
            if abs(q+r)<=radius:
                coords.append((q,r))
    logger.debug(f"Generated {len(coords)} valid hexes within radius {radius}")
    return coords

def is_occupied_or_blocked(q,r,pieces, blocked_set):
    """Check if a hex is occupied by a piece or is in the blocked set."""
    # Check if blocked
    if (q,r) in blocked_set:
        logger.trace(f"Hex ({q},{r}) is blocked")
        return True
    
    # Check if occupied by a piece
    for p in pieces:
        if not p.get("dead",False) and (p["q"],p["r"])==(q,r):
            logger.trace(f"Hex ({q},{r}) is occupied by {p['side']} {p['class']}")
            return True
    
    logger.trace(f"Hex ({q},{r}) is free")
    return False

def line_of_sight(q1,r1,q2,r2, blocked_set, all_pieces):
    """Check if there is line of sight between two points."""
    logger.trace(f"Checking line of sight from ({q1},{r1}) to ({q2},{r2})")
    
    # Same point always has LOS
    if (q1==q2) and (r1==r2):
        logger.trace("Same point - LOS exists")
        return True
    
    N = max(abs(q2-q1), abs(r2-r1), abs((q1+r1)-(q2+r2)))
    if N==0:
        logger.trace("Zero distance - LOS exists")
        return True
    
    s1=-q1-r1
    s2=-q2-r2
    line_hexes=[]
    
    # Calculate all hexes along the line
    for i in range(N+1):
        t = i/N
        qf=q1+(q2-q1)*t
        rf=r1+(r2-r1)*t
        sf=s1+(s2-s1)*t
        rq=round(qf)
        rr=round(rf)
        rs=round(sf)
        
        # Fix rounding to maintain q+r+s=0 constraint
        qdiff=abs(rq-qf)
        rdiff=abs(rr-rf)
        sdiff=abs(rs-sf)
        if qdiff>rdiff and qdiff>sdiff:
            rq=-rr-rs
        elif rdiff>sdiff:
            rr=-rq-rs
        
        line_hexes.append((rq,rr))
    
    # Skip first and last hex (source and destination)
    for idx, (hq,hr) in enumerate(line_hexes[1:-1], 1):
        # Check if hex is blocked
        if (hq,hr) in blocked_set:
            logger.trace(f"LOS blocked by obstacle at ({hq},{hr})")
            return False
        
        # Check if hex is occupied by a piece
        occupant = next((pp for pp in all_pieces if not pp.get("dead",False) and (pp["q"],pp["r"])==(hq,hr)), None)
        if occupant:
            logger.trace(f"LOS blocked by {occupant['side']} {occupant['class']} at ({hq},{hr})")
            return False
    
    logger.trace("LOS exists - path is clear")
    return True

def is_priest_dead(pieces, side="enemy"):
    """Check if the Priest of the specified side is dead."""
    for p in pieces:
        if p["side"]==side and p["class"]=="Priest" and not p.get("dead",False):
            # Only log priest alive status at lower depths to reduce spam
            if logger.isEnabledFor(TRACE_LEVEL):
                logger.trace(f"{side.capitalize()} Priest is alive")
            return False
    
    logger.debug(f"{side.capitalize()} Priest is dead")
    return True

################################################################################
# 3) Scenario generation: at least 1 Priest per side, BloodWarden only for enemy
################################################################################

def generate_random_scenario(args, attempt_number=0):
    """Generate a random puzzle scenario based on provided arguments."""
    logger.info(f"---------- ATTEMPT {attempt_number+1} ----------")
    logger.info(f"Generating random puzzle scenario (attempt {attempt_number+1}/{args.max_attempts})")
    start_time = time.time()
    
    # Pick grid radius
    if args.randomize_radius:
        radius = random.randint(args.radius_min, args.radius_max)
        logger.info(f"Using randomized grid radius: {radius}")
    else:
        radius = random.choice([3,4])
        logger.info(f"Using default grid radius: {radius}")

    # Generate all possible hex coordinates within radius
    coords = all_hexes_in_radius(radius)
    random.shuffle(coords)
    logger.debug(f"Generated and shuffled {len(coords)} potential hex coordinates")

    # Generate blocked hexes
    block_count = 0
    if args.randomize_blocked:
        block_count = random.randint(args.min_blocked, min(args.max_blocked, len(coords)))
        logger.info(f"Using randomized blocked hex count: {block_count}")
    
    blocked = set(coords[:block_count])
    free_spots = coords[block_count:]
    blocked_hexes = [{"q":q,"r":r} for (q,r) in blocked]
    
    logger.debug(f"Created {len(blocked_hexes)} blocked hexes, {len(free_spots)} free spots remaining")
    if len(blocked_hexes) > 0:
        logger.debug(f"Blocked hex coordinates: {blocked_hexes}")

    # Generate pieces for each side
    logger.info(f"Building player pieces (min: {args.player_min_pieces}, max: {args.player_max_pieces})")
    player_pieces = build_side_pieces("player", args.player_min_pieces, args.player_max_pieces)
    
    logger.info(f"Building enemy pieces (min: {args.enemy_min_pieces}, max: {args.enemy_max_pieces})")
    enemy_pieces = build_side_pieces("enemy", args.enemy_min_pieces, args.enemy_max_pieces)
    
    all_pieces = player_pieces + enemy_pieces
    logger.info(f"Created {len(player_pieces)} player pieces and {len(enemy_pieces)} enemy pieces")

    # Ensure we have enough free spots for all pieces
    if len(all_pieces) > len(free_spots):
        logger.error(f"Not enough free spots ({len(free_spots)}) for all pieces ({len(all_pieces)})")
        raise ValueError("Not enough free spots to place pieces.")

    # Place pieces on the board
    logger.debug("Placing pieces on the board")
    random.shuffle(free_spots)
    for i, pc in enumerate(all_pieces):
        pc["q"], pc["r"] = free_spots[i]
        logger.debug(f"Placed {pc['side']} {pc['class']} at ({pc['q']},{pc['r']})")

    # Create the scenario
    scenario = {
        "name": f"Puzzle Scenario {attempt_number+1}",
        "subGridRadius": radius,
        "blockedHexes": blocked_hexes,
        "pieces": all_pieces,
        "difficulty": args.min_difficulty,  # Include difficulty level
        "attempt_number": attempt_number+1  # Store attempt number for reference
    }
    
    # Log piece positions clearly for reference
    logger.info("Initial piece positions:")
    for p in player_pieces:
        logger.info(f"  PLAYER {p['class']} at ({p['q']},{p['r']})")
    for e in enemy_pieces:
        logger.info(f"  ENEMY {e['class']} at ({e['q']},{e['r']})")
    
    generation_time = time.time() - start_time
    logger.info(f"Scenario generation completed in {generation_time:.2f} seconds")
    return scenario

def build_side_pieces(side, min_total, max_total):
    """Build a set of pieces for one side, ensuring at least one Priest."""
    logger.debug(f"Building {side} pieces (min: {min_total}, max: {max_total})")
    
    # Set color based on side
    color = "#556b2f" if side=="player" else "#dc143c"
    
    # Randomize piece count within bounds
    count = random.randint(min_total, max_total)
    logger.debug(f"Selected {count} pieces for {side} side")
    
    # Always include one Priest
    pieces = [{
        "class": "Priest",
        "label": "P",
        "color": color,
        "side": side,
        "q": None,
        "r": None,
        "dead": False,
        "original_class": "Priest"  # Track original class for validation
    }]
    logger.debug(f"Added mandatory Priest for {side} side")
    
    # Get valid classes for remaining pieces
    valid_classes = list(pieces_data["classes"].keys())
    
    # Apply restrictions based on side
    if side == "player":
        valid_classes = [c for c in valid_classes if c != "BloodWarden"]
        logger.debug("Restricted BloodWarden class for player side")
    
    # Remove Priest from valid classes as we already added one
    valid_classes = [c for c in valid_classes if c != "Priest"]
    
    # Add remaining pieces
    needed = count - 1
    logger.debug(f"Adding {needed} more pieces for {side} side")
    
    for i in range(needed):
        c = random.choice(valid_classes)
        lbl = pieces_data["classes"][c].get("label", c[0].upper()) or c[0].upper()
        
        piece = {
            "class": c,
            "label": lbl,
            "color": color,
            "side": side,
            "q": None,
            "r": None,
            "dead": False,
            "original_class": c  # Track original class for validation
        }
        
        logger.debug(f"Added {side} {c} (label: {lbl})")
        pieces.append(piece)
    
    logger.debug(f"Completed building {len(pieces)} pieces for {side} side")
    return pieces

################################################################################
# 4) The multi-step BFS for each half-turn
################################################################################

def is_forced_mate_in_2(scenario):
    """
    Determine if the scenario is a "forced mate in 2" puzzle.
    
    We do 4 half-turns: depth=0..3. 
    Each half-turn => BFS over partial moves for each living piece in all permutations 
    (piece A moves, then piece B moves, etc.). This addresses partial-turn synergy.

    If in all final lines the enemy's Priest is dead => forced mate. 
    Then we check how many distinct winning combos the player has => if >1 => discard puzzle => return False
    """
    attempt_num = scenario.get("attempt_number", 1)
    logger.info(f"Analyzing scenario {attempt_num} for forced mate in 2")
    start_time = time.time()
    
    # Initialize the game state from the scenario
    state = {
        "pieces": copy.deepcopy(scenario["pieces"]),
        "blockedHexes": {(bh["q"], bh["r"]) for bh in scenario["blockedHexes"]},
        "radius": scenario["subGridRadius"],
        "sideToMove": "player"
    }
    
    # Ensure all pieces start alive
    for p in state["pieces"]:
        p["dead"] = False
    
    logger.debug(f"Starting analysis with {len(state['pieces'])} pieces and {len(state['blockedHexes'])} blocked hexes")
    
    # Track all possible game lines
    lines = []
    
    # Recursively enumerate all possible play lines
    logger.debug("Starting line enumeration")
    enumerate_lines(state, depth=0, partialPlayerCombos=[], lines=lines)
    logger.info(f"Found {len(lines)} distinct play lines")
    
    # If any line doesn't result in enemy priest death, not a forced mate
    if any(l["priestDead"] == False for l in lines):
        logger.info(f"ATTEMPT {attempt_num} FAILED: Not all lines result in enemy priest death")
        return False
    
    # If no valid lines found, not a valid puzzle
    if not lines:
        logger.info(f"ATTEMPT {attempt_num} FAILED: No valid play lines found")
        return False
    
    # Check if there is exactly one winning strategy (uniqueness)
    logger.debug("Checking uniqueness of winning strategies")
    winning_combos = set()
    
    for l in lines:
        # Extract player moves on turn 0 and turn 2
        # Format: [ { "turn":0, "comboStr":"(moves)"}, { "turn":2, "comboStr":"(moves)"} ]
        c0 = next((x for x in l["partialPlayerCombos"] if x["turn"] == 0), None)
        c2 = next((x for x in l["partialPlayerCombos"] if x["turn"] == 2), None)
        
        # Create a pair representing this winning strategy
        pair = (
            c0["comboStr"] if c0 else "", 
            c2["comboStr"] if c2 else ""
        )
        winning_combos.add(pair)
    
    logger.info(f"Found {len(winning_combos)} distinct winning strategies")
    
    # If more than one winning strategy, puzzle is not unique
    if len(winning_combos) > 1:
        logger.info(f"ATTEMPT {attempt_num} FAILED: Multiple winning strategies - puzzle not unique")
        return False
    
    analysis_time = time.time() - start_time
    logger.info(f"Success! Forced mate analysis completed in {analysis_time:.2f} seconds")
    logger.info(f"ATTEMPT {attempt_num} SUCCESS: Valid forced mate in 2 puzzle with unique solution")
    
    # Log remaining living pieces in final positions
    winning_line = lines[0]  # Just take the first line since all are valid and lead to the same outcome
    
    if 'finalState' not in winning_line:
        # If not already stored, reconstruct a basic final state display
        logger.info("Final piece positions would vary based on player & enemy choices")
    else:
        final_state = winning_line['finalState']
        logger.info("Final piece positions in winning line:")
        living_player = [p for p in final_state['pieces'] if p['side'] == 'player' and not p.get('dead', False)]
        living_enemy = [p for p in final_state['pieces'] if p['side'] == 'enemy' and not p.get('dead', False)]
        
        for p in living_player:
            logger.info(f"  PLAYER {p['class']} at ({p['q']},{p['r']})")
        for e in living_enemy:
            logger.info(f"  ENEMY {e['class']} at ({e['q']},{e['r']})")
    
    return True

def enumerate_lines(state, depth, partialPlayerCombos, lines):
    """
    BFS for depth up to 4. If enemy Priest is dead => store line.
    Otherwise, build all possible *full-turn sequences* for the side's pieces in permutations.
    Each piece acts one at a time, in any order.
    Then apply the resulting final board => next depth => recursive.

    partialPlayerCombos => track which combos the player used at turn=0 or 2 for uniqueness checking.
    """
    # Use a global counter to track and limit enumeration logs
    global _enum_log_count
    if not '_enum_log_count' in globals():
        _enum_log_count = {}
        
    # Store attempt number to reset logs between attempts
    attempt_key = f"depth_{depth}"
    if attempt_key not in _enum_log_count:
        _enum_log_count[attempt_key] = 0
    
    # Only log a few instances of enumeration for each depth
    # Completely suppress logs for depth 4
    if depth < 4 and _enum_log_count[attempt_key] < 5:
        logger.debug(f"Enumerating lines at depth {depth}, side: {state['sideToMove']} (sample {_enum_log_count[attempt_key]+1}/5)")
        _enum_log_count[attempt_key] += 1
    
    # Check win condition - enemy Priest is dead
    if is_priest_dead(state["pieces"], "enemy"):
        logger.debug(f"Depth {depth}: Enemy priest is dead - winning line found")
        
        # Store winning line with final state for analysis
        line_data = {
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": True,
            "finalState": copy.deepcopy(state),  # Store the final state for analysis
            "depth": depth  # Store the depth at which the win occurred
        }
        lines.append(line_data)
        return

    # Use a global counter to only log the first few depth 4 instances
    global _depth4_log_count
    if not '_depth4_log_count' in globals():
        _depth4_log_count = 0
        
    # Check termination condition - reached maximum depth
    if depth >= 4:
        # Only log a limited number of depth 4 messages
        if _depth4_log_count < 5:
            logger.debug(f"Depth {depth}: Maximum depth reached without priest death (showing 5/{_depth4_log_count+1})")
            _depth4_log_count += 1
        
        # Store non-winning line
        line_data = {
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": False,
            "finalState": copy.deepcopy(state),  # Store the final state for analysis
            "depth": depth
        }
        lines.append(line_data)
        return

    side = state["sideToMove"]
    logger.debug(f"Current side to move: {side}")
    
    # Find living pieces for the current side
    living = [p for p in state["pieces"] if p["side"] == side and not p.get("dead", False)]
    logger.debug(f"Living {side} pieces: {len(living)}")
    
    # If no living pieces, skip turn and continue
    if not living:
        logger.debug(f"No living pieces for {side} - skipping turn")
        st2 = switch_side(state)
        enumerate_lines(st2, depth+1, partialPlayerCombos, lines)
        return

    # Generate all permutations of piece movement order
    # For example: with pieces A, B, C -> (A,B,C), (A,C,B), (B,A,C), etc.
    all_perms = permutations(living, r=len(living))
    perm_count = math.factorial(len(living))
    logger.debug(f"Analyzing {perm_count} permutations of piece movement order")
    
    # Track permutation progress
    perm_index = 0
    
    # For each possible ordering of pieces
    for perm in all_perms:
        perm_index += 1
        piece_labels = [p["label"] for p in perm]
        logger.debug(f"Analyzing permutation {perm_index}/{perm_count}: {piece_labels}")
        
        # For each permutation, do a BFS over piece actions
        # Each state represents the game after applying some sequence of moves
        partial_results = []
        initial_state = copy.deepcopy(state)
        
        # List of (currentState, listOfMovesUsed)
        partial_BFS_states = [(initial_state, [])]
        
        # For each piece in this permutation order
        for piece_index, piece in enumerate(perm):
            logger.trace(f"Processing piece {piece_index+1}/{len(perm)}: {piece['side']} {piece['class']} ({piece['label']})")
            
            # Will hold the new states after this piece acts
            new_partial = []
            state_count = len(partial_BFS_states)
            logger.trace(f"Expanding {state_count} partial states")
            
            # For each partial game state so far
            for state_index, (stSoFar, movesUsedSoFar) in enumerate(partial_BFS_states):
                # Get all possible actions for this piece in this state
                piece_actions = gather_single_piece_actions(stSoFar, piece)
                logger.trace(f"State {state_index+1}/{state_count}: Found {len(piece_actions) if piece_actions else 0} possible actions")
                
                # If no valid actions, add a pass
                if not piece_actions:
                    logger.trace(f"No valid actions for {piece['label']} - adding PASS")
                    newState = apply_single_action(stSoFar, piece, ("pass", 0, 0))
                    newMoves = movesUsedSoFar + [f"{piece['label']}=pass"]
                    new_partial.append((newState, newMoves))
                else:
                    # For each possible action
                    for action_index, action_tuple in enumerate(piece_actions):
                        atype = action_tuple[0]
                        logger.trace(f"Applying action {action_index+1}/{len(piece_actions)}: {atype}")
                        
                        # Apply the action
                        st3 = apply_single_action(stSoFar, piece, action_tuple)
                        
                        # Format action for move string
                        if len(action_tuple) >= 3:
                            tq, tr = action_tuple[1], action_tuple[2]
                        else:
                            tq, tr = 0, 0  # Default values
                            
                        moveStr = move_to_string(piece, atype, tq, tr)
                        newMoves = movesUsedSoFar + [moveStr]
                        new_partial.append((st3, newMoves))
            
            # Update states for next piece
            partial_BFS_states = new_partial
            logger.debug(f"After piece {piece['label']}: {len(partial_BFS_states)} partial states")
        
        # After processing all pieces in this permutation
        logger.debug(f"Completed permutation {perm_index}/{perm_count} with {len(partial_BFS_states)} final states")
        
        # Now process each final state
        for state_index, (finalTurnState, moveList) in enumerate(partial_BFS_states):
            logger.trace(f"Processing final state {state_index+1}/{len(partial_BFS_states)}")
            
            # Describe the current board state for better debugging
            if depth <= 1 and state_index == 0:  # Only for first depth and first state to avoid spam
                living_player = [p for p in finalTurnState["pieces"] if p["side"] == "player" and not p.get("dead", False)]
                living_enemy = [p for p in finalTurnState["pieces"] if p["side"] == "enemy" and not p.get("dead", False)]
                
                logger.debug(f"Board state after turn {depth}:")
                for p in living_player:
                    logger.debug(f"  PLAYER {p['class']} at ({p['q']},{p['r']})")
                for e in living_enemy:
                    logger.debug(f"  ENEMY {e['class']} at ({e['q']},{e['r']})")
            
            # Switch to other side for next turn
            st2 = switch_side(finalTurnState)
            
            # Build move sequence string
            comboStr = ";".join(moveList)
            
            # If player's turn, store the move combination
            newPartialPC = partialPlayerCombos
            if side == "player":
                logger.trace(f"Storing player move combination for turn {depth}")
                newPartialPC = copy.deepcopy(partialPlayerCombos)
                newPartialPC.append({
                    "turn": depth,
                    "comboStr": comboStr,
                    "moves": moveList  # Store individual moves for easier debugging
                })
            
            # Continue with next depth
            enumerate_lines(st2, depth+1, newPartialPC, lines)

def gather_single_piece_actions(state, piece):
    """
    Gather all possible actions for a single piece based on the current state.
    """
    side = piece["side"]
    pieces = state["pieces"]
    blocked_set = state["blockedHexes"]
    radius = state["radius"]
    
    logger.trace(f"Gathering actions for {side} {piece['class']} at ({piece['q']},{piece['r']})")
    
    # Find the piece class data
    piece_class = piece["class"]
    if piece_class not in pieces_data["classes"]:
        logger.warning(f"Unknown piece class: {piece_class}")
        return []  # Unknown class
    class_data = pieces_data["classes"][piece_class]
    
    results = []
    
    # Process each action type
    for action_name, action in class_data.get("actions", {}).items():
        atype = action.get("action_type")
        adesc = action.get("description", "")
        
        # Common action parameters
        rng = action.get("range", 1)
        requires_los = action.get("requires_los", False)
        
        logger.trace(f"Checking action: {action_name} (type: {atype}, range: {rng})")
        
        if atype == "move":
            # Calculate all valid move destinations
            move_count = 0
            for q in range(-radius, radius+1):
                for r in range(-radius, radius+1):
                    if abs(q+r) <= radius:
                        dist = hex_distance(piece["q"], piece["r"], q, r)
                        if 0 < dist <= rng and not is_occupied_or_blocked(q, r, pieces, blocked_set):
                            results.append(("move", q, r))
                            move_count += 1
            logger.trace(f"Found {move_count} valid move destinations")
        
        elif atype == "single_target_attack":
            # Find all valid attack targets
            attack_count = 0
            for e in pieces:
                if e["side"] != side and not e.get("dead", False):
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_set, pieces):
                            results.append(("single_target_attack", e["q"], e["r"]))
                            attack_count += 1
            logger.trace(f"Found {attack_count} valid single-target attack targets")
        
        elif atype == "multi_target_attack":
            # Improved multi-target handling
            max_n = action.get("max_num_targets", 1)
            enemies_in_range = []
            for e in pieces:
                if e["side"] != side and not e.get("dead", False):
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"], blocked_set, pieces):
                            enemies_in_range.append(e)
            
            # Generate all valid combinations of targets up to max_n
            combo_count = 0
            if enemies_in_range:
                logger.trace(f"Found {len(enemies_in_range)} enemies in range for multi-target attack")
                for size in range(1, min(max_n + 1, len(enemies_in_range) + 1)):
                    for cset in combinations(enemies_in_range, size):
                        coords = [(t["q"], t["r"]) for t in cset]
                        results.append(("multi_target_attack", coords, 0))
                        combo_count += 1
                logger.trace(f"Generated {combo_count} target combinations for multi-target attack")
            else:
                logger.trace("No enemies in range for multi-target attack")
        
        elif atype == "aoe":
            # Area of effect attack - find center of attacker
            q, r = piece["q"], piece["r"]
            
            rad_aoe = action.get("radius", 1)
            
            # Find enemies in radius
            enemies_in_range = []
            for p in pieces:
                if p["side"] != side and not p.get("dead", False):
                    dist = hex_distance(q, r, p["q"], p["r"])
                    if dist <= rad_aoe:
                        enemies_in_range.append(p)
            
            # If there are enemies in range, add the AOE action to results
            if enemies_in_range:
                results.append(("aoe", q, r, rad_aoe))
                logger.trace(f"Found {len(enemies_in_range)} enemies in AOE range")
            else:
                logger.trace("No enemies in AOE range")
        
        elif atype == "swap_position":
            # Position swapping (e.g., for certain special abilities)
            swap_count = 0
            for ally in pieces:
                if ally["side"] == side and not ally.get("dead", False) and ally != piece:
                    dist = hex_distance(piece["q"], piece["r"], ally["q"], ally["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], ally["q"], ally["r"], blocked_set, pieces):
                            results.append(("swap_position", ally["q"], ally["r"]))
                            swap_count += 1
            logger.trace(f"Found {swap_count} valid swap targets")
        
        elif atype == "delayed_effect":
            # Handle delayed effect actions like BloodWarden's abilities
            # This is a simplified version - a full implementation would track state across turns
            target_types = action.get("target_types", ["enemy"])
            effect_count = 0
            for p in pieces:
                if ((p["side"] != side and "enemy" in target_types) or 
                    (p["side"] == side and "ally" in target_types)):
                    if not p.get("dead", False):
                        dist = hex_distance(piece["q"], piece["r"], p["q"], p["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(piece["q"], piece["r"], p["q"], p["r"], blocked_set, pieces):
                                results.append(("delayed_effect", p["q"], p["r"]))
                                effect_count += 1
            logger.trace(f"Found {effect_count} valid targets for delayed effect")
    
    logger.trace(f"Total actions gathered: {len(results)}")
    return results

def apply_single_action(state, piece, action_tuple):
    """
    Apply a single action for one piece to the state.
    Returns a new state with the action applied.
    """
    new_state = copy.deepcopy(state)
    new_pieces = new_state["pieces"]
    
    # Find the piece in the new_state
    try:
        piece_idx = next(i for i, p in enumerate(new_pieces) if p["q"] == piece["q"] and p["r"] == piece["r"] 
                        and p["side"] == piece["side"] and p["class"] == piece["class"])
    except StopIteration:
        # If we can't find the exact piece, try to find it by coordinates only (maybe class/side changed)
        try:
            piece_idx = next(i for i, p in enumerate(new_pieces) if p["q"] == piece["q"] and p["r"] == piece["r"])
        except StopIteration:
            # If all else fails, add the piece to new_pieces
            new_pieces.append(copy.deepcopy(piece))
            piece_idx = len(new_pieces) - 1
    
    action_type = action_tuple[0]
    
    if action_type == "move":
        # Simple movement - update piece position
        tq, tr = action_tuple[1], action_tuple[2]
        new_pieces[piece_idx]["q"] = tq
        new_pieces[piece_idx]["r"] = tr
    
    elif action_type == "single_target_attack":
        # Attack a single target
        tq, tr = action_tuple[1], action_tuple[2]
        
        # Find target and mark as dead
        for p in new_pieces:
            if p["q"] == tq and p["r"] == tr and not p.get("dead", False):
                p["dead"] = True
                break
    
    elif action_type == "multi_target_attack":
        # Attack multiple targets
        target_coords = action_tuple[1]  # List of (q,r) tuples
        
        # Mark each target as dead
        for (tq, tr) in target_coords:
            for p in new_pieces:
                if p["q"] == tq and p["r"] == tr and not p.get("dead", False):
                    p["dead"] = True
                    break
    
    elif action_type == "aoe":
        # Area of effect attack - get attacker position and radius
        q, r = new_pieces[piece_idx]["q"], new_pieces[piece_idx]["r"]
        rad_aoe = action_tuple[3] if len(action_tuple) > 3 else 1
        
        # Mark all enemies in radius as dead
        for p in new_pieces:
            if p["side"] != piece["side"] and not p.get("dead", False):
                dist = hex_distance(q, r, p["q"], p["r"])
                if dist <= rad_aoe:
                    # Optionally check LOS based on ability
                    p["dead"] = True
    
    elif action_type == "swap_position":
        # Swap positions with another piece
        tq, tr = action_tuple[1], action_tuple[2]
        
        # Find target piece to swap with
        target_idx = next((i for i, p in enumerate(new_pieces) 
                         if p["q"] == tq and p["r"] == tr and not p.get("dead", False)), None)
        
        if target_idx is not None:
            # Save original positions
            orig_q, orig_r = new_pieces[piece_idx]["q"], new_pieces[piece_idx]["r"]
            
            # Swap positions
            new_pieces[piece_idx]["q"] = tq
            new_pieces[piece_idx]["r"] = tr
            new_pieces[target_idx]["q"] = orig_q
            new_pieces[target_idx]["r"] = orig_r
    
    elif action_type == "delayed_effect":
        # For delayed effects, we'd need to add a marker to the state
        # This is a simplified implementation
        tq, tr = action_tuple[1], action_tuple[2]
        
        # Find target and add a "marked" flag for delayed effect
        for p in new_pieces:
            if p["q"] == tq and p["r"] == tr and not p.get("dead", False):
                p["marked_by"] = piece["class"]
                # In a full implementation, we'd track how many turns until effect triggers
                break
    
    return new_state

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
    MAX_TRIES = args.max_attempts
    min_difficulty = args.min_difficulty
    
    logger.info(f"Starting puzzle generation with {MAX_TRIES} max attempts")
    logger.info(f"Minimum difficulty level: {min_difficulty}")
    
    # Progress tracking
    start_time = time.time()
    found_any = False
    valid_scenarios = []
    total_states_explored = 0
    total_lines_analyzed = 0
    last_progress_time = time.time()
    progress_interval = 10  # Print progress every 10 seconds
    
    # Summary stats for terminal display
    attempts_tried = 0
    failed_reasons = {
        "not_forced": 0,        # No guaranteed win
        "multiple_solutions": 0, # More than one solution
        "no_valid_lines": 0,    # No valid play lines
        "error": 0              # Other errors
    }
    
    for attempt in range(MAX_TRIES):
        attempt_start = time.time()
        current_time = time.time()
        
        # Print periodic progress update
        if current_time - last_progress_time > progress_interval:
            elapsed = current_time - start_time
            logger.info(f"Progress: {attempt}/{MAX_TRIES} attempts ({(attempt/MAX_TRIES)*100:.1f}%), "
                        f"Time elapsed: {elapsed:.1f}s, Avg time per attempt: {elapsed/(attempt+1):.2f}s")
            last_progress_time = current_time
            
            # Also print failure statistics
            if attempt > 0:
                logger.info(f"Failure stats: not forced={failed_reasons['not_forced']}, "
                           f"multiple solutions={failed_reasons['multiple_solutions']}, "
                           f"no valid lines={failed_reasons['no_valid_lines']}, errors={failed_reasons['error']}")
        
        try:
            # Generate a random scenario with attempt number
            scenario = generate_random_scenario(args, attempt)
            
            # Check if it's a valid "forced mate in 2" puzzle
            if is_forced_mate_in_2(scenario):
                valid_scenarios.append(scenario)
                logger.info(f"Found valid puzzle on attempt {attempt+1}!")
                
                # Set difficulty based on configuration
                scenario["difficulty"] = min_difficulty
                
                # Save to output file
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    yaml.dump([scenario], f, sort_keys=False)
                logger.info(f"Puzzle saved to {OUTPUT_FILE}")
                
                # Print detailed scenario summary
                player_pieces = [p for p in scenario["pieces"] if p["side"] == "player"]
                enemy_pieces = [p for p in scenario["pieces"] if p["side"] == "enemy"]
                
                logger.info("==== SUCCESSFUL SCENARIO SUMMARY ====")
                logger.info(f"Grid radius: {scenario['subGridRadius']}")
                logger.info(f"Blocked hexes: {len(scenario['blockedHexes'])}")
                logger.info("Player pieces:")
                for p in player_pieces:
                    logger.info(f"  {p['class']} at ({p['q']},{p['r']})")
                logger.info("Enemy pieces:")
                for e in enemy_pieces:
                    logger.info(f"  {e['class']} at ({e['q']},{e['r']})")
                logger.info("======================================")
                
                print(f"\nSuccess! Valid puzzle found on attempt {attempt+1}")
                print(f"Puzzle saved to {OUTPUT_FILE}")
                
                found_any = True
                break
            else:
                # Detect failure reason from log messages (simplistic approach)
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        with open(handler.baseFilename, 'r') as f:
                            last_lines = f.readlines()[-10:]  # Last 10 lines
                            
                            # Try to infer failure reason from logs
                            if any("multiple winning" in line.lower() for line in last_lines):
                                failed_reasons["multiple_solutions"] += 1
                            elif any("no valid play lines" in line.lower() for line in last_lines):
                                failed_reasons["no_valid_lines"] += 1
                            elif any("not all lines" in line.lower() for line in last_lines):
                                failed_reasons["not_forced"] += 1
                            else:
                                # Default if we can't determine the exact reason
                                failed_reasons["not_forced"] += 1
        
        except ValueError as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
            failed_reasons["error"] += 1
            continue
        
        except Exception as e:
            logger.error(f"Unexpected error in attempt {attempt+1}: {str(e)}", exc_info=True)
            failed_reasons["error"] += 1
            continue
        
        finally:
            attempts_tried += 1
            attempt_time = time.time() - attempt_start
            logger.debug(f"Attempt {attempt+1} took {attempt_time:.2f} seconds")
    
    # Final summary
    total_time = time.time() - start_time
    if found_any:
        logger.info(f"SUCCESS: Generated a valid puzzle in {total_time:.2f} seconds after {attempt+1} attempts")
        logger.info(f"Failure stats: not forced={failed_reasons['not_forced']}, "
                   f"multiple solutions={failed_reasons['multiple_solutions']}, "
                   f"no valid lines={failed_reasons['no_valid_lines']}, errors={failed_reasons['error']}")
        
        print(f"\nSuccess! Puzzle generation completed in {total_time:.2f} seconds after {attempt+1} attempts")
    else:
        logger.warning(f"FAILED: Could not generate a valid puzzle after {MAX_TRIES} attempts ({total_time:.2f} seconds)")
        logger.warning(f"Failure stats: not forced={failed_reasons['not_forced']}, "
                      f"multiple solutions={failed_reasons['multiple_solutions']}, "
                      f"no valid lines={failed_reasons['no_valid_lines']}, errors={failed_reasons['error']}")
        
        print(f"\nNo forced mate-in-2 puzzle (with unique solution) found after {MAX_TRIES} attempts.")
        print(f"Time elapsed: {total_time:.2f} seconds")
        print(f"Failure statistics:")
        print(f"  - No guaranteed win: {failed_reasons['not_forced']}")
        print(f"  - Multiple solutions: {failed_reasons['multiple_solutions']}")
        print(f"  - No valid play lines: {failed_reasons['no_valid_lines']}")
        print(f"  - Errors: {failed_reasons['error']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical("Unhandled exception", exc_info=True)
        raise
