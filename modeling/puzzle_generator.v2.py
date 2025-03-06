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
import os
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Grid size options
    parser.add_argument("--randomize-radius", action="store_true",
                      help="Randomize the grid radius")
    parser.add_argument("--radius-min", type=int, default=3,
                      help="Minimum grid radius (default: 3)")
    parser.add_argument("--radius-max", type=int, default=4,
                      help="Maximum grid radius (default: 4)")
    
    # Blocked hex options
    parser.add_argument("--randomize-blocked", action="store_true",
                      help="Randomize the number of blocked hexes")
    parser.add_argument("--min-blocked", type=int, default=1,
                      help="Minimum number of blocked hexes (default: 1)")
    parser.add_argument("--max-blocked", type=int, default=3,
                      help="Maximum number of blocked hexes (default: 3)")
    
    # Enemy piece options
    parser.add_argument("--min-extra-enemies", type=int, default=1,
                      help="Minimum number of extra enemy pieces (default: 1)")
    parser.add_argument("--max-extra-enemies", type=int, default=2,
                      help="Maximum number of extra enemy pieces (default: 2)")
    parser.add_argument("--allow-bloodwarden", action="store_true",
                      help="Allow BloodWarden as an enemy piece")
    
    # Generation options
    parser.add_argument("--max-attempts", type=int, default=100,
                      help="Maximum number of generation attempts (default: 100)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument("--output", type=str, default="data/generated_puzzles_v2.yaml",
                      help="Output file path (default: data/generated_puzzles_v2.yaml)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.radius_min < 2:
        parser.error("Minimum radius must be at least 2")
    if args.radius_max < args.radius_min:
        parser.error("Maximum radius must be greater than or equal to minimum radius")
    if args.min_blocked < 0:
        parser.error("Minimum blocked hexes must be non-negative")
    if args.max_blocked < args.min_blocked:
        parser.error("Maximum blocked hexes must be greater than or equal to minimum")
    if args.min_extra_enemies < 0:
        parser.error("Minimum extra enemies must be non-negative")
    if args.max_extra_enemies < args.min_extra_enemies:
        parser.error("Maximum extra enemies must be greater than or equal to minimum")
    
    return args

################################################################################
# 2) Basic helpers: distance, LOS, occupancy
################################################################################

def hex_distance(q1, r1, q2, r2):
    """Calculate the distance between two hex coordinates."""
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2

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

def line_of_sight(q1, r1, q2, r2, blocked_hexes, pieces):
    """
    Check if there is line of sight between two hex coordinates.
    Takes into account blocked hexes and pieces that could block vision.
    """
    # Same point always has line of sight
    if (q1 == q2) and (r1 == r2):
        return True
    
    # Calculate all hexes along the line
    N = max(abs(q2 - q1), abs(r2 - r1), abs((q1 + r1) - (q2 + r2)))
    if N == 0:
        return True
    
    s1 = -q1 - r1
    s2 = -q2 - r2
    line_hexes = []
    
    for i in range(N + 1):
        t = i / N
        qf = q1 + (q2 - q1) * t
        rf = r1 + (r2 - r1) * t
        sf = s1 + (s2 - s1) * t
        rq = round(qf)
        rr = round(rf)
        rs = round(sf)
        
        # Fix rounding to maintain q + r + s = 0 constraint
        qdiff = abs(rq - qf)
        rdiff = abs(rr - rf)
        sdiff = abs(rs - sf)
        if qdiff > rdiff and qdiff > sdiff:
            rq = -rr - rs
        elif rdiff > sdiff:
            rr = -rq - rs
        
        line_hexes.append((rq, rr))
    
    # Skip first and last hex (source and destination)
    for hq, hr in line_hexes[1:-1]:
        # Check if hex is blocked
        if (hq, hr) in blocked_hexes:
            return False
        
        # Check if hex is occupied by a piece
        for p in pieces:
            if not p.get("dead", False) and p["q"] == hq and p["r"] == hr:
                return False
    
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

def can_kill_piece(attacker_class, attacker_q, attacker_r, target_piece, blocked_hexes, pieces):
    """
    Check if a piece of attacker_class at (attacker_q, attacker_r) can kill the target piece.
    Takes into account piece abilities, range, and line of sight requirements.
    """
    # Get attacker's abilities from pieces.yaml
    with open("data/pieces.yaml", "r") as f:
        pieces_data = yaml.safe_load(f)
    
    if attacker_class not in pieces_data["classes"]:
        logger.warning(f"Unknown attacker class: {attacker_class}")
        return False
    
    class_data = pieces_data["classes"][attacker_class]
    
    # Check each action for attack capability
    for action_name, action_data in class_data["actions"].items():
        action_type = action_data["action_type"]
        
        if action_type in ["single_target_attack", "multi_target_attack", "aoe"]:
            attack_range = action_data.get("range", 0)
            requires_los = action_data.get("requires_los", False)
            
            # Calculate distance to target
            dist = hex_distance(attacker_q, attacker_r, target_piece["q"], target_piece["r"])
            
            # For AOE attacks, use radius instead of range
            if action_type == "aoe":
                attack_range = action_data.get("radius", 0)
            
            # Check if target is in range
            if dist <= attack_range:
                # Check line of sight if required
                if not requires_los or line_of_sight(
                    attacker_q, attacker_r,
                    target_piece["q"], target_piece["r"],
                    blocked_hexes, pieces
                ):
                    return True
    
    return False

def is_position_safe(q, r, enemy_pieces, blocked_hexes, pieces):
    """
    Check if a position is safe from all enemy pieces.
    Returns True if no enemy piece can attack the position.
    """
    # Create a dummy piece at the target position
    target = {"q": q, "r": r}
    
    # Check if any enemy piece can attack this position
    for enemy in enemy_pieces:
        if enemy.get("dead", False):
            continue
        if can_kill_piece(enemy["class"], enemy["q"], enemy["r"], target, blocked_hexes, pieces):
            return False
    
    return True

def can_be_killed_by_enemies(piece, enemy_pieces, blocked_hexes, pieces):
    """
    Check if a piece can be killed by any enemy piece.
    Returns True if any enemy piece can kill the target piece.
    """
    # Check each enemy piece
    for enemy in enemy_pieces:
        if enemy.get("dead", False):
            continue
        if can_kill_piece(enemy["class"], enemy["q"], enemy["r"], piece, blocked_hexes, pieces):
            return True
    
    return False

def can_move_to(piece, target_q, target_r, blocked_hexes, pieces):
    """
    Check if a piece can move to the target position.
    Takes into account piece movement abilities and blocked hexes/pieces.
    """
    # Get piece's abilities from pieces.yaml
    with open("data/pieces.yaml", "r") as f:
        pieces_data = yaml.safe_load(f)
    
    if piece["class"] not in pieces_data["classes"]:
        logger.warning(f"Unknown piece class: {piece['class']}")
        return False
    
    class_data = pieces_data["classes"][piece["class"]]
    
    # Check if target hex is blocked or occupied
    if blocked_hexes and (target_q, target_r) in blocked_hexes:
        return False
    for p in pieces:
        if not p.get("dead", False) and p["q"] == target_q and p["r"] == target_r:
            return False
    
    # Check each action for movement capability
    for action_name, action_data in class_data["actions"].items():
        action_type = action_data["action_type"]
        
        if action_type == "move":
            move_range = action_data.get("range", 0)
            requires_los = action_data.get("requires_los", False)
            
            # Calculate distance to target
            dist = hex_distance(piece["q"], piece["r"], target_q, target_r)
            
            # Check if target is in range
            if dist <= move_range:
                # Check line of sight if required
                if not requires_los or line_of_sight(
                    piece["q"], piece["r"],
                    target_q, target_r,
                    blocked_hexes, pieces
                ):
                    return True
    
    return False

def can_reach_in_moves(piece, target_q, target_r, num_moves, blocked_hexes, pieces):
    """
    Check if a piece can reach a target position in a given number of moves.
    Uses breadth-first search to find all reachable positions.
    """
    # Get piece's movement range
    with open("data/pieces.yaml", "r") as f:
        pieces_data = yaml.safe_load(f)
    
    if piece["class"] not in pieces_data["classes"]:
        logger.warning(f"Unknown piece class: {piece['class']}")
        return False
    
    class_data = pieces_data["classes"][piece["class"]]
    move_range = 0
    requires_los = False
    
    # Find movement action
    for action_name, action_data in class_data["actions"].items():
        if action_data["action_type"] == "move":
            move_range = action_data.get("range", 0)
            requires_los = action_data.get("requires_los", False)
            break
    
    if move_range == 0:
        return False
    
    # BFS to find reachable positions
    visited = set()
    current_positions = {(piece["q"], piece["r"])}
    
    for _ in range(num_moves):
        next_positions = set()
        
        for q, r in current_positions:
            # Try all positions within move range
            for dq in range(-move_range, move_range + 1):
                for dr in range(-move_range, move_range + 1):
                    new_q = q + dq
                    new_r = r + dr
                    
                    # Skip if we've already visited this position
                    if (new_q, new_r) in visited:
                        continue
                    
                    # Skip if position is out of range
                    dist = hex_distance(q, r, new_q, new_r)
                    if dist > move_range:
                        continue
                    
                    # Skip if position is blocked or occupied
                    if (new_q, new_r) in blocked_hexes:
                        continue
                    if any(p["q"] == new_q and p["r"] == new_r and not p.get("dead", False) for p in pieces):
                        continue
                    
                    # Check line of sight if required
                    if not requires_los or line_of_sight(q, r, new_q, new_r, blocked_hexes, pieces):
                        next_positions.add((new_q, new_r))
                        visited.add((new_q, new_r))
        
        current_positions = next_positions
        
        # If we can reach the target, return True
        if (target_q, target_r) in current_positions:
            return True
    
    return False

def can_be_protected(piece, protectors, enemy_pieces, blocked_hexes, pieces, radius):
    """
    Check if a piece can be protected by other pieces.
    A piece is protected if:
    1. For each enemy that can attack it
    2. At least one protector can kill that enemy before it attacks
    """
    # Find all enemies that can attack the piece
    attackers = []
    for enemy in enemy_pieces:
        if enemy.get("dead", False):
            continue
        if can_kill_piece(enemy["class"], enemy["q"], enemy["r"], piece, blocked_hexes, pieces):
            attackers.append(enemy)
    
    # If no attackers, piece is safe
    if not attackers:
        return True
    
    # For each attacker
    for attacker in attackers:
        # Check if any protector can kill it
        can_be_killed = False
        for protector in protectors:
            # Check if protector can reach a position to kill the attacker
            for q in range(-radius, radius + 1):
                for r in range(-radius, radius + 1):
                    if abs(q + r) <= radius:
                        # Skip if position is blocked or occupied
                        if (q, r) in blocked_hexes:
                            continue
                        if any(p["q"] == q and p["r"] == r and not p.get("dead", False) for p in pieces):
                            continue
                        
                        # Check if protector can reach this position in 1 move
                        if can_reach_in_moves(protector, q, r, 1, blocked_hexes, pieces):
                            # Check if protector can kill the attacker from here
                            if can_kill_piece(protector["class"], q, r, attacker, blocked_hexes, pieces):
                                can_be_killed = True
                                break
                if can_be_killed:
                    break
            if can_be_killed:
                break
        
        # If no protector can kill this attacker, piece is not protected
        if not can_be_killed:
            return False
    
    return True

def generate_forced_mate_puzzle(radius, blocked_hexes=None, args=None):
    """
    Generate a puzzle that guarantees a forced mate in 2 by working backward from the checkmate position.
    
    Strategy:
    1. Start with enemy Priest position
    2. Place a ranged attacker (Warlock/Sorcerer) at max range
    3. Add blocked hexes and other pieces to constrain enemy movement
    4. Ensure player pieces can't be killed during enemy's turn
    
    Returns a dictionary with the puzzle scenario if successful, None if failed.
    """
    logger.info("Generating forced mate puzzle using backward-working approach")
    
    # Initialize pieces list
    pieces = []
    blocked_hexes = set(blocked_hexes or set())
    
    # 1. Place enemy Priest in center-ish position
    priest_q = random.randint(-1, 1)
    priest_r = random.randint(-1, 1)
    enemy_priest = {
        "class": "Priest",
        "label": "P",
        "color": "#dc143c",
        "side": "enemy",
        "q": priest_q,
        "r": priest_r
    }
    pieces.append(enemy_priest)
    logger.info(f"Placed enemy Priest at ({priest_q},{priest_r})")
    
    # Find all possible moves for enemy Priest
    priest_moves = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                if can_move_to(enemy_priest, q, r, blocked_hexes, pieces):
                    priest_moves.append((q, r))
    
    # We want to limit Priest's movement options to 2-3 hexes
    target_move_count = random.randint(2, 3)
    if len(priest_moves) > target_move_count:
        # Block some hexes around Priest to limit movement
        moves_to_block = random.sample(priest_moves, len(priest_moves) - target_move_count)
        for q, r in moves_to_block:
            blocked_hexes.add((q, r))
        logger.info(f"Added {len(moves_to_block)} blocked hexes to limit Priest movement")
    
    # 2. Place ranged attacker at max range
    # Choose between Warlock (range 2) and Sorcerer (range 3)
    attacker_class = random.choice(["Warlock", "Sorcerer"])
    attack_range = 2 if attacker_class == "Warlock" else 3
    
    # Find all positions at max range that have line of sight
    max_range_positions = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                if (q, r) in blocked_hexes:
                    continue
                dist = hex_distance(q, r, priest_q, priest_r)
                if dist == attack_range:
                    # Check if this position has line of sight to priest
                    if line_of_sight(q, r, priest_q, priest_r, blocked_hexes, pieces):
                        # Check if position is safe from enemy Priest
                        if not can_be_killed_by_enemies({"q": q, "r": r}, [enemy_priest], blocked_hexes, pieces):
                            # Check if position has line of sight to all Priest's possible moves
                            has_los_to_all_moves = True
                            for move_q, move_r in priest_moves:
                                if (move_q, move_r) not in blocked_hexes:
                                    if not line_of_sight(q, r, move_q, move_r, blocked_hexes, pieces):
                                        has_los_to_all_moves = False
                                        break
                            if has_los_to_all_moves:
                                max_range_positions.append((q, r))
    
    if not max_range_positions:
        logger.warning("Could not find valid position for ranged attacker")
        return None
    
    # Pick random position for attacker
    attacker_q, attacker_r = random.choice(max_range_positions)
    attacker = {
        "class": attacker_class,
        "label": "W" if attacker_class == "Warlock" else "S",
        "color": "#556b2f",
        "side": "player",
        "q": attacker_q,
        "r": attacker_r
    }
    pieces.append(attacker)
    logger.info(f"Placed {attacker_class} at ({attacker_q},{attacker_r})")
    
    # 3. Add player Priest (required)
    # Find safe position for player Priest that can't be attacked
    safe_positions = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                if (q, r) in blocked_hexes:
                    continue
                # Skip if position is already taken
                if any(p["q"] == q and p["r"] == r for p in pieces):
                    continue
                # Skip if position can be attacked by enemy Priest from any of its possible moves
                is_safe = True
                for move_q, move_r in priest_moves:
                    if (move_q, move_r) not in blocked_hexes:
                        if can_kill_piece("Priest", move_q, move_r, {"q": q, "r": r}, blocked_hexes, pieces):
                            is_safe = False
                            break
                if is_safe:
                    safe_positions.append((q, r))
    
    if not safe_positions:
        logger.warning("Could not find safe position for player Priest")
        return None
    
    player_priest_q, player_priest_r = random.choice(safe_positions)
    player_priest = {
        "class": "Priest",
        "label": "P",
        "color": "#556b2f",
        "side": "player",
        "q": player_priest_q,
        "r": player_priest_r
    }
    pieces.append(player_priest)
    logger.info(f"Placed player Priest at ({player_priest_q},{player_priest_r})")
    
    # 4. Add enemy pieces to make puzzle more interesting
    # Add extra enemy pieces that can't immediately kill player pieces
    enemy_classes = ["Guardian", "Hunter"]
    if args and args.allow_bloodwarden:
        enemy_classes.append("BloodWarden")
    
    num_extra_enemies = random.randint(
        args.min_extra_enemies if args else 1,
        args.max_extra_enemies if args else 2
    )
    
    for _ in range(num_extra_enemies):
        enemy_class = random.choice(enemy_classes)
        valid_positions = []
        
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q + r) <= radius:
                    if (q, r) in blocked_hexes:
                        continue
                    if any(p["q"] == q and p["r"] == r for p in pieces):
                        continue
                    
                    # Create a temporary enemy piece at this position
                    temp_enemy = {
                        "class": enemy_class,
                        "label": "G" if enemy_class == "Guardian" else ("H" if enemy_class == "Hunter" else "BW"),
                        "color": "#dc143c",
                        "side": "enemy",
                        "q": q,
                        "r": r
                    }
                    
                    # Check if any player piece would be vulnerable
                    can_kill = False
                    for p in pieces:
                        if p["side"] == "player":
                            if can_kill_piece(enemy_class, q, r, p, blocked_hexes, pieces):
                                can_kill = True
                                break
                    
                    if not can_kill:
                        valid_positions.append((q, r))
        
        if valid_positions:
            pos_q, pos_r = random.choice(valid_positions)
            enemy = {
                "class": enemy_class,
                "label": "G" if enemy_class == "Guardian" else ("H" if enemy_class == "Hunter" else "BW"),
                "color": "#dc143c",
                "side": "enemy",
                "q": pos_q,
                "r": pos_r
            }
            pieces.append(enemy)
            logger.info(f"Added {enemy_class} at ({pos_q},{pos_r})")
    
    # Create final scenario
    scenario = {
        "name": "Puzzle Scenario",
        "subGridRadius": radius,
        "blockedHexes": [{"q": q, "r": r} for (q, r) in blocked_hexes],
        "pieces": pieces
    }
    
    return scenario

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
        radius = args.radius_min
        logger.info(f"Using fixed grid radius: {radius}")
    
    # Generate all possible hex coordinates within radius
    coords = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                coords.add((q, r))
    
    # Generate blocked hexes
    blocked_hexes = set()
    if args.randomize_blocked:
        block_count = random.randint(args.min_blocked, args.max_blocked)
        logger.info(f"Using randomized blocked hex count: {block_count}")
        if block_count > 0:
            blocked_hexes = set(random.sample(list(coords), block_count))
            logger.info(f"Created {len(blocked_hexes)} blocked hexes")
            if blocked_hexes:
                logger.debug(f"Blocked hex coordinates: {blocked_hexes}")
    
    # Generate puzzle using forced mate approach
    scenario = generate_forced_mate_puzzle(radius, blocked_hexes, args)
    if scenario is None:
        logger.warning("Failed to generate forced mate puzzle")
        return None
    
    # Add attempt number
    scenario["attempt_number"] = attempt_number + 1
    
    generation_time = time.time() - start_time
    logger.info(f"Scenario generation completed in {generation_time:.2f} seconds")
    
    # Log piece positions for reference
    player_pieces = [p for p in scenario["pieces"] if p["side"] == "player"]
    enemy_pieces = [p for p in scenario["pieces"] if p["side"] == "enemy"]
    
    logger.info("Initial piece positions:")
    for p in player_pieces:
        logger.info(f"  PLAYER {p['class']} at ({p['q']},{p['r']})")
    for e in enemy_pieces:
        logger.info(f"  ENEMY {e['class']} at ({e['q']},{e['r']})")
    
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
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Track statistics
    start_time = time.time()
    found_any = False
    attempt = 0
    valid_scenarios = []
    failed_reasons = {
        "not_forced": 0,
        "multiple_solutions": 0,
        "no_valid_lines": 0,
        "error": 0,
        "not_solvable": 0
    }
    
    while attempt < args.max_attempts:
        logger.info(f"\n=== Starting attempt {attempt + 1}/{args.max_attempts} ===")
        
        try:
            # Generate a random scenario with attempt number
            scenario = generate_random_scenario(args, attempt)
            if scenario is None:
                logger.warning("Failed to generate scenario")
                failed_reasons["error"] += 1
                attempt += 1
                continue
            
            # Check if it's solvable in exactly 2 turns
            if not is_solvable_in_two_turns(scenario):
                logger.info("Scenario is not solvable in exactly 2 turns")
                failed_reasons["not_solvable"] += 1
                attempt += 1
                continue
            
            # Check if it's a valid "forced mate in 2" puzzle
            if is_forced_mate_in_2(scenario):
                valid_scenarios.append(scenario)
                logger.info(f"Found valid puzzle on attempt {attempt+1}!")
                
                # Save to output file
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(args.output), exist_ok=True)
                    
                    # Load existing scenarios if file exists
                    existing_scenarios = []
                    if os.path.exists(args.output):
                        with open(args.output, "r", encoding="utf-8") as f:
                            existing_scenarios = yaml.safe_load(f) or []
                    
                    # Add new scenario
                    existing_scenarios.append(scenario)
                    
                    # Save all scenarios
                    with open(args.output, "w", encoding="utf-8") as f:
                        yaml.dump(existing_scenarios, f, sort_keys=False)
                    logger.info(f"Puzzle saved to {args.output}")
                except Exception as e:
                    logger.error(f"Failed to save puzzle: {str(e)}")
                    failed_reasons["error"] += 1
                    attempt += 1
                    continue
                
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
                print(f"Puzzle saved to {args.output}")
                
                found_any = True
                break
            else:
                logger.info("Scenario is not a valid forced mate puzzle")
                failed_reasons["not_forced"] += 1
        
        except Exception as e:
            logger.error(f"Error during attempt {attempt+1}: {str(e)}")
            failed_reasons["error"] += 1
        
        attempt += 1
    
    total_time = time.time() - start_time
    
    if found_any:
        logger.info(f"SUCCESS: Generated a valid puzzle in {total_time:.2f} seconds after {attempt+1} attempts")
        logger.info(f"Failure stats: not forced={failed_reasons['not_forced']}, "
                   f"multiple solutions={failed_reasons['multiple_solutions']}, "
                   f"no valid lines={failed_reasons['no_valid_lines']}, "
                   f"not solvable={failed_reasons['not_solvable']}, "
                   f"errors={failed_reasons['error']}")
        
        print(f"\nSuccess! Puzzle generation completed in {total_time:.2f} seconds after {attempt+1} attempts")
    else:
        logger.warning(f"FAILED: Could not generate a valid puzzle after {args.max_attempts} attempts ({total_time:.2f} seconds)")
        logger.warning(f"Failure stats: not forced={failed_reasons['not_forced']}, "
                      f"multiple solutions={failed_reasons['multiple_solutions']}, "
                      f"no valid lines={failed_reasons['no_valid_lines']}, "
                      f"not solvable={failed_reasons['not_solvable']}, "
                      f"errors={failed_reasons['error']}")
        
        print(f"\nNo valid puzzle found after {args.max_attempts} attempts.")
        print(f"Time elapsed: {total_time:.2f} seconds")
        print(f"Failure statistics:")
        print(f"  - Not forced mate: {failed_reasons['not_forced']}")
        print(f"  - Multiple solutions: {failed_reasons['multiple_solutions']}")
        print(f"  - No valid lines: {failed_reasons['no_valid_lines']}")
        print(f"  - Not solvable in 2 turns: {failed_reasons['not_solvable']}")
        print(f"  - Errors: {failed_reasons['error']}")

def is_solvable_in_two_turns(scenario):
    """
    Check if a puzzle is solvable in exactly 2 turns.
    Returns True if:
    1. Player can force a win in 2 turns
    2. Enemy cannot prevent the win
    3. Player cannot win in 1 turn
    """
    # Get pieces data
    with open("data/pieces.yaml", "r") as f:
        pieces_data = yaml.safe_load(f)
    
    # Convert blocked hexes to set for faster lookup
    blocked_hexes = {(bh["q"], bh["r"]) for bh in scenario["blockedHexes"]}
    
    # Get player and enemy pieces
    pieces = scenario["pieces"]
    player_pieces = [p for p in pieces if p["side"] == "player"]
    enemy_pieces = [p for p in pieces if p["side"] == "enemy"]
    
    # Find enemy Priest
    enemy_priest = next(p for p in enemy_pieces if p["class"] == "Priest")
    
    # Check if player can win in 1 turn
    for piece in player_pieces:
        if can_kill_piece(piece["class"], piece["q"], piece["r"], enemy_priest, blocked_hexes, pieces):
            logger.info("Puzzle is too easy - can win in 1 turn")
            return False
    
    # Find all possible moves for enemy Priest
    priest_moves = []
    for q in range(-scenario["subGridRadius"], scenario["subGridRadius"] + 1):
        for r in range(-scenario["subGridRadius"], scenario["subGridRadius"] + 1):
            if abs(q + r) <= scenario["subGridRadius"]:
                if can_reach_in_moves(enemy_priest, q, r, 1, blocked_hexes, pieces):
                    priest_moves.append((q, r))
    
    # For each player piece with ranged attack
    ranged_attackers = []
    for piece in player_pieces:
        class_data = pieces_data["classes"][piece["class"]]
        for action_name, action_data in class_data["actions"].items():
            if action_data["action_type"] in ["single_target_attack", "multi_target_attack"]:
                attack_range = action_data.get("range", 0)
                if attack_range > 1:
                    ranged_attackers.append(piece)
                    break
    
    if not ranged_attackers:
        logger.info("No ranged attackers found")
        return False
    
    # For each ranged attacker
    for attacker in ranged_attackers:
        # Check if attacker can hit all possible Priest positions
        can_hit_all = True
        for move_q, move_r in priest_moves:
            # Check if attacker can reach a position to hit the Priest
            can_reach_and_hit = False
            for q in range(-scenario["subGridRadius"], scenario["subGridRadius"] + 1):
                for r in range(-scenario["subGridRadius"], scenario["subGridRadius"] + 1):
                    if abs(q + r) <= scenario["subGridRadius"]:
                        # Skip if position is blocked or occupied
                        if (q, r) in blocked_hexes:
                            continue
                        if any(p["q"] == q and p["r"] == q and not p.get("dead", False) for p in pieces):
                            continue
                        
                        # Check if we can reach this position in 1 move
                        if can_reach_in_moves(attacker, q, r, 1, blocked_hexes, pieces):
                            # Create a temporary piece at this position
                            temp_piece = {
                                "class": attacker["class"],
                                "q": q,
                                "r": r
                            }
                            # Check if we can hit the Priest from here
                            if can_kill_piece(
                                attacker["class"], q, r,
                                {"q": move_q, "r": move_r}, blocked_hexes, pieces
                            ):
                                can_reach_and_hit = True
                                break
                if can_reach_and_hit:
                    break
            
            if not can_reach_and_hit:
                can_hit_all = False
                break
        
        if can_hit_all:
            logger.info(f"Found winning strategy with {attacker['class']}")
            return True
    
    logger.info("No winning strategy found")
    return False

if __name__ == "__main__":
    main()
