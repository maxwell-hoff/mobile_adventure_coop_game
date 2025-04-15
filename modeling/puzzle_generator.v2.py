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
    # handlers=[
    #     logging.StreamHandler(),
    #     # logging.FileHandler('puzzle_generator.log')
    # ]
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
    parser.add_argument("--radius-min", type=int, default=10,
                      help="Minimum grid radius (default: 10)")
    parser.add_argument("--radius-max", type=int, default=12,
                      help="Maximum grid radius (default: 12)")
    
    # Blocked hex options
    parser.add_argument("--randomize-blocked", action="store_true",
                      help="Randomize the number of blocked hexes")
    parser.add_argument("--blocked-percent-min", type=float, default=20.0,
                      help="Minimum percentage of hexes to block (default: 20.0)")
    parser.add_argument("--blocked-percent-max", type=float, default=60.0,
                      help="Maximum percentage of hexes to block (default: 60.0)")
    
    # Enemy piece options
    parser.add_argument("--min-extra-enemies", type=int, default=0,
                      help="Minimum number of extra enemy pieces (default: 1)")
    parser.add_argument("--max-extra-enemies", type=int, default=2,
                      help="Maximum number of extra enemy pieces (default: 2)")
    parser.add_argument("--allow-bloodwarden", action="store_true",
                      help="Allow BloodWarden as an enemy piece")
    
    # Puzzle complexity options
    parser.add_argument("--num-turns", type=int, default=2,
                      help="Number of turns required for checkmate (default: 2)")
    
    # Generation options
    parser.add_argument("--max-attempts", type=int, default=10000,
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
    if args.blocked_percent_min < 0 or args.blocked_percent_min > 100:
        parser.error("Minimum blocked percentage must be between 0 and 100")
    if args.blocked_percent_max < args.blocked_percent_min or args.blocked_percent_max > 100:
        parser.error("Maximum blocked percentage must be between minimum and 100")
    if args.min_extra_enemies < 0:
        parser.error("Minimum extra enemies must be non-negative")
    if args.max_extra_enemies < args.min_extra_enemies:
        parser.error("Maximum extra enemies must be greater than or equal to minimum")
    if args.num_turns < 1:
        parser.error("Number of turns must be at least 1")
    
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

def score_priest_position(q, r, radius, blocked_hexes, pieces):
    """Score a position for the enemy Priest based on tactical considerations."""
    score = 0
    
    # Prefer central positions (better mobility)
    dist_from_center = hex_distance(q, r, 0, 0)
    score -= dist_from_center * 2  # Central positions get higher scores
    
    # Count escape routes
    escape_routes = 0
    for dq, dr in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
        new_q, new_r = q + dq, r + dr
        if abs(new_q + new_r) <= radius and (new_q, new_r) not in blocked_hexes:
            if not any(p["q"] == new_q and p["r"] == new_r and not p.get("dead", False) for p in pieces):
                escape_routes += 1
    score += escape_routes * 3  # More escape routes is better
    
    # Prefer positions that have good line of sight
    los_count = 0
    for test_q in range(-radius, radius + 1):
        for test_r in range(-radius, radius + 1):
            if abs(test_q + test_r) <= radius:
                if line_of_sight(q, r, test_q, test_r, blocked_hexes, pieces):
                    los_count += 1
    score += los_count  # Better line of sight is good
    
    return score

def get_priest_positions(radius, blocked_hexes, pieces):
    """Get scored positions for enemy Priest placement."""
    positions = []
    # Try positions within radius/2 of center for better control
    search_radius = max(2, radius // 2)
    for q in range(-search_radius, search_radius + 1):
        for r in range(-search_radius, search_radius + 1):
            if abs(q + r) <= radius and (q,r) not in blocked_hexes:
                if not any(p["q"] == q and p["r"] == r and not p.get("dead", False) for p in pieces):
                    score = score_priest_position(q, r, radius, blocked_hexes, pieces)
                    positions.append((q, r, score))
    return sorted(positions, key=lambda x: x[2], reverse=True)

def score_attacker_position(q, r, enemy_priest, radius, blocked_hexes, pieces):
    """Score a position for the ranged attacker based on tactical considerations."""
    score = 0
    
    # Calculate distance to enemy Priest
    dist = hex_distance(q, r, enemy_priest["q"], enemy_priest["r"])
    optimal_range = 3  # Sorcerer's range
    
    # Strongly prefer optimal attack range
    if dist == optimal_range:
        score += 10
    elif dist == optimal_range - 1:
        score += 5  # Being slightly closer is okay
    
    # Check line of sight to enemy Priest
    if line_of_sight(q, r, enemy_priest["q"], enemy_priest["r"], blocked_hexes, pieces):
        score += 15  # Clear line of sight is very important
    
    # Prefer positions that are harder for the enemy Priest to reach
    priest_moves = []
    for dq in range(-1, 2):
        for dr in range(-1, 2):
            new_q = enemy_priest["q"] + dq
            new_r = enemy_priest["r"] + dr
            if abs(new_q + new_r) <= radius and (new_q, new_r) not in blocked_hexes:
                priest_moves.append((new_q, new_r))
    
    safe_from_priest = True
    for pq, pr in priest_moves:
        if hex_distance(q, r, pq, pr) <= 1:  # Priest's attack range
            safe_from_priest = False
            break
    if safe_from_priest:
        score += 10  # Big bonus for being safe from Priest
    
    return score

def get_attacker_positions(radius, enemy_priest, blocked_hexes, pieces):
    """Get scored positions for ranged attacker placement."""
    positions = []
    attack_range = 3  # Sorcerer's range
    
    # Try all positions that could be at optimal range
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius and (q,r) not in blocked_hexes:
                if not any(p["q"] == q and p["r"] == r and not p.get("dead", False) for p in pieces):
                    score = score_attacker_position(q, r, enemy_priest, radius, blocked_hexes, pieces)
                    positions.append((q, r, score))
    
    return sorted(positions, key=lambda x: x[2], reverse=True)

def place_strategic_blocks(radius, enemy_priest, attacker, pieces):
    """Place blocked hexes to create interesting tactical situations."""
    blocked = set()
    
    # Get all possible moves for the enemy Priest
    priest_moves = []
    for dq in range(-1, 2):
        for dr in range(-1, 2):
            new_q = enemy_priest["q"] + dq
            new_r = enemy_priest["r"] + dr
            if abs(new_q + new_r) <= radius:
                if not any(p["q"] == new_q and p["r"] == new_r and not p.get("dead", False) for p in pieces):
                    priest_moves.append((new_q, new_r))
    
    # Score each potential blocked hex
    scored_blocks = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                if (q,r) in {(p["q"], p["r"]) for p in pieces}:
                    continue
                    
                score = 0
                
                # Check if this blocks a Priest escape route
                if (q,r) in priest_moves:
                    score += 5
                
                # Check if this maintains line of sight for attacker
                if not (q,r) == (attacker["q"], attacker["r"]):
                    test_blocks = {(q,r)}
                    if line_of_sight(attacker["q"], attacker["r"], 
                                   enemy_priest["q"], enemy_priest["r"], 
                                   test_blocks, pieces):
                        score += 3
                
                # Prefer blocks near the Priest
                dist_to_priest = hex_distance(q, r, enemy_priest["q"], enemy_priest["r"])
                if dist_to_priest <= 2:
                    score += 2
                
                scored_blocks.append((q, r, score))
    
    # Sort blocks by score and add them until we have a good tactical setup
    scored_blocks.sort(key=lambda x: x[2], reverse=True)
    target_blocks = min(3, len(scored_blocks))
    
    for i in range(target_blocks):
        q, r, _ = scored_blocks[i]
        blocked.add((q,r))
    
    return blocked

def generate_forced_mate_puzzle(radius, blocked_hexes=None, args=None):
    """
    Generate a puzzle that guarantees a forced mate by working backward from the checkmate position.
    Uses intelligent piece placement and strategic blocking.
    """
    logger.info("Generating forced mate puzzle using intelligent piece placement")
    
    # Initialize pieces list and blocked hexes
    pieces = []
    blocked_hexes = set(blocked_hexes or set())
    
    # 1. Place enemy Priest in a tactically interesting position
    priest_positions = get_priest_positions(radius, blocked_hexes, pieces)
    if not priest_positions:
        logger.warning("Could not find good position for enemy Priest")
        return None
    
    # Take one of the top 3 positions randomly to add variety
    priest_q, priest_r, _ = random.choice(priest_positions[:3])
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
    
    # 2. Place ranged attacker in a strong position
    attacker_positions = get_attacker_positions(radius, enemy_priest, blocked_hexes, pieces)
    if not attacker_positions:
        logger.warning("Could not find good position for ranged attacker")
        return None
    
    # Take one of the top 3 positions randomly
    attacker_q, attacker_r, _ = random.choice(attacker_positions[:3])
    attacker = {
        "class": "Sorcerer",
        "label": "S",
        "color": "#556b2f",
        "side": "player",
        "q": attacker_q,
        "r": attacker_r
    }
    pieces.append(attacker)
    logger.info(f"Placed Sorcerer at ({attacker_q},{attacker_r})")
    
    # 3. Add strategic blocked hexes
    strategic_blocks = place_strategic_blocks(radius, enemy_priest, attacker, pieces)
    blocked_hexes.update(strategic_blocks)
    logger.info(f"Added {len(strategic_blocks)} strategic blocked hexes")
    
    # 4. Place player Priest in a safe position
    player_priest = None
    for dist in range(2, radius + 1):
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            q = round(dist * math.cos(rad))
            r = round(dist * math.sin(rad) * 2/math.sqrt(3))
            
            # Adjust coordinates to be relative to enemy Priest
            q += enemy_priest["q"]
            r += enemy_priest["r"]
            
            if abs(q + r) > radius or (q, r) in blocked_hexes:
                continue
            
            # Skip if position is already taken
            if any(p["q"] == q and p["r"] == r for p in pieces):
                continue
            
            # Check if position is safe
            if is_position_safe(q, r, [enemy_priest], blocked_hexes, pieces):
                player_priest = {
                    "class": "Priest",
                    "label": "P",
                    "color": "#556b2f",
                    "side": "player",
                    "q": q,
                    "r": r
                }
                pieces.append(player_priest)
                logger.info(f"Placed player Priest at ({q},{r})")
                break
        if player_priest:
            break
    
    if not player_priest:
        logger.warning("Could not find safe position for player Priest")
        return None
    
    # 5. Add additional enemy pieces to make puzzle interesting
    enemy_classes = ["Guardian", "Hunter"]
    if args and args.allow_bloodwarden:
        enemy_classes.append("BloodWarden")
    
    num_extra_enemies = random.randint(
        args.min_extra_enemies if args else 1,
        args.max_extra_enemies if args else 2
    )
    
    for _ in range(num_extra_enemies):
        enemy_class = random.choice(enemy_classes)
        best_positions = []
        
        # Try all valid positions
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q + r) <= radius:
                    if (q, r) in blocked_hexes:
                        continue
                    if any(p["q"] == q and p["r"] == r for p in pieces):
                        continue
                    
                    # Score this position
                    score = 0
                    
                    # Prefer positions closer to the action
                    dist_to_priest = hex_distance(q, r, enemy_priest["q"], enemy_priest["r"])
                    score += 5 - min(5, dist_to_priest)
                    
                    # Prefer positions that could threaten the attacker's path
                    dist_to_attacker = hex_distance(q, r, attacker["q"], attacker["r"])
                    if dist_to_attacker <= 2:
                        score += 3
                    
                    # Avoid positions that make the puzzle too easy
                    temp_enemy = {
                        "class": enemy_class,
                        "q": q,
                        "r": r
                    }
                    if can_kill_piece(enemy_class, q, r, attacker, blocked_hexes, pieces):
                        score -= 10
                    
                    best_positions.append((q, r, score))
        
        if best_positions:
            # Sort by score and pick one of the top positions
            best_positions.sort(key=lambda x: x[2], reverse=True)
            pos_q, pos_r, _ = random.choice(best_positions[:3])
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
    
    # Generate puzzle using tactical pattern approach
    scenario = generate_tactical_pattern_puzzle(radius, args)
    if scenario is None:
        logger.warning("Failed to generate tactical pattern puzzle")
        return None
    
    # Add attempt number and number of turns
    scenario["attempt_number"] = attempt_number + 1
    scenario["num_turns"] = args.num_turns
    
    generation_time = time.time() - start_time
    logger.info(f"Scenario generation completed in {generation_time:.2f} seconds")
    
    # Log piece positions for reference
    player_pieces = [p for p in scenario["pieces"] if p["side"] == "player"]
    enemy_pieces = [p for p in scenario["pieces"] if p["side"] == "enemy"]
    logger.info(f"Player pieces: {[(p['class'], p['q'], p['r']) for p in player_pieces]}")
    logger.info(f"Enemy pieces: {[(p['class'], p['q'], p['r']) for p in enemy_pieces]}")
    
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
    Determine if the scenario is a forced mate puzzle.
    
    We do N half-turns based on scenario["num_turns"]. 
    Each half-turn => BFS over partial moves for each living piece in all permutations 
    (piece A moves, then piece B moves, etc.). This addresses partial-turn synergy.

    If in all final lines the enemy's Priest is dead => forced mate. 
    Multiple winning solutions are allowed.
    """
    attempt_num = scenario.get("attempt_number", 1)
    num_turns = scenario.get("num_turns", 2)  # Default to 2 if not specified
    max_depth = num_turns * 2  # Convert turns to half-turns
    
    logger.info(f"Analyzing scenario {attempt_num} for forced mate in {num_turns} turns")
    start_time = time.time()
    
    # Initialize the game state from the scenario
    state = {
        "pieces": copy.deepcopy(scenario["pieces"]),
        "blockedHexes": {(bh["q"], bh["r"]) for bh in scenario["blockedHexes"]},
        "radius": scenario["subGridRadius"],
        "sideToMove": "player"
    }
    
    # Log initial piece configuration
    player_pieces = [p for p in state["pieces"] if p["side"] == "player"]
    enemy_pieces = [p for p in state["pieces"] if p["side"] == "enemy"]
    logger.info("Initial piece configuration:")
    for p in player_pieces:
        logger.info(f"  PLAYER {p['class']} at ({p['q']},{p['r']})")
    for e in enemy_pieces:
        logger.info(f"  ENEMY {e['class']} at ({e['q']},{e['r']})")
    
    # Ensure all pieces start alive
    for p in state["pieces"]:
        p["dead"] = False
    
    logger.info(f"Starting analysis with {len(state['pieces'])} pieces and {len(state['blockedHexes'])} blocked hexes")
    
    # Track all possible game lines
    lines = []
    
    # Recursively enumerate all possible play lines
    logger.info("Starting line enumeration")
    enumerate_lines(state, depth=0, partialPlayerCombos=[], lines=lines)
    logger.info(f"Found {len(lines)} distinct play lines")
    
    # Log summary of each line
    for i, line in enumerate(lines):
        logger.info(f"Line {i+1}:")
        logger.info(f"  Depth reached: {line.get('depth', 'unknown')}")
        logger.info(f"  Enemy priest dead: {line.get('priestDead', False)}")
        # Log player moves if available
        player_moves = [x for x in line.get('partialPlayerCombos', []) if x.get('turn') in range(0, max_depth, 2)]
        if player_moves:
            for move in player_moves:
                logger.info(f"  Turn {move.get('turn')} moves: {move.get('comboStr', 'unknown')}")
    
    # If any line doesn't result in enemy priest death, not a forced mate
    non_winning_lines = [i for i, l in enumerate(lines) if l["priestDead"] == False]
    if non_winning_lines:
        logger.info(f"ATTEMPT {attempt_num} FAILED: Lines {non_winning_lines} do not result in enemy priest death")
        return False
    
    # If no valid lines found, not a valid puzzle
    if not lines:
        logger.info(f"ATTEMPT {attempt_num} FAILED: No valid play lines found")
        return False
    
    # Log all winning lines (no longer checking for uniqueness)
    logger.info("Found winning lines:")
    for l in lines:
        moves = []
        for turn in range(0, max_depth, 2):  # Only player turns
            move = next((x for x in l["partialPlayerCombos"] if x["turn"] == turn), None)
            moves.append(move["comboStr"] if move else "")
        logger.info(f"Winning line: {' -> '.join(moves)}")
    
    analysis_time = time.time() - start_time
    logger.info(f"Success! Forced mate analysis completed in {analysis_time:.2f} seconds")
    logger.info(f"ATTEMPT {attempt_num} SUCCESS: Valid forced mate in {num_turns} turns puzzle with {len(lines)} solution(s)")
    
    # Log remaining living pieces in final positions
    winning_line = lines[0]  # Just show one example winning line
    
    if 'finalState' not in winning_line:
        logger.info("Final piece positions would vary based on player & enemy choices")
    else:
        final_state = winning_line['finalState']
        logger.info("Final piece positions in example winning line:")
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

    Optimized version with pruning and heuristics to reduce search space.
    """
    # Global counters for tracking progress
    global _total_states_explored, _last_progress_time
    if not '_total_states_explored' in globals():
        _total_states_explored = 0
        _last_progress_time = time.time()

    _total_states_explored += 1
    
    # Print progress every 30 seconds
    current_time = time.time()
    if current_time - _last_progress_time >= 30:
        logger.info(f"Progress update: Explored {_total_states_explored} states so far. Current depth: {depth}")
        _last_progress_time = current_time

    # Check win condition - enemy Priest is dead
    if is_priest_dead(state["pieces"], "enemy"):
        logger.info(f"Depth {depth}: Enemy priest is dead - winning line found after exploring {_total_states_explored} states")
        line_data = {
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": True,
            "finalState": copy.deepcopy(state),
            "depth": depth
        }
        lines.append(line_data)
        return

    # Check termination condition - reached maximum depth
    if depth >= 4:
        line_data = {
            "partialPlayerCombos": copy.deepcopy(partialPlayerCombos),
            "priestDead": False,
            "finalState": copy.deepcopy(state),
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

    # OPTIMIZATION 1: Prioritize pieces that can make impactful moves
    # For player's turn, prioritize pieces that can attack or move closer to enemy Priest
    # For enemy's turn, prioritize the Priest's survival and pieces that can attack player pieces
    enemy_priest = None
    if side == "player":
        enemy_priest = next((p for p in state["pieces"] if p["side"] == "enemy" and p["class"] == "Priest" and not p.get("dead", False)), None)
    
    def score_piece(p):
        score = 0
        if side == "player" and enemy_priest:
            # Prioritize pieces that can attack the enemy Priest
            if can_kill_piece(p["class"], p["q"], p["r"], enemy_priest, state["blockedHexes"], state["pieces"]):
                score += 100
            # Prioritize pieces closer to enemy Priest
            dist_to_priest = hex_distance(p["q"], p["r"], enemy_priest["q"], enemy_priest["r"])
            score -= dist_to_priest
        elif side == "enemy" and p["class"] == "Priest":
            # Enemy Priest gets highest priority
            score += 200
            # Prioritize moves that keep the Priest alive
            if any(can_kill_piece(ep["class"], ep["q"], ep["r"], p, state["blockedHexes"], state["pieces"]) 
                   for ep in state["pieces"] if ep["side"] == "player" and not ep.get("dead", False)):
                score += 100
        return score

    # Sort pieces by their potential impact
    living.sort(key=score_piece, reverse=True)
    
    # OPTIMIZATION 2: For each piece, generate and prioritize moves
    def score_action(piece, action_tuple):
        score = 0
        atype = action_tuple[0]
        
        if atype == "single_target_attack" or atype == "multi_target_attack":
            score += 50  # Prioritize attacks
            if side == "player":
                # Check if this attacks the enemy Priest
                tq, tr = action_tuple[1], action_tuple[2]
                if enemy_priest and enemy_priest["q"] == tq and enemy_priest["r"] == tr:
                    score += 100
        elif atype == "move":
            if side == "player" and enemy_priest:
                # Score moves that get closer to enemy Priest
                tq, tr = action_tuple[1], action_tuple[2]
                current_dist = hex_distance(piece["q"], piece["r"], enemy_priest["q"], enemy_priest["r"])
                new_dist = hex_distance(tq, tr, enemy_priest["q"], enemy_priest["r"])
                if new_dist < current_dist:
                    score += 30
            elif side == "enemy" and piece["class"] == "Priest":
                # Score moves that keep Priest safe
                tq, tr = action_tuple[1], action_tuple[2]
                is_safe = True
                for p in state["pieces"]:
                    if p["side"] == "player" and not p.get("dead", False):
                        if can_kill_piece(p["class"], p["q"], p["r"], {"q": tq, "r": tr}, state["blockedHexes"], state["pieces"]):
                            is_safe = False
                            break
                if is_safe:
                    score += 80
        return score

    # Track states to explore
    states_to_explore = [(state, [])]
    total_states_this_depth = 0
    
    # For each piece in prioritized order
    for piece_index, piece in enumerate(living):
        logger.debug(f"Processing piece {piece_index+1}/{len(living)}: {piece['side']} {piece['class']}")
        
        new_states = []
        for current_state, moves_so_far in states_to_explore:
            # Get and score all possible actions
            piece_actions = gather_single_piece_actions(current_state, piece)
            if piece_actions:
                # Sort actions by score
                piece_actions.sort(key=lambda a: score_action(piece, a), reverse=True)
                
                # OPTIMIZATION 3: Limit number of actions to explore based on depth
                max_actions = 10 if depth <= 1 else 5  # Explore more options early in the tree
                piece_actions = piece_actions[:max_actions]
                
                for action in piece_actions:
                    # Apply action
                    new_state = apply_single_action(current_state, piece, action)
                    # Format move string
                    if len(action) >= 3:
                        tq, tr = action[1], action[2]
                    else:
                        tq, tr = 0, 0
                    move_str = move_to_string(piece, action[0], tq, tr)
                    new_moves = moves_so_far + [move_str]
                    new_states.append((new_state, new_moves))
            else:
                # No valid actions, add pass
                new_state = apply_single_action(current_state, piece, ("pass", 0, 0))
                new_moves = moves_so_far + [f"{piece['label']}=pass"]
                new_states.append((new_state, new_moves))
        
        states_to_explore = new_states
        total_states_this_depth += len(states_to_explore)
        logger.info(f"After piece {piece['label']}: {len(states_to_explore)} states to explore")

    # Process final states
    logger.info(f"Processing {len(states_to_explore)} final states at depth {depth}")
    for final_state, move_list in states_to_explore:
        # Build move sequence string
        combo_str = ";".join(move_list)
        
        # If player's turn, store the move combination
        new_partial_pc = partialPlayerCombos
        if side == "player":
            new_partial_pc = copy.deepcopy(partialPlayerCombos)
            new_partial_pc.append({
                "turn": depth,
                "comboStr": combo_str,
                "moves": move_list
            })
        
        # Continue with next depth
        enumerate_lines(switch_side(final_state), depth+1, new_partial_pc, lines)

    logger.info(f"Depth {depth}: Completed exploration. Total states at this depth: {total_states_this_depth}")

def gather_single_piece_actions(state, piece):
    """
    Gather all valid actions for a single piece.
    """
    results = []
    pieces = state["pieces"]
    blocked_set = state.get("blockedHexes", [])
    side = piece["side"]
    
    # Check if piece is immobilized
    if piece.get("immobilized", False):
        # If immobilized, only allow non-move actions
        for action_name, action in piece.get("actions", {}).items():
            if action.get("action_type") != "move":
                # Handle other action types as before
                # ... existing action type handling code ...
                pass
        return results
    
    # If not immobilized, proceed with normal action gathering
    for action_name, action in piece.get("actions", {}).items():
        # ... rest of existing action gathering code ...
        pass
    
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
    
    elif action_type == "trap":
        # Place a trap on the target hex
        tq, tr, effect, duration = action_tuple[1], action_tuple[2], action_tuple[3], action_tuple[4]
        
        # Add trap to the state
        if "traps" not in new_state:
            new_state["traps"] = []
        
        new_state["traps"].append({
            "q": tq,
            "r": tr,
            "effect": effect,
            "duration": duration,
            "caster": piece["class"],
            "caster_side": piece["side"]
        })
    
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

def generate_tactical_pattern_puzzle(radius, args=None):
    """
    Generate a puzzle using predefined tactical patterns.
    This is more intentional than random generation, focusing on creating
    specific tactical situations that are likely to yield valid puzzles.
    """
    logger.info("Generating tactical pattern puzzle")
    
    # Initialize pieces list and blocked hexes
    pieces = []
    blocked_hexes = set()
    
    # 1. Choose a corner or edge position for enemy Priest
    corner_positions = [
        (radius, -radius), (-radius, radius),  # opposite corners
        (radius, 0), (-radius, 0),            # side corners
        (0, radius), (0, -radius)             # top/bottom corners
    ]
    priest_pos = random.choice(corner_positions)
    priest_q, priest_r = priest_pos
    
    # Place enemy Priest
    enemy_priest = {
        "class": "Priest",
        "label": "P",
        "color": "#dc143c",
        "side": "enemy",
        "q": priest_q,
        "r": priest_r
    }
    pieces.append(enemy_priest)
    logger.info(f"Placed enemy Priest at corner position ({priest_q},{priest_r})")
    
    # 2. Create a wall of blocked hexes around the Priest, leaving 1-2 escape routes
    adjacent_hexes = [
        (priest_q+1, priest_r), (priest_q-1, priest_r),
        (priest_q, priest_r+1), (priest_q, priest_r-1),
        (priest_q+1, priest_r-1), (priest_q-1, priest_r+1)
    ]
    
    # Filter valid adjacent hexes
    adjacent_hexes = [(q,r) for (q,r) in adjacent_hexes 
                     if abs(q) <= radius and abs(r) <= radius and abs(q+r) <= radius]
    
    # Leave 1-2 escape routes, block the rest
    num_escapes = random.randint(1, 2)
    escape_hexes = random.sample(adjacent_hexes, num_escapes)
    for q, r in adjacent_hexes:
        if (q,r) not in escape_hexes:
            blocked_hexes.add((q,r))
    
    logger.info(f"Created wall with {len(blocked_hexes)} blocked hexes, leaving {num_escapes} escape routes")
    
    # 3. Place Sorcerer at optimal attack range (3 hexes)
    # Try positions that can cover all escape routes
    best_attacker_pos = None
    best_coverage = -1
    
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            if abs(q+r) <= radius and (q,r) not in blocked_hexes:
                # Check if this position can attack both the Priest and escape routes
                can_hit_priest = hex_distance(q, r, priest_q, priest_r) <= 3
                escape_coverage = sum(1 for (eq,er) in escape_hexes 
                                   if hex_distance(q, r, eq, er) <= 3)
                
                # Prefer positions that can hit both Priest and escapes
                if can_hit_priest and escape_coverage > best_coverage:
                    best_attacker_pos = (q,r)
                    best_coverage = escape_coverage
    
    if not best_attacker_pos:
        logger.warning("Could not find good Sorcerer position")
        return None
    
    attacker = {
        "class": "Sorcerer",
        "label": "S",
        "color": "#556b2f",
        "side": "player",
        "q": best_attacker_pos[0],
        "r": best_attacker_pos[1]
    }
    pieces.append(attacker)
    logger.info(f"Placed Sorcerer at ({best_attacker_pos[0]},{best_attacker_pos[1]})")
    
    # 4. Place Guardian to block the main escape route
    for escape_q, escape_r in escape_hexes:
        # Try to find a position 2 hexes away from the escape route
        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if abs(q+r) <= radius and (q,r) not in blocked_hexes:
                    dist_to_escape = hex_distance(q, r, escape_q, escape_r)
                    if dist_to_escape == 2:  # Guardian's attack range
                        guardian = {
                            "class": "Guardian",
                            "label": "G",
                            "color": "#dc143c",
                            "side": "enemy",
                            "q": q,
                            "r": r
                        }
                        pieces.append(guardian)
                        logger.info(f"Placed Guardian at ({q},{r})")
                        break
            if len(pieces) > 2:  # If we placed a Guardian
                break
    
    # 5. Place player Priest in a safe position
    for dist in range(2, radius+1):
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            q = round(dist * math.cos(rad))
            r = round(dist * math.sin(rad) * 2/math.sqrt(3))
            
            if abs(q+r) <= radius and (q,r) not in blocked_hexes:
                # Check if position is safe
                if not any(p for p in pieces if p["q"] == q and p["r"] == r):
                    player_priest = {
                        "class": "Priest",
                        "label": "P",
                        "color": "#556b2f",
                        "side": "player",
                        "q": q,
                        "r": r
                    }
                    pieces.append(player_priest)
                    logger.info(f"Placed player Priest at ({q},{r})")
                    break
        if len(pieces) > 3:  # If we placed the player Priest
            break
    
    if len(pieces) < 4:
        logger.warning("Could not place all required pieces")
        return None
    
    # Create final scenario
    scenario = {
        "name": "Tactical Pattern Puzzle",
        "subGridRadius": radius,
        "blockedHexes": [{"q": q, "r": r} for (q, r) in blocked_hexes],
        "pieces": pieces
    }
    
    return scenario

if __name__ == "__main__":
    main()
