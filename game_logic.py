from typing import List, Dict, Optional, Tuple, Set
import math

class HexCoordinate:
    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r

    def __eq__(self, other):
        return self.q == other.q and self.r == other.r

    def __hash__(self):
        return hash((self.q, self.r))

class Piece:
    def __init__(self, class_: str, label: str, q: int, r: int, side: str = "player"):
        self.class_ = class_
        self.label = label
        self.q = q
        self.r = r
        self.side = side
        self.immobilized = False
        self.immobilized_turns = 0
        self.dead = False
        self.health = 100
        self.max_health = 100
        self.active_effects: Dict[str, Dict] = {}  # effect_name -> {duration: int, ...}
        self.cooldowns: Dict[str, int] = {}  # action_name -> turns_remaining

class GameState:
    def __init__(self, pieces: List[Piece], blocked_hexes: List[Dict]):
        self.pieces = pieces
        self.blocked_hexes = blocked_hexes
        self.turn_counter = 0

    def get_piece_at(self, q: int, r: int) -> Optional[Piece]:
        for piece in self.pieces:
            if piece.q == q and piece.r == r and not piece.dead:
                return piece
        return None

    def is_hex_blocked(self, q: int, r: int) -> bool:
        return any(h["q"] == q and h["r"] == r for h in self.blocked_hexes)

    def is_hex_occupied(self, q: int, r: int) -> bool:
        return self.get_piece_at(q, r) is not None

    def get_hex_distance(self, start: HexCoordinate, end: HexCoordinate) -> int:
        return (abs(start.q - end.q) + 
                abs(start.q + start.r - end.q - end.r) + 
                abs(start.r - end.r)) // 2

    def has_line_of_sight(self, start: HexCoordinate, end: HexCoordinate, 
                         ignore_piece: Optional[Piece] = None) -> bool:
        # Get all hexes in the line
        hexes = self.get_hexes_in_line(start, end)
        
        # Check each hex for blocking
        for hex_coord in hexes:
            # Skip start and end points
            if hex_coord == start or hex_coord == end:
                continue
                
            # Check for blocking pieces
            piece = self.get_piece_at(hex_coord.q, hex_coord.r)
            if piece and piece != ignore_piece and not piece.dead:
                return False
                
            # Check for blocked hexes
            if self.is_hex_blocked(hex_coord.q, hex_coord.r):
                return False
                
        return True

    def get_hexes_in_line(self, start: HexCoordinate, end: HexCoordinate) -> List[HexCoordinate]:
        """Bresenham's line algorithm for hex grids"""
        N = self.get_hex_distance(start, end)
        results = []
        
        for i in range(N + 1):
            t = 1.0 * i / N
            q = round(start.q * (1.0 - t) + end.q * t)
            r = round(start.r * (1.0 - t) + end.r * t)
            results.append(HexCoordinate(q, r))
            
        return results

    def validate_move(self, piece: Piece, target_q: int, target_r: int, 
                     max_range: int) -> Tuple[bool, str]:
        """Validate a move action"""
        # Check if piece is immobilized
        if piece.immobilized:
            return False, f"{piece.class_} is immobilized and cannot move"

        # Check if target is in range
        distance = self.get_hex_distance(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        )
        if distance > max_range:
            return False, f"Target is out of range (max {max_range})"

        # Check if target is occupied
        if self.is_hex_occupied(target_q, target_r):
            return False, "Target hex is occupied"

        # Check if target is blocked by non-trap obstacle
        if self.is_hex_blocked(target_q, target_r):
            return False, "Target hex is blocked by obstacle"

        # Check line of sight if required
        if not self.has_line_of_sight(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        ):
            return False, "No line of sight to target"

        return True, ""

    def apply_trap_effect(self, piece: Piece, trap: Dict) -> str:
        """Apply trap effect to a piece"""
        if trap["effect"] == "immobilize":
            piece.immobilized = True
            piece.immobilized_turns = trap["duration"]
            return f"{piece.class_} stepped on a trap and is immobilized for {trap['duration']} turns"
        return ""

    def decrement_immobilized_turns(self) -> List[str]:
        """Decrement immobilized turns for all pieces and return messages"""
        messages = []
        for piece in self.pieces:
            if piece.immobilized and piece.immobilized_turns > 0:
                piece.immobilized_turns -= 1
                if piece.immobilized_turns <= 0:
                    piece.immobilized = False
                    messages.append(f"{piece.class_} is no longer immobilized")
        return messages

    def validate_attack(self, piece: Piece, target_q: int, target_r: int, 
                       max_range: int, max_targets: int = 1) -> Tuple[bool, str]:
        """Validate an attack action"""
        # Check if piece is immobilized
        if piece.immobilized:
            return False, f"{piece.class_} is immobilized and cannot attack"

        # Check if target is in range
        distance = self.get_hex_distance(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        )
        if distance > max_range:
            return False, f"Target is out of range (max {max_range})"

        # Check if target has enemy
        target_piece = self.get_piece_at(target_q, target_r)
        if not target_piece or target_piece.side == piece.side:
            return False, "Target hex has no enemy"

        # Check line of sight if required
        if not self.has_line_of_sight(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        ):
            return False, "No line of sight to target"

        return True, ""

    def validate_swap_position(self, piece: Piece, target_q: int, target_r: int, 
                             max_range: int, ally_only: bool = False) -> Tuple[bool, str]:
        """Validate a position swap action"""
        # Check if piece is immobilized
        if piece.immobilized:
            return False, f"{piece.class_} is immobilized and cannot swap positions"

        # Check if target is in range
        distance = self.get_hex_distance(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        )
        if distance > max_range:
            return False, f"Target is out of range (max {max_range})"

        # Check if target has a piece to swap with
        target_piece = self.get_piece_at(target_q, target_r)
        if not target_piece:
            return False, "Target hex has no piece to swap with"

        # Check ally_only constraint
        if ally_only and target_piece.side != piece.side:
            return False, "Can only swap with allies"

        return True, ""

    def validate_pull_push(self, piece: Piece, target_q: int, target_r: int, 
                         max_range: int, distance: int, is_pull: bool) -> Tuple[bool, str]:
        """Validate a pull or push action"""
        # Check if piece is immobilized
        if piece.immobilized:
            return False, f"{piece.class_} is immobilized and cannot {('pull' if is_pull else 'push')}"

        # Check if target is in range
        target_distance = self.get_hex_distance(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(target_q, target_r)
        )
        if target_distance > max_range:
            return False, f"Target is out of range (max {max_range})"

        # Check if target has a piece
        target_piece = self.get_piece_at(target_q, target_r)
        if not target_piece:
            return False, "Target hex has no piece"

        # Calculate destination
        if is_pull:
            # For pull, destination is between piece and target
            dest_q = piece.q + (target_q - piece.q) // 2
            dest_r = piece.r + (target_r - piece.r) // 2
        else:
            # For push, destination is beyond target
            dest_q = target_q + (target_q - piece.q)
            dest_r = target_r + (target_r - piece.r)

        # Check if destination is occupied
        if self.is_hex_occupied(dest_q, dest_r):
            return False, "Destination is occupied"

        # Check if destination is blocked
        if self.is_hex_blocked(dest_q, dest_r):
            return False, "Destination is blocked"

        return True, ""

    def validate_aoe(self, piece: Piece, center_q: int, center_r: int, 
                    max_range: int, radius: int) -> Tuple[bool, str]:
        """Validate an area of effect action"""
        # Check if piece is immobilized
        if piece.immobilized:
            return False, f"{piece.class_} is immobilized and cannot use AOE"

        # Check if center is in range
        distance = self.get_hex_distance(
            HexCoordinate(piece.q, piece.r),
            HexCoordinate(center_q, center_r)
        )
        if distance > max_range:
            return False, f"Center is out of range (max {max_range})"

        return True, ""

    def apply_damage(self, piece: Piece, damage: int) -> str:
        """Apply damage to a piece and return status message"""
        piece.health = max(0, piece.health - damage)
        if piece.health <= 0:
            piece.dead = True
            return f"{piece.class_} ({piece.label}) has been defeated"
        return f"{piece.class_} ({piece.label}) took {damage} damage"

    def decrement_cooldowns(self) -> List[str]:
        """Decrement cooldowns for all pieces and return messages"""
        messages = []
        for piece in self.pieces:
            for action_name in list(piece.cooldowns.keys()):
                piece.cooldowns[action_name] -= 1
                if piece.cooldowns[action_name] <= 0:
                    del piece.cooldowns[action_name]
                    messages.append(f"{piece.class_} ({piece.label}) can use {action_name} again")
        return messages

    def decrement_effects(self) -> List[str]:
        """Decrement effect durations and return messages"""
        messages = []
        for piece in self.pieces:
            for effect_name in list(piece.active_effects.keys()):
                effect = piece.active_effects[effect_name]
                effect["duration"] -= 1
                if effect["duration"] <= 0:
                    del piece.active_effects[effect_name]
                    messages.append(f"{piece.class_} ({piece.label}) is no longer affected by {effect_name}")
        return messages 