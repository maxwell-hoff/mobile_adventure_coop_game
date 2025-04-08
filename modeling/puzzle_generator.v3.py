"""
puzzle_generator.v3.py

An object-oriented puzzle generator that creates tactically interesting puzzles
through intelligent piece placement and difficulty optimization.

Key features:
- Flexible piece placement order based on tactical impact
- Iterative difficulty optimization
- Multiple solution paths allowed
- PPO-based difficulty evaluation
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
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from itertools import combinations
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# PPO imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_training import HexPuzzleEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load piece data
with open("data/pieces.yaml", "r", encoding="utf-8") as f:
    PIECES_DATA = yaml.safe_load(f)

@dataclass
class Position:
    """Represents a position on the hex grid."""
    q: int
    r: int

    def distance_to(self, other: 'Position') -> int:
        """Calculate hex distance to another position."""
        return (abs(self.q - other.q) 
                + abs(self.r - other.r)
                + abs((self.q + self.r) - (other.q + other.r))) // 2

    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.q == other.q and self.r == other.r

@dataclass
class Piece:
    """Represents a game piece with its properties and abilities."""
    class_name: str
    side: str
    position: Optional[Position] = None
    label: str = ""
    color: str = ""
    dead: bool = False
    moved_this_turn: bool = False

    def __post_init__(self):
        if not self.label:
            self.label = self.class_name[0].upper()
        if not self.color:
            self.color = "#556b2f" if self.side == "player" else "#dc143c"

    def to_dict(self) -> dict:
        """Convert piece to dictionary format for YAML output."""
        return {
            "class": self.class_name,
            "label": self.label,
            "color": self.color,
            "side": self.side,
            "q": self.position.q if self.position else None,
            "r": self.position.r if self.position else None
        }

    def can_attack(self, target: 'Piece', board: 'Board') -> bool:
        """Check if this piece can attack the target piece."""
        if not self.position or not target.position or target.dead:
            return False

        piece_data = PIECES_DATA["classes"].get(self.class_name, {})
        for action_name, action in piece_data.get("actions", {}).items():
            atype = action.get("action_type")
            if atype in ["single_target_attack", "multi_target_attack", "aoe"]:
                rng = action.get("range", 1)
                requires_los = action.get("requires_los", False)
                dist = self.position.distance_to(target.position)

                if dist <= rng:
                    if not requires_los or board.has_line_of_sight(
                        self.position, target.position
                    ):
                        return True
        return False

class Board:
    """Represents the game board state."""
    def __init__(self, radius: int):
        self.radius = radius
        self.blocked_positions: Set[Position] = set()
        self.pieces: List[Piece] = []
        self._build_all_positions()

    def _build_all_positions(self):
        """Build set of all valid positions on the board."""
        self.all_positions = set()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                if abs(q + r) <= self.radius:
                    self.all_positions.add(Position(q, r))

    def add_piece(self, piece: Piece) -> bool:
        """Add a piece to the board. Returns False if position is invalid/occupied."""
        if not piece.position:
            return False
        if piece.position not in self.all_positions:
            return False
        if self.is_position_occupied(piece.position):
            return False
        if piece.position in self.blocked_positions:
            return False
        self.pieces.append(piece)
        return True

    def is_position_occupied(self, pos: Position) -> bool:
        """Check if a position is occupied by a piece."""
        return any(p.position == pos and not p.dead for p in self.pieces)

    def get_piece_at(self, pos: Position) -> Optional[Piece]:
        """Get piece at given position, if any."""
        for p in self.pieces:
            if p.position == pos and not p.dead:
                return p
        return None

    def has_line_of_sight(self, start: Position, end: Position) -> bool:
        """Check if there is line of sight between two positions."""
        if start == end:
            return True

        # Calculate all hexes along the line
        N = max(abs(end.q - start.q), 
                abs(end.r - start.r),
                abs((start.q + start.r) - (end.q + end.r)))
        if N == 0:
            return True

        s1 = -start.q - start.r
        s2 = -end.q - end.r
        
        for i in range(1, N):  # Skip start and end points
            t = i / N
            qf = start.q + (end.q - start.q) * t
            rf = start.r + (end.r - start.r) * t
            sf = s1 + (s2 - s1) * t
            
            rq = round(qf)
            rr = round(rf)
            rs = round(sf)
            
            # Fix rounding to maintain q + r + s = 0
            if abs(rq - qf) > abs(rr - rf) and abs(rq - qf) > abs(rs - sf):
                rq = -rr - rs
            elif abs(rr - rf) > abs(rs - sf):
                rr = -rq - rs
            
            pos = Position(rq, rr)
            if pos in self.blocked_positions:
                return False
            if self.is_position_occupied(pos):
                return False
        
        return True

    def get_available_positions(self) -> Set[Position]:
        """Get all positions that aren't blocked or occupied."""
        return {p for p in self.all_positions 
                if p not in self.blocked_positions
                and not self.is_position_occupied(p)}

    def to_scenario_dict(self) -> dict:
        """Convert board state to scenario dictionary format."""
        return {
            "name": "Puzzle Scenario",
            "subGridRadius": self.radius,
            "blockedHexes": [{"q": p.q, "r": p.r} for p in self.blocked_positions],
            "pieces": [p.to_dict() for p in self.pieces]
        }

class PuzzleValidator:
    """Validates puzzle properties and solvability."""
    def __init__(self, board: Board):
        self.board = board

    def is_valid_configuration(self) -> bool:
        """Check if current board configuration is valid."""
        # Check piece counts
        player_priests = sum(1 for p in self.board.pieces 
                           if p.side == "player" and p.class_name == "Priest")
        enemy_priests = sum(1 for p in self.board.pieces
                          if p.side == "enemy" and p.class_name == "Priest")
        
        if player_priests < 1 or enemy_priests != 1:
            return False

        # Check blocked hex percentage
        total_hexes = len(self.board.all_positions)
        blocked_percent = (len(self.board.blocked_positions) / total_hexes) * 100
        if not (20 <= blocked_percent <= 60):
            return False

        return True

    def has_valid_moves(self, side: str) -> bool:
        """Check if given side has any valid moves."""
        side_pieces = [p for p in self.board.pieces if p.side == side and not p.dead]
        if not side_pieces:
            return False

        for piece in side_pieces:
            piece_data = PIECES_DATA["classes"].get(piece.class_name, {})
            for action_name, action in piece_data.get("actions", {}).items():
                if action.get("action_type") == "move":
                    move_range = action.get("range", 1)
                    # Check if piece can move anywhere
                    for pos in self.board.get_available_positions():
                        if piece.position.distance_to(pos) <= move_range:
                            return True
                elif action.get("action_type") in ["single_target_attack", "multi_target_attack"]:
                    # Check if piece can attack any enemy
                    for target in self.board.pieces:
                        if target.side != side and not target.dead:
                            if piece.can_attack(target, self.board):
                                return True
        return False

class DifficultyEvaluator:
    """Evaluates puzzle difficulty through PPO simulation."""
    def __init__(self, model_path: str = "ppo_model.zip"):
        self.model = MaskablePPO.load(model_path)
        # Get observation space dimensions from model
        self.obs_dim = self.model.observation_space.shape[0]

    def _create_env(self, scenario: dict) -> HexPuzzleEnv:
        """Create environment with correct observation space."""
        env = HexPuzzleEnv(
            puzzle_scenario=copy.deepcopy(scenario),
            max_turns=10,
            randomize_positions=False,
            observation_dim=self.obs_dim  # Pass expected dimension
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    def evaluate_difficulty(
        self, 
        scenario: dict,
        num_simulations: int = 100
    ) -> dict:
        """
        Evaluate puzzle difficulty through multiple simulations.
        Returns dict with win rates and difficulty score.
        """
        player_wins = 0
        enemy_wins = 0
        draws = 0

        vec_env = DummyVecEnv([lambda: self._create_env(scenario)])

        for _ in range(num_simulations):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = vec_env.step(action)
                if done:
                    if reward >= 30:  # Priest kill
                        player_wins += 1
                    elif reward <= -30:  # Player's Priest killed
                        enemy_wins += 1
                    else:  # Draw (usually time limit)
                        draws += 1

        results = {
            "player_wins": player_wins,
            "enemy_wins": enemy_wins,
            "draws": draws,
            "total_games": num_simulations,
            "player_win_rate": player_wins / num_simulations,
            "difficulty_score": 1.0 - (player_wins / num_simulations)
        }

        return results

class PuzzleGenerator:
    """Main class for generating puzzles with difficulty optimization."""
    def __init__(
        self,
        radius: int,
        min_blocked_percent: float = 20.0,
        max_blocked_percent: float = 60.0,
        min_player_pieces: int = 2,
        max_player_pieces: int = 5,
        min_enemy_pieces: int = 2,
        max_enemy_pieces: int = 5,
        difficulty_evaluator: Optional[DifficultyEvaluator] = None
    ):
        self.radius = radius
        self.min_blocked_percent = min_blocked_percent
        self.max_blocked_percent = max_blocked_percent
        self.min_player_pieces = min_player_pieces
        self.max_player_pieces = max_player_pieces
        self.min_enemy_pieces = min_enemy_pieces
        self.max_enemy_pieces = max_enemy_pieces
        self.difficulty_evaluator = difficulty_evaluator or DifficultyEvaluator()

    def _create_initial_board(self) -> Board:
        """Create initial board with random blocked hexes."""
        board = Board(self.radius)
        
        # Calculate blocked hex count
        total_hexes = len(board.all_positions)
        min_blocked = int((self.min_blocked_percent / 100.0) * total_hexes)
        max_blocked = int((self.max_blocked_percent / 100.0) * total_hexes)
        num_blocked = random.randint(min_blocked, max_blocked)
        
        # Add random blocked positions
        available = list(board.all_positions)
        for pos in random.sample(available, num_blocked):
            board.blocked_positions.add(pos)
        
        return board

    def _create_piece_pool(self) -> Tuple[List[Piece], List[Piece]]:
        """Create pools of pieces for both sides."""
        def create_side_pieces(side: str, min_count: int, max_count: int) -> List[Piece]:
            count = random.randint(min_count, max_count)
            pieces = [Piece("Priest", side)]  # Always one Priest
            
            # Available classes for each side
            player_classes = ["Warlock", "Sorcerer"]
            enemy_classes = ["Guardian", "BloodWarden", "Hunter"]
            valid_classes = player_classes if side == "player" else enemy_classes
            
            # Add random pieces up to count
            for _ in range(count - 1):
                class_name = random.choice(valid_classes)
                pieces.append(Piece(class_name, side))
            
            return pieces

        player_pieces = create_side_pieces(
            "player", self.min_player_pieces, self.max_player_pieces
        )
        enemy_pieces = create_side_pieces(
            "enemy", self.min_enemy_pieces, self.max_enemy_pieces
        )
        
        return player_pieces, enemy_pieces

    def _score_position_for_piece(
        self,
        piece: Piece,
        pos: Position,
        board: Board
    ) -> float:
        """Score a potential position for a piece based on tactical considerations."""
        score = 0.0
        
        # Distance from center
        center = Position(0, 0)
        dist_from_center = pos.distance_to(center)
        score -= dist_from_center * 0.5  # Prefer central positions
        
        # Count available moves
        available_positions = board.get_available_positions()
        move_options = sum(1 for p in available_positions 
                         if pos.distance_to(p) <= 2)  # Assume range 2
        score += move_options * 0.3
        
        # Count attack opportunities
        if piece.class_name != "Priest":
            potential_targets = [p for p in board.pieces 
                              if p.side != piece.side and not p.dead]
            attack_options = sum(1 for t in potential_targets 
                               if Position(pos.q, pos.r).distance_to(t.position) <= 3)
            score += attack_options * 2.0
        
        # Special considerations for Priests
        if piece.class_name == "Priest":
            # Prefer defensive positions for Priests
            defensive_score = sum(1 for p in board.pieces
                                if p.side == piece.side and not p.dead
                                and Position(pos.q, pos.r).distance_to(p.position) <= 2)
            score += defensive_score * 1.5
        
        return score

    def _optimize_piece_placement(
        self,
        board: Board,
        piece: Piece,
        available_positions: List[Position],
        num_candidates: int = 5
    ) -> Optional[Position]:
        """Find the best position for a piece among available positions."""
        if not available_positions:
            return None

        # Score all positions
        scored_positions = [
            (pos, self._score_position_for_piece(piece, pos, board))
            for pos in available_positions
        ]
        
        # Sort by score and take top candidates
        scored_positions.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_positions[:num_candidates]
        
        # Randomly select from top candidates (with weighted probability)
        total_score = sum(score for _, score in candidates)
        if total_score <= 0:
            return random.choice([pos for pos, _ in candidates])
        
        weights = [score/total_score for _, score in candidates]
        chosen_pos = random.choices(
            [pos for pos, _ in candidates],
            weights=weights,
            k=1
        )[0]
        
        return chosen_pos

    def generate_puzzle(
        self,
        max_attempts: int = 100,
        min_difficulty: float = 0.7
    ) -> Optional[dict]:
        """
        Generate a puzzle with the specified parameters.
        Returns None if no valid puzzle could be generated.
        """
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Create board and piece pools
            board = self._create_initial_board()
            player_pieces, enemy_pieces = self._create_piece_pool()
            
            # Place pieces in order of tactical importance
            all_pieces = (
                [p for p in enemy_pieces if p.class_name == "Priest"] +  # Enemy Priest first
                [p for p in player_pieces if p.class_name != "Priest"] +  # Player attackers
                [p for p in enemy_pieces if p.class_name != "Priest"] +   # Enemy support
                [p for p in player_pieces if p.class_name == "Priest"]    # Player Priest last
            )
            
            placement_success = True
            for piece in all_pieces:
                available = list(board.get_available_positions())
                if not available:
                    placement_success = False
                    break
                
                # Find best position for this piece
                best_pos = self._optimize_piece_placement(board, piece, available)
                if not best_pos:
                    placement_success = False
                    break
                
                # Place piece
                piece.position = best_pos
                if not board.add_piece(piece):
                    placement_success = False
                    break
            
            if not placement_success:
                logger.info("Failed to place all pieces, retrying...")
                continue
            
            # Validate puzzle
            validator = PuzzleValidator(board)
            if not validator.is_valid_configuration():
                logger.info("Invalid puzzle configuration, retrying...")
                continue
            
            # Convert to scenario and evaluate difficulty
            scenario = board.to_scenario_dict()
            evaluation = self.difficulty_evaluator.evaluate_difficulty(scenario)
            
            difficulty = evaluation["difficulty_score"]
            if difficulty < min_difficulty:
                logger.info(f"Puzzle too easy (difficulty={difficulty:.2f}), retrying...")
                continue
            
            # Success! Add evaluation data and return
            scenario["evaluation"] = evaluation
            logger.info(f"Successfully generated puzzle with difficulty {difficulty:.2f}")
            return scenario
        
        logger.warning(f"Failed to generate valid puzzle after {max_attempts} attempts")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate tactical puzzles with difficulty optimization")
    
    # Grid size options
    parser.add_argument("--radius", type=int, default=10,
                      help="Grid radius (default: 10)")
    
    # Blocked hex options
    parser.add_argument("--min-blocked-percent", type=float, default=20.0,
                      help="Minimum percentage of hexes to block (default: 20.0)")
    parser.add_argument("--max-blocked-percent", type=float, default=60.0,
                      help="Maximum percentage of hexes to block (default: 60.0)")
    
    # Piece count options
    parser.add_argument("--min-player-pieces", type=int, default=2,
                      help="Minimum number of player pieces (default: 2)")
    parser.add_argument("--max-player-pieces", type=int, default=5,
                      help="Maximum number of player pieces (default: 5)")
    parser.add_argument("--min-enemy-pieces", type=int, default=2,
                      help="Minimum number of enemy pieces (default: 2)")
    parser.add_argument("--max-enemy-pieces", type=int, default=5,
                      help="Maximum number of enemy pieces (default: 5)")
    
    # Generation options
    parser.add_argument("--max-attempts", type=int, default=100,
                      help="Maximum generation attempts (default: 100)")
    parser.add_argument("--min-difficulty", type=float, default=0.7,
                      help="Minimum acceptable difficulty score (default: 0.7)")
    parser.add_argument("--num-simulations", type=int, default=100,
                      help="Number of simulations for difficulty evaluation (default: 100)")
    parser.add_argument("--output", type=str, default="data/generated_puzzles_v3.yaml",
                      help="Output file path (default: data/generated_puzzles_v3.yaml)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Initialize generator
    generator = PuzzleGenerator(
        radius=args.radius,
        min_blocked_percent=args.min_blocked_percent,
        max_blocked_percent=args.max_blocked_percent,
        min_player_pieces=args.min_player_pieces,
        max_player_pieces=args.max_player_pieces,
        min_enemy_pieces=args.min_enemy_pieces,
        max_enemy_pieces=args.max_enemy_pieces
    )
    
    # Generate puzzle
    puzzle = generator.generate_puzzle(
        max_attempts=args.max_attempts,
        min_difficulty=args.min_difficulty
    )
    
    if puzzle:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Load existing puzzles if any
        existing_puzzles = []
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                existing_puzzles = yaml.safe_load(f) or []
        
        # Add new puzzle
        existing_puzzles.append(puzzle)
        
        # Save all puzzles
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(existing_puzzles, f, sort_keys=False)
        
        logger.info(f"Puzzle saved to {args.output}")
        
        # Print summary
        eval_data = puzzle["evaluation"]
        print("\nPUZZLE GENERATION SUMMARY")
        print("========================")
        print(f"Grid radius: {puzzle['subGridRadius']}")
        print(f"Blocked hexes: {len(puzzle['blockedHexes'])}")
        print(f"Total pieces: {len(puzzle['pieces'])}")
        print(f"Difficulty score: {eval_data['difficulty_score']:.2f}")
        print(f"Player win rate: {eval_data['player_win_rate']:.2%}")
        print(f"Total simulations: {eval_data['total_games']}")
    else:
        logger.error("Failed to generate valid puzzle")
        sys.exit(1)

if __name__ == "__main__":
    main()
