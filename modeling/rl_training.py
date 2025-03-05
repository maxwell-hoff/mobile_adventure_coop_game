import argparse
import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time
import sys
import math

from copy import deepcopy
from itertools import combinations

# PPO-related
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

np.random.seed(42)
random.seed(42)

# Load scenario + pieces data
with open("data/world.yaml","r") as f:
    world_data = yaml.safe_load(f)
with open("data/pieces.yaml","r") as f:
    pieces_data = yaml.safe_load(f)

###############################################
# Utility
###############################################

def hex_distance(q1, r1, q2, r2):
    return (
        abs(q1 - q2)
        + abs(r1 - r2)
        + abs((q1 + r1) - (q2 + r2))

def line_of_sight(q1, r1, q2, r2, blocked_hexes, all_pieces):
    if (q1 == q2) and (r1 == r2):
        return True
    N = max(abs(q2 - q1), abs(r2 - r1), abs((q1 + r1) - (q2 + r2)))
    if N == 0:
        return True

    s1 = -q1 - r1
    s2 = -q2 - r2
    line_hexes = []
    for i in range(N + 1):
        t = i / N
        qf = q1 + (q2 - q1)*t
        rf = r1 + (r2 - r1)*t
        sf = s1 + (s2 - s1)*t
        rq = round(qf)
        rr = round(rf)
        rs = round(sf)
        # fix rounding so q + r + s = 0
        qdiff = abs(rq - qf)
        rdiff = abs(rr - rf)
        sdiff = abs(rs - sf)
        if qdiff > rdiff and qdiff > sdiff:
            rq = -rr - rs
        elif rdiff > sdiff:
            rr = -rq - rs
        line_hexes.append((rq, rr))

    for (hq, hr) in line_hexes[1:-1]:
        if (hq, hr) in blocked_hexes:
            return False
        for p in all_pieces:
            if (not p.get("dead", False)) and (p["q"], p["r"]) == (hq, hr):
                return False
    return True

##################################
# Environment
##################################

class HexPuzzleEnv(gym.Env):
    def __init__(
        self,
        puzzle_scenario,
        max_turns=10,
        render_mode=None,
        randomize_positions=False,
        randomize_radius=False,
        radius_min=2,
        radius_max=5,
        randomize_blocked=False,
        min_blocked=1,
        max_blocked=5,
        randomize_pieces=False,
        player_min_pieces=3,
        player_max_pieces=4,
        enemy_min_pieces=3,
        enemy_max_pieces=5
    ):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)

        self.max_turns = max_turns
        self.render_mode = render_mode

        self.randomize_positions = randomize_positions
        self.randomize_radius = randomize_radius
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.randomize_blocked = randomize_blocked
        self.min_blocked = min_blocked
        self.max_blocked = max_blocked
        self.randomize_pieces = randomize_pieces
        self.player_min_pieces = player_min_pieces
        self.player_max_pieces = player_max_pieces
        self.enemy_min_pieces = enemy_min_pieces
        self.enemy_max_pieces = enemy_max_pieces

        self.grid_radius = self.scenario["subGridRadius"]

        self._init_pieces_from_scenario(self.scenario)
        self.all_hexes = []
        for q in range(-self.grid_radius,self.grid_radius+1):
            for r in range(-self.grid_radius,self.grid_radius+1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q,r))

        self.max_actions_for_side = 500
        self.action_space = gym.spaces.Discrete(self.max_actions_for_side)

        if self.randomize_pieces:
            max_pieces = self.player_max_pieces + self.enemy_max_pieces
        else:
            max_pieces = len(self.player_pieces) + len(self.enemy_pieces)
        self.obs_size = 2 * max_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.iteration_number = 0
        self.step_number = 0
        self.turn_side = "player"
        self.done_forced = False
        self.current_step_reward = 0.0

        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0
        self.delayedAttacks = []

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        """Initialize pieces with explicit, unambiguous labels based on side and class."""
        valid_classes = set(pieces_data["classes"].keys())
        
        # First validate and fix any invalid classes
        for pc in scenario_dict["pieces"]:
            if pc["class"] not in valid_classes:
                print(f"[WARNING] Invalid class {pc['class']}. Defaulting to 'Priest'.")
                pc["class"] = "Priest"
            # Ensure original_class is set
            if "original_class" not in pc:
                pc["original_class"] = pc["class"]
            # Validate class consistency
            elif pc["class"] != pc["original_class"]:
                print(f"[WARNING] Class mismatch in piece! Original: {pc['original_class']}, Current: {pc['class']}")
                pc["class"] = pc["original_class"]  # Restore original class
        
        # Split pieces by side first
        player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]
        
        # Create class counters for each side
        player_class_counts = {}
        enemy_class_counts = {}
        
        # Label player pieces
        for piece in player_pieces:
            piece_class = piece["class"]
            if piece_class not in player_class_counts:
                player_class_counts[piece_class] = 1
            else:
                player_class_counts[piece_class] += 1
            piece["label"] = f"P-{piece_class}-{player_class_counts[piece_class]}"
        
        # Label enemy pieces
        for piece in enemy_pieces:
            piece_class = piece["class"]
            if piece_class not in enemy_class_counts:
                enemy_class_counts[piece_class] = 1
            else:
                enemy_class_counts[piece_class] += 1
            piece["label"] = f"E-{piece_class}-{enemy_class_counts[piece_class]}"
        
        # Final validation
        for p in player_pieces + enemy_pieces:
            assert p["class"] == p["original_class"], f"Class consistency error: {p['class']} != {p['original_class']}"
            assert p["class"] in valid_classes, f"Invalid piece class after initialization: {p['class']}"
        
        # Store pieces in instance variables
        self.player_pieces = player_pieces
        self.enemy_pieces = enemy_pieces

    def _build_all_hexes(self):
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q,r))

    def _randomize_scenario(self):
        if self.randomize_radius:
            self.scenario["subGridRadius"] = random.randint(self.radius_min, self.radius_max)
        else:
            self.scenario["subGridRadius"] = self.original_scenario["subGridRadius"]
        self.grid_radius = self.scenario["subGridRadius"]
        self._build_all_hexes()

        if self.randomize_blocked:
            count = random.randint(self.min_blocked, self.max_blocked)
            picks = random.sample(self.all_hexes, min(count, len(self.all_hexes)))
            self.scenario["blockedHexes"] = [{"q":q,"r":r} for (q,r) in picks]
        else:
            self.scenario["blockedHexes"] = deepcopy(self.original_scenario["blockedHexes"])

        if self.randomize_pieces:
            player_classes = ["Warlock","Sorcerer","Priest"]
            enemy_classes  = ["Guardian","BloodWarden","Hunter","Priest"]
            pcount = random.randint(self.player_min_pieces, self.player_max_pieces)
            ecount = random.randint(self.enemy_min_pieces, self.enemy_max_pieces)

            def pick_side_pieces(side, class_list, count):
                pieces = []
                # ensure 1 Priest
                pieces.append({
                    "class": "Priest",
                    "color": "#556b2f" if side=="player" else "#dc143c",
                    "side": side,
                    "q": 0,
                    "r": 0
                })
                remainder = count - 1
                other_classes = [c for c in class_list if c!="Priest"]
                for _ in range(remainder):
                    c = random.choice(other_classes)
                    pieces.append({
                        "class": c,
                        "color": "#556b2f" if side=="player" else "#dc143c",
                        "side": side,
                        "q": 0,
                        "r": 0
                    })
                return pieces

            new_player = pick_side_pieces("player", player_classes, pcount)
            new_enemy  = pick_side_pieces("enemy",  enemy_classes, ecount)
            self.scenario["pieces"] = new_player + new_enemy
        else:
            self.scenario["pieces"] = deepcopy(self.original_scenario["pieces"])

    def _randomize_piece_positions(self):
        blocked_hexes = {(bh["q"], bh["r"]) for bh in self.scenario["blockedHexes"]}
        valid_hexes = [(q,r) for (q,r) in self.all_hexes if (q,r) not in blocked_hexes]
        needed = len(self.player_pieces) + len(self.enemy_pieces)
        if needed > len(valid_hexes):
            print("[WARNING] Not enough valid hexes for random placement.")
            return
        picks = random.sample(valid_hexes, needed)
        idx = 0
        for p in self.player_pieces:
            p["q"], p["r"] = picks[idx]
            idx += 1
        for e in self.enemy_pieces:
            e["q"], e["r"] = picks[idx]
            idx += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.current_episode) > 0:
            # Add iteration info to the episode before saving
            for step in self.current_episode:
                step["iteration_number"] = self.iteration_number
            self.all_episodes.append(self.current_episode)
        
        print(f"\n[RESET] Starting iteration {self.iteration_number + 1}")
        
        self.current_episode = []
        self.current_step_reward = 0.0

        # Start fresh
        self.scenario = deepcopy(self.original_scenario)
        
        # First randomize the scenario if needed
        if self.randomize_radius:
            old_radius = self.scenario.get("subGridRadius", 0)
            self.scenario["subGridRadius"] = random.randint(self.radius_min, self.radius_max)
            self.grid_radius = self.scenario["subGridRadius"]
            print(f"[RESET] Randomized grid radius: {old_radius} -> {self.grid_radius}")
            self._build_all_hexes()

        if self.randomize_blocked:
            count = random.randint(self.min_blocked, self.max_blocked)
            picks = random.sample(self.all_hexes, min(count, len(self.all_hexes)))
            self.scenario["blockedHexes"] = [{"q":q,"r":r} for (q,r) in picks]
            print(f"[RESET] Created {len(self.scenario['blockedHexes'])} blocked hexes")

        # Initialize pieces first
        if self.randomize_pieces:
            print("[RESET] Creating random pieces")
            self._create_random_pieces()
        
        # Now initialize all pieces properly
        self._init_pieces_from_scenario(self.scenario)
        
        # Randomize positions if needed
        if self.randomize_positions:
            print("[RESET] Randomizing piece positions")
            self._randomize_piece_positions()

        # Reset game state
        self.turn_number = 1
        self.iteration_number += 1
        self.step_number = 1
        self.turn_side = "player"
        self.done_forced = False
        self.non_bloodwarden_kills = 0
        self.delayedAttacks.clear()

        for p in self.all_pieces:
            p["moved_this_turn"] = False
            
        # Display piece information for debugging
        print("[RESET] Initial piece configuration:")
        for i, p in enumerate(self.player_pieces):
            print(f"  Player {i+1}: {p['label']} ({p['class']}) at ({p['q']},{p['r']})")
        for i, e in enumerate(self.enemy_pieces):
            print(f"  Enemy {i+1}: {e['label']} ({e['class']}) at ({e['q']},{e['r']})")

        init_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions(),
            "grid_radius": self.grid_radius,
            "blocked_hexes": deepcopy(self.scenario["blockedHexes"]),
            "iteration_number": self.iteration_number,
            "step_number": 0
        }
        self.current_episode.append(init_dict)
        
        print(f"[RESET] Iteration {self.iteration_number} initialized successfully")
        
        return self.get_obs(), {}

    def _create_random_pieces(self):
        """Create random pieces with proper initialization."""
        player_classes = ["Warlock","Sorcerer","Priest"]
        enemy_classes  = ["Guardian","BloodWarden","Hunter","Priest"]
        pcount = random.randint(self.player_min_pieces, self.player_max_pieces)
        ecount = random.randint(self.enemy_min_pieces, self.enemy_max_pieces)

        def create_side_pieces(side, class_list, count):
            pieces = []
            # Always ensure one Priest
            priest_piece = {
                "class": "Priest",
                "side": side,
                "color": "#556b2f" if side=="player" else "#dc143c",
                "q": 0,
                "r": 0,
                "dead": False,
                "moved_this_turn": False,
                "original_class": "Priest"  # Track original class
            }
            pieces.append(priest_piece)
            
            # Add remaining pieces
            other_classes = [c for c in class_list if c!="Priest"]
            for _ in range(count - 1):
                c = random.choice(other_classes)
                piece = {
                    "class": c,
                    "side": side,
                    "color": "#556b2f" if side=="player" else "#dc143c",
                    "q": 0,
                    "r": 0,
                    "dead": False,
                    "moved_this_turn": False,
                    "original_class": c  # Track original class
                }
                pieces.append(piece)
            return pieces

        # Create pieces for both sides
        new_player = create_side_pieces("player", player_classes, pcount)
        new_enemy = create_side_pieces("enemy", enemy_classes, ecount)
        
        # Validate piece classes before updating
        for p in new_player + new_enemy:
            if p["class"] != p["original_class"]:
                print(f"WARNING: Class mismatch detected! Original: {p['original_class']}, Current: {p['class']}")
            assert p["class"] in pieces_data["classes"], f"Invalid piece class: {p['class']}"
        
        # Update scenario
        self.scenario["pieces"] = new_player + new_enemy

    def step(self, action_idx):
        # Reset the current step reward at the start of each step
        self.current_step_reward = 0.0
        self.step_number += 1
        
        print(f"\n[STEP {self.step_number}] Turn {self.turn_number}, {self.turn_side.upper()} side")
        
        if self.done_forced:
            print(f"[STEP {self.step_number}] Environment already terminated")
            return self.get_obs(), 0.0, True, False, {}

        valid = self.build_action_list()
        if action_idx >= len(valid):
            print(f"[STEP {self.step_number}] Invalid action index: {action_idx} (valid range: 0-{len(valid)-1})")
            return self.get_obs(), 0.0, True, False, {}

        (pidx, sub_action) = valid[action_idx]
        piece = self.all_pieces[pidx]
        piece["moved_this_turn"] = True
        
        print(f"[STEP {self.step_number}] Piece {piece['label']} ({piece['class']}) at ({piece['q']},{piece['r']}) selected")
        print(f"[STEP {self.step_number}] Action: {sub_action['type']}")

        # Before we execute the action, check if piece could have attacked but chooses
        # something else => apply small penalty.
        could_attack = self._could_have_attacked(piece)
        is_attack = sub_action["type"] in ["single_target_attack", "multi_target_attack", "aoe"]
        if could_attack and not is_attack:
            self.current_step_reward -= 4.0
            print(f"[STEP {self.step_number}] Could have attacked but didn't => -4.0 penalty")

        atype = sub_action["type"]
        if atype == "move":
            (q, r) = sub_action["dest"]
            old_q, old_r = piece["q"], piece["r"]
            piece["q"] = q
            piece["r"] = r
            print(f"[STEP {self.step_number}] Move from ({old_q},{old_r}) to ({q},{r})")

        elif atype == "pass":
            self.current_step_reward -= 1.0
            print(f"[STEP {self.step_number}] Pass action => -1.0 penalty")

        elif atype == "aoe":
            print(f"[STEP {self.step_number}] AOE attack")
            if sub_action.get("name") == "necrotizing_consecrate":
                if piece["class"] == "BloodWarden":  # Extra validation
                    self._schedule_necro(piece)
                    print(f"[STEP {self.step_number}] Scheduled delayed AOE: necrotizing_consecrate")
            else:
                targets = sub_action["targets"]
                print(f"[STEP {self.step_number}] AOE targeting {len(targets)} enemies")
                for i, tgt in enumerate(targets):
                    if not tgt.get("dead", False):
                        print(f"[STEP {self.step_number}] Target {i+1}: {tgt['label']} ({tgt['class']})")
                        self._kill_piece(tgt, killer_side=piece["side"])
                        # If we killed a Priest, stop processing remaining targets and end immediately
                        if self.done_forced:
                            print(f"[STEP {self.step_number}] Killed a Priest => terminating episode")
                            return self._finish_step(True, False)

        elif atype == "single_target_attack":
            tgt = sub_action["target_piece"]
            if tgt and not tgt.get("dead", False):
                print(f"[STEP {self.step_number}] Single target attack on {tgt['label']} ({tgt['class']})")
                self._kill_piece(tgt, killer_side=piece["side"])
                if self.done_forced:  # Will be True if we killed a Priest
                    print(f"[STEP {self.step_number}] Killed a Priest => terminating episode")
                    return self._finish_step(True, False)
            else:
                print(f"[STEP {self.step_number}] Invalid target for single_target_attack: {tgt}")

        elif atype == "multi_target_attack":
            targets = sub_action["targets"]
            print(f"[STEP {self.step_number}] Multi-target attack on {len(targets)} targets")
            for i, tgt in enumerate(targets):
                if not tgt.get("dead", False):
                    print(f"[STEP {self.step_number}] Target {i+1}: {tgt['label']} ({tgt['class']})")
                    self._kill_piece(tgt, killer_side=piece["side"])
                    # If we killed a Priest, stop processing remaining targets and end immediately
                    if self.done_forced:
                        print(f"[STEP {self.step_number}] Killed a Priest => terminating episode")
                        return self._finish_step(True, False)

        elif atype == "swap_position":
            # Extra validation that piece can actually swap
            pclass_data = pieces_data["classes"].get(piece["class"], {})
            valid_actions = pclass_data.get("actions", {})
            if "swap_position" in valid_actions:
                tgt = sub_action["target_piece"]
                if tgt and not tgt.get("dead", False):
                    old_q, old_r = piece["q"], piece["r"]
                    piece["q"], piece["r"] = tgt["q"], tgt["r"]
                    tgt["q"], tgt["r"] = old_q, old_r
                else:
                    self.current_step_reward -= 1.0
            else:
                self.current_step_reward -= 1.0

        # Apply end conditions (like turn limit)
        if self.turn_number >= self.max_turns:
            self.current_step_reward -= 20
            return self._finish_step(True, True)

        return self._finish_step(False, False)

    def _finish_step(self, terminated, truncated):
        print(f"[FINISH_STEP] Step {self.step_number}, Turn {self.turn_number}, side {self.turn_side}")
        print(f"[FINISH_STEP] Reward: {self.current_step_reward}")
        print(f"[FINISH_STEP] Terminated: {terminated}, Truncated: {truncated}")
        
        step_data = {
            "turn_number": self.turn_number,
            "turn_side": self.turn_side,
            "reward": self.current_step_reward,
            "positions": self._log_positions(),
            "grid_radius": self.grid_radius,
            "blocked_hexes": deepcopy(self.scenario["blockedHexes"]),
            "non_bloodwarden_kills": self.non_bloodwarden_kills,
            "iteration_number": self.iteration_number,
            "step_number": self.step_number
        }
        self.current_episode.append(step_data)

        if terminated or truncated:
            step_data["non_bloodwarden_kills"] = self.non_bloodwarden_kills
            self.done_forced = True
            print(f"[FINISH_STEP] Episode ending: terminated={terminated}, truncated={truncated}")
            if terminated:
                if self.current_step_reward >= 30:
                    print(f"[FINISH_STEP] Priest kill detected!")
                else:
                    print(f"[FINISH_STEP] Episode terminated for other reason")
            if truncated:
                print(f"[FINISH_STEP] Episode truncated due to turn limit")

        if not (terminated or truncated):
            # proceed to next side if all moved
            if self._all_side_pieces_have_moved(self.turn_side):
                old_side = self.turn_side
                if self.turn_side == "player":
                    self.turn_side = "enemy"
                    print(f"[FINISH_STEP] Side change: {old_side} → {self.turn_side}")
                else:
                    self.turn_side = "player"
                    self.turn_number += 1
                    print(f"[FINISH_STEP] Side change: {old_side} → {self.turn_side}, advancing to turn {self.turn_number}")
                
                for pc in self.all_pieces:
                    pc["moved_this_turn"] = False
                
                # Show alive pieces for next turn
                alive_player = [p for p in self.player_pieces if not p.get("dead", False)]
                alive_enemy = [e for e in self.enemy_pieces if not e.get("dead", False)]
                print(f"[FINISH_STEP] Alive player pieces: {len(alive_player)}, alive enemy pieces: {len(alive_enemy)}")
                
                self._check_delayed_attacks()
            else:
                print(f"[FINISH_STEP] Continuing with current side ({self.turn_side}) - not all pieces have moved")

        obs = self.get_obs()
        return obs, self.current_step_reward, terminated, truncated, {}

    def build_action_list(self):
        actions = []
        living_side = [
            (i, pc)
            for (i, pc) in enumerate(self.all_pieces)
            if pc["side"] == self.turn_side
            and not pc.get("dead", False)
            and not pc.get("moved_this_turn", False)
        ]

        if not living_side:
            return actions

        # Identify enemies vs. allies
        if self.turn_side == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
            allies  = [p for p in self.player_pieces if not p.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]
            allies  = [e for e in self.enemy_pieces if not e.get("dead", False)]

        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}

        for (pidx, piece) in living_side:
            pclass = piece["class"]
            if pclass not in pieces_data["classes"]:
                continue  # Skip invalid piece classes
            pclass_data = pieces_data["classes"][pclass]
            
            # Only add actions that are valid for this piece class
            valid_actions = pclass_data.get("actions", {})

            # Move
            if "move" in valid_actions:
                mrange = valid_actions["move"]["range"]
                for (q, r) in self.all_hexes:
                    if (q, r) != (piece["q"], piece["r"]):
                        if hex_distance(piece["q"], piece["r"], q, r) <= mrange:
                            if not self._occupied_or_blocked(q, r):
                                actions.append((pidx, {"type": "move", "dest": (q, r)}))

            # Pass
            actions.append((pidx, {"type": "pass"}))

            # Other actions
            for aname, adata in valid_actions.items():
                if aname == "move":
                    continue
                if aname == "necrotizing_consecrate":
                    # Special AoE check
                    if pclass == "BloodWarden":  # Only BloodWarden can use this
                        actions.append((pidx, {"type": "aoe", "name": "necrotizing_consecrate"}))
                    continue

                atype = adata.get("action_type")
                if not atype:
                    continue

                rng = adata.get("range", 0)
                requires_los = adata.get("requires_los", False)
                ally_only = adata.get("ally_only", False)

                # Single-target Attack
                if atype == "single_target_attack":
                    for eP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], eP["q"], eP["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                eP["q"], eP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                actions.append((pidx, {
                                    "type": "single_target_attack",
                                    "action_name": aname,
                                    "target_piece": eP,
                                }))

                # Multi-target Attack
                elif atype == "multi_target_attack":
                    max_tg = adata.get("max_num_targets", 1)
                    in_range = []
                    for eP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], eP["q"], eP["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                eP["q"], eP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                in_range.append(eP)
                    for size in range(1, max_tg + 1):
                        for combo in combinations(in_range, size):
                            actions.append((pidx, {
                                "type": "multi_target_attack",
                                "action_name": aname,
                                "targets": list(combo)
                            }))

                # Swap position (only if piece class explicitly allows it)
                elif atype == "swap_position":
                    if "swap_position" in valid_actions:  # Extra validation
                        possible = allies if ally_only else self.all_pieces
                        possible = [x for x in possible if x is not piece and not x.get("dead", False)]
                        for tgt in possible:
                            dist = hex_distance(piece["q"], piece["r"], tgt["q"], tgt["r"])
                            if dist <= rng:
                                if (not requires_los) or line_of_sight(
                                    piece["q"], piece["r"],
                                    tgt["q"], tgt["r"],
                                    blocked_hexes, self.all_pieces
                                ):
                                    actions.append((pidx, {
                                        "type": "swap_position",
                                        "action_name": aname,
                                        "target_piece": tgt
                                    }))

                # AoE (generic)
                elif atype == "aoe":
                    radius = adata.get("radius", 0)
                    in_range = []
                    for eP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], eP["q"], eP["r"])
                        if dist <= radius:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                eP["q"], eP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                in_range.append(eP)
                    if in_range:
                        actions.append((pidx, {
                            "type": "aoe",
                            "action_name": aname,
                            "name": aname,
                            "targets": in_range
                        }))

        return actions[: self.max_actions_for_side]

    def _occupied_or_blocked(self, q, r):
        blocked_hexes = {(bh["q"], bh["r"]) for bh in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return True
        for p in self.all_pieces:
            if (not p.get("dead", False)) and (p["q"], p["r"]) == (q, r):
                return True
        return False

    def _could_have_attacked(self, piece):
        # Only check if there's at least one enemy in range for an action
        if piece["side"] == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]
        if not enemies:
            return False

        pclass_data = pieces_data["classes"][piece["class"]]
        blocked_hexes = {(bh["q"], bh["r"]) for bh in self.scenario["blockedHexes"]}

        for aname, adata in pclass_data["actions"].items():
            atype = adata["action_type"]
            if atype in ["single_target_attack", "multi_target_attack", "aoe"]:
                rng = adata.get("range", 0)
                requires_los = adata.get("requires_los", False)
                if atype == "aoe":
                    radius = adata.get("radius", 0)
                    for e in enemies:
                        dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                        if dist <= radius:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"], e["q"], e["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                return True
                else:
                    # single or multi
                    for e in enemies:
                        dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"], e["q"], e["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                return True
        return False

    def _schedule_necro(self, piece):
        # Delayed AoE from BloodWarden
        pclass_data = pieces_data["classes"][piece["class"]]
        necro_data = pclass_data["actions"]["necrotizing_consecrate"]
        c_speed = necro_data.get("cast_speed", 0)
        if c_speed > 0:
            trig = self.turn_number + c_speed
            evt = {
                "turn_number": self.turn_number,
                "turn_side": piece["side"],
                "reward": 0.0,
                "positions": self._log_positions(),
                "desc": f"{piece['class']}({piece['label']}) starts necro => triggers turn {trig}"
            }
            self.current_episode.append(evt)
            self.delayedAttacks.append({
                "caster_side": piece["side"],
                "caster_label": piece["label"],
                "trigger_turn": trig,
                "action_name": "necrotizing_consecrate"
            })
        else:
            # Immediate effect => kills all enemies
            if piece["side"] == "enemy":
                for p in self.player_pieces:
                    if not p.get("dead", False):
                        self._kill_piece(p, piece["side"])
                        # If we killed a Priest, stop processing remaining targets
                        if self.done_forced:
                            break
            else:
                for e in self.enemy_pieces:
                    if not e.get("dead", False):
                        self._kill_piece(e, piece["side"])
                        # If we killed a Priest, stop processing remaining targets
                        if self.done_forced:
                            break

    def _check_delayed_attacks(self):
        # Trigger delayed necro if time
        to_remove = []
        for i, att in enumerate(self.delayedAttacks):
            if att["trigger_turn"] == self.turn_number:
                side = att["caster_side"]
                if side == "enemy":
                    for p in self.player_pieces:
                        if not p.get("dead", False):
                            self._kill_piece(p, side)
                            # If we killed a Priest, stop processing remaining targets
                            if self.done_forced:
                                break
                else:
                    for e in self.enemy_pieces:
                        if not e.get("dead", False):
                            self._kill_piece(e, side)
                            # If we killed a Priest, stop processing remaining targets
                            if self.done_forced:
                                break
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _apply_end_conditions(self, base_reward):
        # CHANGES MADE:
        # Now, the only end conditions are:
        # 1) Turn limit => -20 and truncated.
        # 2) A priest has been killed (handled in _kill_piece => self.done_forced).
        rew = base_reward
        term = False
        trunc = False

        if self.turn_number >= self.max_turns:
            rew -= 20
            trunc = True

        return rew, term, trunc

    def _kill_piece(self, piece, killer_side):
        """Kill a piece and handle rewards.
        
        Only ends episode if a Priest is killed by the opposing side.
        Gives +30 reward for killing enemy Priest.
        Gives +5 reward for killing any other enemy piece.
        """
        if piece.get("dead", False):
            return

        # Debug info
        print(f"DEBUG: Killing piece - Class: {piece['class']}, Side: {piece['side']}, Killer: {killer_side}")

        # Validate the piece class and state
        if piece["class"] not in pieces_data["classes"]:
            print(f"[WARNING] Invalid piece class: {piece['class']}")
            return

        # Only give rewards and end episode for enemy kills
        if piece["side"] != killer_side:
            # Check if it's a Priest kill BEFORE marking as dead
            is_priest = piece["class"].strip() == "Priest"  # Add strip() to handle any whitespace
            print(f"DEBUG: is_priest={is_priest}, class={piece['class']}")
            
            if is_priest:
                print(f"DEBUG: Priest kill detected! Reward +30. Iteration: {self.iteration_number}, step: {self.step_number}, turn: {self.turn_number}, side {self.turn_side}, piece {piece['class']}, killer {killer_side}, piece side {piece['side']}, piece label {piece['label']}")
                self.current_step_reward += 30
                self.done_forced = True  # End episode immediately
            else:
                print(f"DEBUG: Non-priest kill detected! Reward +5")
                self.current_step_reward += 5  # Regular enemy kill

        # Mark piece as dead and move off board
        piece["dead"] = True
        piece["q"] = 9999
        piece["r"] = 9999

        # Track non-bloodwarden kills
        if piece["class"] != "Priest" and killer_side != "BloodWarden":
            self.non_bloodwarden_kills += 1

    def get_obs(self):
        """
        Return an 18-length observation vector by padding with zeros
        if we have fewer than 9 total pieces.
        """
        # Build coords for each piece, [p1_q, p1_r, p2_q, p2_r, ...]
        coords = []
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        
        # The model expects 18 floats (for 9 pieces).
        # If we have fewer than 9 pieces total, pad with zeros.
        # If we have exactly 9 or more, you might want to either slice or keep them
        # (depending on how you trained).
        
        desired_size = 18  # 9 pieces * 2 coords
        current_size = len(coords)
        
        if current_size < desired_size:
            # zero-pad at the end
            coords += [0] * (desired_size - current_size)
        elif current_size > desired_size:
            # or if we have more pieces than expected, either slice them
            # or raise an error. The line below just slices to 18:
            coords = coords[:desired_size]
        
        return np.array(coords, dtype=np.float32)

    def _log_positions(self):
        # Add debug info about piece order
        print("\nTRAINING DEBUG: Current piece order:")
        print("Player pieces:")
        for i, p in enumerate(self.player_pieces):
            print(f"  {i}: {p['label']} ({p['class']}) at ({p['q']},{p['r']})")
        print("Enemy pieces:")
        for i, e in enumerate(self.enemy_pieces):
            print(f"  {i}: {e['label']} ({e['class']}) at ({e['q']},{e['r']})")
        print("------------------------")
        
        pa = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        ea = np.array([[e["q"], e["r"]] for e in self.enemy_pieces], dtype=np.float32)
        
        # Add piece information to the returned data
        return {
            "player": pa,
            "enemy": ea,
            "player_pieces": [{
                "label": p["label"],
                "class": p["class"],
                "side": p["side"]
            } for p in self.player_pieces],
            "enemy_pieces": [{
                "label": e["label"],
                "class": e["class"],
                "side": e["side"]
            } for e in self.enemy_pieces]
        }

    def _get_action_mask(self):
        # We only mask valid actions. If done, mask all except action 0 to avoid crashes.
        if self.done_forced:
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        valid = self.build_action_list()
        mask = np.zeros(self.max_actions_for_side, dtype=bool)
        for i in range(len(valid)):
            mask[i] = True
        return mask

    def action_masks(self):
        return self._get_action_mask()

    def sync_with_puzzle_scenario(self, scenario_dict, turn_side="enemy"):
        self.scenario = deepcopy(scenario_dict)
        self._init_pieces_from_scenario(self.scenario)
        self.grid_radius = self.scenario["subGridRadius"]
        self._build_all_hexes()
        self.turn_side = turn_side
        self.turn_number = 1
        self.done_forced = False
        self.delayedAttacks.clear()

        self.obs_size = 2 * (len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        self.current_episode = []
        init_dict = {
            "turn_number": 0,
            "turn_side": self.turn_side,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(init_dict)

    def _all_side_pieces_have_moved(self, side):
        for piece in self.all_pieces:
            if piece["side"] == side and not piece.get("dead", False):
                if not piece.get("moved_this_turn", False):
                    return False
        return True


def make_env_fn(
    scenario_dict,
    randomize_positions=False,
    randomize_radius=False,
    radius_min=2,
    radius_max=5,
    randomize_blocked=False,
    min_blocked=1,
    max_blocked=5,
    randomize_pieces=False,
    player_min_pieces=3,
    player_max_pieces=4,
    enemy_min_pieces=3,
    enemy_max_pieces=5
):
    def _init():
        env = HexPuzzleEnv(
            puzzle_scenario=scenario_dict,
            max_turns=10,
            randomize_positions=randomize_positions,
            randomize_radius=randomize_radius,
            radius_min=radius_min,
            radius_max=radius_max,
            randomize_blocked=randomize_blocked,
            min_blocked=min_blocked,
            max_blocked=max_blocked,
            randomize_pieces=randomize_pieces,
            player_min_pieces=player_min_pieces,
            player_max_pieces=player_max_pieces,
            enemy_min_pieces=enemy_min_pieces,
            enemy_max_pieces=enemy_max_pieces
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

###########################################
#  Simple MCTS / Tree / PPO approach below
###########################################

class MCTSNode:
    def __init__(self, obs_bytes, turn_side):
        self.obs_bytes = obs_bytes
        self.turn_side = turn_side
        self.actions = None
        self.stats = {}
        self.untried = []
        self.visit_sum = 0

def obs_to_key(obs, turn_side):
    return (obs.tobytes(), turn_side)

mcts_tree = {}

def mcts_policy(env, max_iterations=50):
    root_obs = env.get_obs()
    root_side = env.turn_side
    root_key = obs_to_key(root_obs, root_side)

    if root_key not in mcts_tree:
        node = MCTSNode(root_obs.tobytes(), root_side)
        valid_acts = env.build_action_list()
        node.actions = list(range(len(valid_acts)))
        node.untried = list(range(len(valid_acts)))
        for a_idx in node.actions:
            node.stats[a_idx] = [0, 0.0]
        mcts_tree[root_key] = node
    else:
        node = mcts_tree[root_key]

    for _ in range(max_iterations):
        env_copy = deepcopy(env)
        search_path = []
        rollout_return = mcts_search(env_copy, search_path)

        for (visited_key, act_idx) in search_path:
            if visited_key not in mcts_tree:
                continue
            visited_node = mcts_tree[visited_key]
            N, Q = visited_node.stats[act_idx]
            N += 1
            new_Q = Q + (rollout_return - Q) / N
            visited_node.stats[act_idx] = [N, new_Q]
            visited_node.visit_sum += 1

    # Pick the action with the highest visit count
    best_action = 0
    best_visits = -1
    for a_idx, (visits, qval) in node.stats.items():
        if visits > best_visits:
            best_visits = visits
            best_action = a_idx

    mcts_debug_info = {}
    for a_idx, (visits, qval) in node.stats.items():
        mcts_debug_info[a_idx] = {"visits": visits, "q_value": round(qval, 3)}
    mcts_debug_info["chosen_action_idx"] = best_action
    env.mcts_debug = mcts_debug_info

    return best_action

def mcts_search(env_copy, path, depth=0, max_depth=20):
    # We only do MCTS on the "enemy" side in this example. 
    if env_copy.turn_side != "enemy":
        return env_copy.current_episode[-1]["reward"]

    if depth >= max_depth or env_copy.done_forced:
        return env_copy.current_episode[-1]["reward"]

    obs = env_copy.get_obs()
    side = env_copy.turn_side
    node_key = obs_to_key(obs, side)

    if node_key not in mcts_tree:
        new_node = MCTSNode(obs.tobytes(), side)
        valid_actions = env_copy.build_action_list()
        new_node.actions = list(range(len(valid_actions)))
        new_node.untried = list(range(len(valid_actions)))
        for a_idx in new_node.actions:
            new_node.stats[a_idx] = [0, 0.0]
        mcts_tree[node_key] = new_node

        if not new_node.untried:
            return env_copy.current_episode[-1]["reward"]
        action_idx = new_node.untried.pop()
        path.append((node_key, action_idx))
        obs2, rew, term, trunc, _ = env_copy.step(action_idx)
        if term or trunc:
            return env_copy.current_episode[-1]["reward"]
        return rollout(env_copy, depth + 1, max_depth)
    else:
        node = mcts_tree[node_key]
        # If there's still an untried action, pick it
        if node.untried:
            a_idx = node.untried.pop()
            path.append((node_key, a_idx))
            obs2, rew, term, trunc, _ = env_copy.step(a_idx)
            if term or trunc:
                return env_copy.current_episode[-1]["reward"]
            return rollout(env_copy, depth+1, max_depth)
        else:
            a_idx = best_uct_action(node)
            if a_idx is None:
                return env_copy.current_episode[-1]["reward"]
            path.append((node_key, a_idx))
            obs2, rew, term, trunc, _ = env_copy.step(a_idx)
            if term or trunc:
                return env_copy.current_episode[-1]["reward"]
            return mcts_search(env_copy, path, depth+1, max_depth)

def best_uct_action(node, c=1.4):
    best_score = -999999
    best_action = None
    for a_idx in node.actions:
        N, Q = node.stats[a_idx]
        if N == 0:
            return a_idx
        uct_val = Q + c * math.sqrt(math.log(node.visit_sum + 1) / N)
        if uct_val > best_score:
            best_score = uct_val
            best_action = a_idx
    return best_action

def rollout(env_copy, depth, max_depth):
    while depth < max_depth and not env_copy.done_forced:
        if env_copy.turn_side != "enemy":
            # We only push random actions for the enemy in this example
            break
        valid = env_copy.build_action_list()
        if not valid:
            break
        a_idx = random.randint(0, len(valid)-1)
        obs2, rew, term, trunc, _ = env_copy.step(a_idx)
        if term or trunc:
            break
        depth += 1
    return env_copy.current_episode[-1]["reward"]

def tree_select_action(env, depth=1):
    # A simple heuristic "tree" approach for the active side
    valid_actions = env.build_action_list()
    if not valid_actions:
        return 0  # If no actions, do nothing
    best_score = -99999.0
    best_idx = 0
    for i, (pidx, sub_action) in enumerate(valid_actions):
        env_copy = deepcopy(env)
        obs_next, reward_next, terminated, truncated, info = env_copy.step(i)
        total = reward_next
        # Just a shallow look-ahead
        if not terminated and not truncated and depth > 1:
            enemy_actions = env_copy.build_action_list()
            if enemy_actions:
                rand_idx = random.randint(0, len(enemy_actions)-1)
                obs2, rew2, ter2, tru2, inf2 = env_copy.step(rand_idx)
                total += rew2
        if total > best_score:
            best_score = total
            best_idx = i
    return best_idx

def run_tree_search(env):
    obs, info = env.reset()
    done = False
    while not done:
        action_idx = tree_select_action(env, depth=1)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
    if len(env.current_episode) > 0:
        env.all_episodes.append(env.current_episode)

def _run_one_episode(model, env):
    obs, info = env.reset()
    done, state = False, None
    while not done:
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, _, info = env.step(action)

def main():
    parser = argparse.ArgumentParser(description="Hex Puzzle RL with optional randomization.")
    parser.add_argument("--randomize", action="store_true", help="Randomize piece positions each reset.")
    parser.add_argument("--approach", choices=["ppo", "tree", "mcts"], default="ppo")
    parser.add_argument("--randomize-radius", action="store_true")
    parser.add_argument("--radius-min", type=int, default=2)
    parser.add_argument("--radius-max", type=int, default=5)
    parser.add_argument("--randomize-blocked", action="store_true")
    parser.add_argument("--min-blocked", type=int, default=1)
    parser.add_argument("--max-blocked", type=int, default=5)
    parser.add_argument("--randomize-pieces", action="store_true")
    parser.add_argument("--player-min-pieces", type=int, default=3)
    parser.add_argument("--player-max-pieces", type=int, default=4)
    parser.add_argument("--enemy-min-pieces", type=int, default=3)
    parser.add_argument("--enemy-max-pieces", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION:")
    print(f"Approach: {args.approach}")
    print(f"Randomize positions: {args.randomize}")
    print(f"Randomize radius: {args.randomize_radius} (min={args.radius_min}, max={args.radius_max})")
    print(f"Randomize blocked: {args.randomize_blocked} (min={args.min_blocked}, max={args.max_blocked})")
    print(f"Randomize pieces: {args.randomize_pieces}")
    if args.randomize_pieces:
        print(f"  Player pieces: min={args.player_min_pieces}, max={args.player_max_pieces}")
        print(f"  Enemy pieces: min={args.enemy_min_pieces}, max={args.enemy_max_pieces}")
    print("="*80 + "\n")

    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy = deepcopy(scenario)

    env_fn = make_env_fn(
        scenario_copy,
        randomize_positions=args.randomize,
        randomize_radius=args.randomize_radius,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        randomize_blocked=args.randomize_blocked,
        min_blocked=args.min_blocked,
        max_blocked=args.max_blocked,
        randomize_pieces=args.randomize_pieces,
        player_min_pieces=args.player_min_pieces,
        player_max_pieces=args.player_max_pieces,
        enemy_min_pieces=args.enemy_min_pieces,
        enemy_max_pieces=args.enemy_max_pieces
    )

    if args.approach == "ppo":
        vec_env = DummyVecEnv([env_fn])
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=1e-5,
            n_steps=4096,
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.1,
            max_grad_norm=0.3
        )
        print("\n" + "-"*80)
        print("STARTING PPO TRAINING SESSION")
        print("-"*80)
        print("Training PPO for ~2 minutes. Adjust as desired.")
        start_time = time.time()
        time_limit = 2 * 60  # seconds

        iteration_count_before = 0
        total_steps = 0
        training_iterations = 0
        
        while True:
            training_iterations += 1
            print(f"\nTraining iteration {training_iterations}...")
            model.learn(total_timesteps=1000)
            
            total_steps += 1000
            elapsed = time.time() - start_time
            
            print(f"Total steps: {total_steps} | Elapsed time: {elapsed:.2f}s")
            
            if elapsed >= time_limit:
                print("Time limit reached => stopping training.")
                break

            all_eps = vec_env.envs[0].all_episodes
            num_new_episodes = len(all_eps) - iteration_count_before
            
            print(f"New episodes completed: {num_new_episodes}")
            
            # Detailed episode analysis
            if num_new_episodes > 0:
                priest_kills_player = 0
                priest_kills_enemy = 0
                timeouts = 0
                
                for i, ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
                    if len(ep) == 0:
                        continue
                    final = ep[-1]
                    rew = final["reward"]
                    side = final["turn_side"]
                    
                    # Only print when a priest is killed (reward >= 30)
                    if rew >= 30:
                        if side == "player":
                            priest_kills_player += 1
                            print(f"  Player killed enemy priest in iteration {i+1} (reward: +{rew})")
                        else:
                            priest_kills_enemy += 1
                            print(f"  Enemy killed player priest in iteration {i+1} (reward: +{rew})")
                    elif rew == -20:
                        timeouts += 1
                
                print(f"Episode outcomes: Player priest kills: {priest_kills_player}, Enemy priest kills: {priest_kills_enemy}, Timeouts: {timeouts}")
            
            iteration_count_before = len(all_eps)
            print(f"Current total episodes: {iteration_count_before}")

        all_episodes = vec_env.envs[0].all_episodes
        print("\nTRAINING COMPLETED")
        print(f"Total episodes: {len(all_episodes)}")
        print(f"Total steps: {total_steps}")
        print(f"Total time: {time.time() - start_time:.2f}s")

        model.save("ppo_model.zip")
        print("PPO model saved to ppo_model.zip")

    elif args.approach == "tree":
        print("Running simple 'tree' approach (no PPO).")
        start_time = time.time()
        time_limit = 60
        all_episodes = []
        ep_count = 0
        while True:
            if time.time() - start_time >= time_limit:
                print("Time limit => stop tree-based approach.")
                break
            env = env_fn()
            run_tree_search(env)
            all_episodes.extend(env.all_episodes)
            ep_count += 1
            print(f"Tree-based episode {ep_count} finished.")

    else:  # MCTS
        print("Running MCTS approach. Both Player and Enemy side uses MCTS")
        start_time = time.time()
        time_limit = 60
        all_episodes = []
        ep_count = 0
        while True:
            if time.time() - start_time >= time_limit:
                print("Time limit => stop MCTS approach.")
                break
            env = env_fn()
            eps = run_mcts_episode(env, max_iterations=2000)
            if len(env.current_episode) > 0:
                env.all_episodes.append(env.current_episode)
            all_episodes.extend(env.all_episodes)
            ep_count += 1
            print(f"MCTS-based episode {ep_count} done.")

    iteration_outcomes = []
    for i, episode in enumerate(all_episodes):
        if len(episode) == 0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue
        final = episode[-1]
        rew = final["reward"]
        side = final["turn_side"]
        nbk = final.get("non_bloodwarden_kills", 0)

        # Only mention priest kills in the summary
        if rew >= 30:
            if side == "player":
                outcome_str = f"Iteration {i+1}: PLAYER kills enemy Priest => +30"
            else:
                outcome_str = f"Iteration {i+1}: ENEMY kills player Priest => +30"
        elif rew == -20:
            outcome_str = f"Iteration {i+1}: time-limit => -20"
        else:
            outcome_str = f"Iteration {i+1}: final reward={rew}, side={side}, nb_kills={nbk}"
        iteration_outcomes.append(outcome_str)

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print(f"\nSaved actions_log.npy with scenario data. Approach = {args.approach}")
    print(f"Collected {len(all_episodes)} total episodes.")
    for line in iteration_outcomes[-10:]:
        print(line)

def run_mcts_episode(env, max_iterations=2000):
    """
    Simple demonstration of how you might run an MCTS-based enemy turn
    while also controlling the player's turn with MCTS (or anything else).
    """
    obs, info = env.reset()
    done = False
    while not done:
        if env.turn_side == "enemy":
            a_idx = mcts_policy(env, max_iterations=max_iterations)
        else:
            valid = env.build_action_list()
            if not valid:
                a_idx = 0
            else:
                a_idx = random.randint(0, len(valid)-1)
        obs, reward, terminated, truncated, info = env.step(a_idx)
        done = terminated or truncated
    return env.current_episode

if __name__ == "__main__":
    main()