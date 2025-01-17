import argparse
import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time
import sys

from copy import deepcopy
from itertools import combinations
import math

# PPO-related
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

# Force a fixed seed
np.random.seed(42)
random.seed(42)

with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)
with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)

def hex_distance(q1, r1, q2, r2):
    """Cube distance in axial coords."""
    return (
        abs(q1 - q2)
        + abs(r1 - r2)
        + abs((q1 + r1) - (q2 + r2))
    ) // 2

def line_of_sight(q1, r1, q2, r2, blocked_hexes, all_pieces):
    """
    True if no blocked hex or living piece fully blocks the line [start->end].
    Skip the start & end hex in the check. If any interior hex is blocked or occupied => no LOS.
    """
    if q1 == q2 and r1 == r2:
        return True
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
        # fix rounding so q + r + s = 0
        qdiff = abs(rq - qf)
        rdiff = abs(rr - rf)
        sdiff = abs(rs - sf)
        if qdiff > rdiff and qdiff > sdiff:
            rq = -rr - rs
        elif rdiff > sdiff:
            rr = -rq - rs

        line_hexes.append((rq, rr))

    # skip first & last
    for (hq, hr) in line_hexes[1:-1]:
        if (hq, hr) in blocked_hexes:
            return False
        for p in all_pieces:
            if not p.get("dead", False) and (p["q"], p["r"]) == (hq, hr):
                return False
    return True


class HexPuzzleEnv(gym.Env):
    """
    Your existing environment code, unchanged, just copied here.
    """
    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None, randomize_positions=False):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        # Whether to randomize piece placements
        self.randomize_positions = randomize_positions

        self._init_pieces_from_scenario(self.scenario)

        # Build all hex coords
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # We'll define a max so stable_baselines3 sees a fixed discrete action space:
        self.max_actions_for_side = 500
        self.action_space = gym.spaces.Discrete(self.max_actions_for_side)

        # Observations = (q, r) for all pieces
        self.obs_size = 2 * (len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        # Logging / state
        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0
        self.delayedAttacks = []

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

    def _randomize_piece_positions(self):
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        valid_hexes = [(q, r) for (q, r) in self.all_hexes if (q, r) not in blocked_hexes]
        total_pieces_needed = len(self.player_pieces) + len(self.enemy_pieces)
        if total_pieces_needed > len(valid_hexes):
            print("[WARNING] Not enough valid hexes to place all pieces randomly!")
            return

        chosen_hexes = random.sample(valid_hexes, total_pieces_needed)
        idx = 0
        for p in self.player_pieces:
            (q, r) = chosen_hexes[idx]
            p["q"], p["r"] = q, r
            idx += 1
        for e in self.enemy_pieces:
            (q, r) = chosen_hexes[idx]
            e["q"], e["r"] = q, r
            idx += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.current_episode) > 0:
            self.all_episodes.append(self.current_episode)
        self.current_episode = []

        self.scenario = deepcopy(self.original_scenario)
        self._init_pieces_from_scenario(self.scenario)

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False
        self.non_bloodwarden_kills = 0
        self.delayedAttacks.clear()

        if self.randomize_positions:
            self._randomize_piece_positions()

        init_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(init_dict)

        return self.get_obs(), {}

    def step(self, action_idx):
        if self.done_forced:
            return self.get_obs(), 0.0, True, False, {}

        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        valid_actions = self.build_action_list()
        if len(valid_actions) == 0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        if action_idx < 0 or action_idx >= len(valid_actions):
            return self._finish_step(-1.0, False, False)

        (pidx, sub_action) = valid_actions[action_idx]
        piece = self.all_pieces[pidx]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, False, False)

        could_attack = self._could_have_attacked(piece)
        is_attack = sub_action["type"] in ["single_target_attack", "multi_target_attack", "aoe"]
        step_mod = 0.0
        if could_attack and not is_attack:
            step_mod -= 4.0

        atype = sub_action["type"]
        if atype == "move":
            (q, r) = sub_action["dest"]
            piece["q"] = q
            piece["r"] = r
        elif atype == "pass":
            step_mod -= 1.0
        elif atype == "aoe":
            if sub_action.get("name") == "necrotizing_consecrate":
                self._schedule_necro(piece)
            else:
                for tgt in sub_action["targets"]:
                    if not tgt.get("dead", False):
                        self._kill_piece(tgt)
        elif atype == "single_target_attack":
            target_piece = sub_action["target_piece"]
            if target_piece is not None and not target_piece.get("dead", False):
                self._kill_piece(target_piece)
        elif atype == "multi_target_attack":
            for tgt in sub_action["targets"]:
                if not tgt.get("dead", False):
                    self._kill_piece(tgt)
        elif atype == "swap_position":
            target_piece = sub_action["target_piece"]
            if target_piece is not None and not target_piece.get("dead", False):
                old_q, old_r = piece["q"], piece["r"]
                piece["q"], piece["r"] = target_piece["q"], target_piece["r"]
                target_piece["q"], target_piece["r"] = old_q, old_r
            else:
                step_mod -= 1.0

        final_reward, terminated, truncated = self._apply_end_conditions(step_mod)
        return self._finish_step(final_reward, terminated, truncated)

    def _finish_step(self, reward, terminated, truncated):
        step_data = {
            "turn_number": self.turn_number,
            "turn_side": self.turn_side,
            "reward": reward,
            "positions": self._log_positions()
        }
        if terminated or truncated:
            step_data["non_bloodwarden_kills"] = self.non_bloodwarden_kills
            self.done_forced = True

        self.current_episode.append(step_data)

        if not (terminated or truncated):
            if self.turn_side == "player":
                self.turn_side = "enemy"
            else:
                self.turn_side = "player"
                self.turn_number += 1
            self._check_delayed_attacks()

        obs = self.get_obs()
        return obs, reward, terminated, truncated, {}

    def build_action_list(self):
        actions = []
        living_side = [
            (i, pc)
            for (i, pc) in enumerate(self.all_pieces)
            if pc["side"] == self.turn_side and not pc.get("dead", False)
        ]
        if len(living_side) == 0:
            return actions

        if self.turn_side == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
            allies = [p for p in self.player_pieces if not p.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]
            allies = [e for e in self.enemy_pieces if not e.get("dead", False)]

        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}

        for (pidx, piece) in living_side:
            pclass = pieces_data["classes"][piece["class"]]
            if "move" in pclass["actions"]:
                mrange = pclass["actions"]["move"]["range"]
                for (q, r) in self.all_hexes:
                    if (q, r) != (piece["q"], piece["r"]):
                        if hex_distance(piece["q"], piece["r"], q, r) <= mrange:
                            if not self._occupied_or_blocked(q, r):
                                actions.append((pidx, {"type": "move", "dest": (q, r)}))

            actions.append((pidx, {"type": "pass"}))

            for aname, adata in pclass["actions"].items():
                if aname == "move":
                    continue
                if aname == "necrotizing_consecrate":
                    actions.append((pidx, {"type": "aoe", "name": "necrotizing_consecrate"}))
                    continue

                atype = adata["action_type"]
                rng = adata.get("range", 0)
                requires_los = adata.get("requires_los", False)
                ally_only = adata.get("ally_only", False)

                if atype == "single_target_attack":
                    for enemyP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], enemyP["q"], enemyP["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                enemyP["q"], enemyP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                actions.append((pidx, {
                                    "type": "single_target_attack",
                                    "action_name": aname,
                                    "target_piece": enemyP,
                                }))
                elif atype == "multi_target_attack":
                    max_tg = adata.get("max_num_targets", 1)
                    in_range_enemies = []
                    for eP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], eP["q"], eP["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                eP["q"], eP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                in_range_enemies.append(eP)
                    from itertools import combinations
                    for size in range(1, max_tg + 1):
                        for combo in combinations(in_range_enemies, size):
                            actions.append((pidx, {
                                "type": "multi_target_attack",
                                "action_name": aname,
                                "targets": list(combo)
                            }))
                elif atype == "swap_position":
                    if ally_only:
                        possible_targets = allies
                    else:
                        possible_targets = self.all_pieces
                    possible_targets = [
                        x for x in possible_targets
                        if x is not piece and not x.get("dead", False)
                    ]
                    for tgt in possible_targets:
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
                elif atype == "aoe":
                    radius = adata.get("radius", 0)
                    in_range_enemies = []
                    for eP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], eP["q"], eP["r"])
                        if dist <= radius:
                            if (not requires_los) or line_of_sight(
                                piece["q"], piece["r"],
                                eP["q"], eP["r"],
                                blocked_hexes, self.all_pieces
                            ):
                                in_range_enemies.append(eP)
                    if len(in_range_enemies) > 0:
                        actions.append((pidx, {
                            "type": "aoe",
                            "action_name": aname,
                            "name": aname,
                            "targets": in_range_enemies
                        }))
        return actions[: self.max_actions_for_side]

    def _occupied_or_blocked(self, q, r):
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return True
        for p in self.all_pieces:
            if not p.get("dead", False) and (p["q"], p["r"]) == (q, r):
                return True
        return False

    def _could_have_attacked(self, piece):
        if piece["side"] == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]
        if not enemies:
            return False

        pclass = pieces_data["classes"][piece["class"]]
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}

        for aname, adata in pclass["actions"].items():
            if aname == "move":
                continue
            atype = adata["action_type"]
            rng = adata.get("range", 0)
            requires_los = adata.get("requires_los", False)

            if atype in ["single_target_attack", "multi_target_attack"]:
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(
                            piece["q"], piece["r"],
                            e["q"], e["r"],
                            blocked_hexes, self.all_pieces
                        ):
                            return True
            elif atype == "aoe":
                radius = adata.get("radius", 0)
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= radius:
                        if (not requires_los) or line_of_sight(
                            piece["q"], piece["r"],
                            e["q"], e["r"],
                            blocked_hexes, self.all_pieces
                        ):
                            return True
        return False

    def _schedule_necro(self, piece):
        pclass = pieces_data["classes"][piece["class"]]
        necro_data = pclass["actions"]["necrotizing_consecrate"]
        c_speed = necro_data.get("cast_speed", 0)
        if c_speed > 0:
            trig = self.turn_number + c_speed
            evt = {
                "turn_number": self.turn_number,
                "turn_side": piece["side"],
                "reward": 0.0,
                "positions": self._log_positions(),
                "desc": f"{piece['class']}({piece['label']}) starts necro => triggers on turn {trig}"
            }
            self.current_episode.append(evt)
            self.delayedAttacks.append({
                "caster_side": piece["side"],
                "caster_label": piece["label"],
                "trigger_turn": trig,
                "action_name": "necrotizing_consecrate"
            })
        else:
            # immediate effect => kills all enemies
            if piece["side"] == "enemy":
                for p in self.player_pieces:
                    if not p.get("dead", False):
                        self._kill_piece(p)
            else:
                for e in self.enemy_pieces:
                    if not e.get("dead", False):
                        self._kill_piece(e)

    def _check_delayed_attacks(self):
        to_remove = []
        for i, att in enumerate(self.delayedAttacks):
            if att["trigger_turn"] == self.turn_number:
                c_side = att["caster_side"]
                if c_side == "enemy":
                    for p in self.player_pieces:
                        if not p.get("dead", False):
                            self._kill_piece(p)
                else:
                    for e in self.enemy_pieces:
                        if not e.get("dead", False):
                            self._kill_piece(e)
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _apply_end_conditions(self, base_reward):
        rew = base_reward
        term = False
        trunc = False

        p_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        e_alive = [e for e in self.enemy_pieces if not e.get("dead", False)]
        player_priest_alive = any(p["class"] == "Priest" for p in p_alive)
        enemy_priest_alive = any(e["class"] == "Priest" for e in e_alive)

        # both wiped => -30
        if len(p_alive) == 0 and len(e_alive) == 0:
            rew -= 30
            term = True
        else:
            # one side wiped => +30 if you caused it, -30 if it's your side
            if len(p_alive) == 0:
                if self.turn_side == "enemy":
                    rew += 30
                else:
                    rew -= 30
                term = True
            elif len(e_alive) == 0:
                if self.turn_side == "player":
                    rew += 30
                else:
                    rew -= 30
                term = True

        if not term:
            # priest => kill theirs => +30, lose yours => -30
            if self.turn_side == "player":
                if not enemy_priest_alive:
                    rew += 30
                    term = True
                elif not player_priest_alive:
                    rew -= 30
                    term = True
            else:
                if not player_priest_alive:
                    rew += 30
                    term = True
                elif not enemy_priest_alive:
                    rew -= 30
                    term = True

        # If we haven't ended, also check turn_number => tie => -20
        if not term:
            if self.turn_number >= self.max_turns:
                rew -= 20
                trunc = True

        return rew, term, trunc

    def _kill_piece(self, piece):
        if not piece.get("dead", False):
            if piece["side"] == self.turn_side:
                self.current_episode[-1]["reward"] += -5
            else:
                self.current_episode[-1]["reward"] += +5

            if piece["class"] == "Priest":
                self.current_episode[-1]["reward"] -= 3

            if piece["class"] != "BloodWarden":
                self.non_bloodwarden_kills += 1
            piece["dead"] = True
            piece["q"] = 9999
            piece["r"] = 9999

    def get_obs(self):
        coords = []
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords, dtype=np.float32)

    def _log_positions(self):
        pa = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        ea = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": pa, "enemy": ea}

    def _get_action_mask(self):
        if self.done_forced:
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        val_acts = self.build_action_list()
        mask = np.zeros(self.max_actions_for_side, dtype=bool)
        for i in range(len(val_acts)):
            mask[i] = True
        return mask

    def action_masks(self):
        return self._get_action_mask()


def make_env_fn(scenario_dict, randomize=False):
    def _init():
        env = HexPuzzleEnv(
            puzzle_scenario=scenario_dict,
            max_turns=10,
            randomize_positions=randomize
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


############################
#  SIMPLE MCTS IMPLEMENTATION
############################

class MCTSNode:
    """
    A node in the MCTS search tree, keyed by (obs_bytes, turn_side).
    """
    def __init__(self, obs_bytes, turn_side):
        self.obs_bytes = obs_bytes
        self.turn_side = turn_side

        # valid actions and children
        self.actions = None  # list of action indices
        # stats: action_idx -> (N, Q)
        self.stats = {}
        self.untried = []

        self.visit_sum = 0  # sum of visits across all actions

def obs_to_key(obs, turn_side):
    """
    Key for the MCTS dictionary, e.g. (obs_bytes, turn_side).
    """
    return (obs.tobytes(), turn_side)

mcts_tree = {}  # global dictionary: (obs_bytes, turn_side) -> MCTSNode

def mcts_policy(env, max_iterations=50):
    """
    Perform MCTS from the current env state, returning the best action.
    Now used for BOTH 'player' and 'enemy' sides.
    """
    # Original code had: if env.turn_side == "enemy": do random.
    # We'll remove that; we do MCTS for both sides.

    root_obs = env.get_obs()
    root_key = obs_to_key(root_obs, env.turn_side)

    # Create node if missing
    if root_key not in mcts_tree:
        node = MCTSNode(root_obs.tobytes(), env.turn_side)
        val_acts = env.build_action_list()
        node.actions = list(range(len(val_acts)))
        node.untried = list(range(len(val_acts)))
        for a_idx in node.actions:
            node.stats[a_idx] = [0, 0.0]  # [N, Q]
        mcts_tree[root_key] = node
    else:
        node = mcts_tree[root_key]

    # run expansions
    for _ in range(max_iterations):
        env_copy = deepcopy(env)
        search_path = []
        ret = mcts_search(env_copy, search_path)
        # backprop
        for (st_key, act_idx) in search_path:
            nnode = mcts_tree[st_key]
            st = nnode.stats[act_idx]
            st[0] += 1
            st[1] += (ret - st[1]) / st[0]
            nnode.visit_sum += 1

    # pick best action by highest N
    bestA, bestN = None, -1
    for a_idx, st in node.stats.items():
        if st[0] > bestN:
            bestN = st[0]
            bestA = a_idx
    return bestA if bestA is not None else 0

def mcts_search(env_copy, path, depth=0, max_depth=20):
    """
    Single MCTS iteration (selection->expansion->rollout->backprop).
    path is a list of (node_key, action_idx).
    """
    if depth >= max_depth:
        # just return final reward
        return env_copy.current_episode[-1]["reward"]

    if env_copy.done_forced:
        return env_copy.current_episode[-1]["reward"]

    obs = env_copy.get_obs()
    side = env_copy.turn_side
    node_key = obs_to_key(obs, side)

    # create node if missing
    if node_key not in mcts_tree:
        new_node = MCTSNode(obs.tobytes(), side)
        valid_acts = env_copy.build_action_list()
        new_node.actions = list(range(len(valid_acts)))
        new_node.untried = list(range(len(valid_acts)))
        for a_idx in new_node.actions:
            new_node.stats[a_idx] = [0, 0.0]  # [N, Q]
        mcts_tree[node_key] = new_node

        # expansion => pick one untried action
        if new_node.untried:
            a_idx = new_node.untried.pop()
            path.append((node_key, a_idx))

            obs2, rew, term, trunc, _ = env_copy.step(a_idx)
            if term or trunc:
                return env_copy.current_episode[-1]["reward"]
            # rollout
            r = rollout(env_copy, depth+1, max_depth)
            return r
        else:
            # no untried => fallback
            return env_copy.current_episode[-1]["reward"]
    else:
        node = mcts_tree[node_key]
        # if untried => expand
        if node.untried:
            a_idx = node.untried.pop()
            path.append((node_key, a_idx))
            obs2, rew, term, trunc, _ = env_copy.step(a_idx)
            if term or trunc:
                return env_copy.current_episode[-1]["reward"]
            return rollout(env_copy, depth+1, max_depth)
        else:
            # selection => pick child by UCB
            a_idx = best_uct_action(node)
            path.append((node_key, a_idx))

            obs2, rew, term, trunc, _ = env_copy.step(a_idx)
            if term or trunc:
                return env_copy.current_episode[-1]["reward"]
            return mcts_search(env_copy, path, depth+1, max_depth)

def best_uct_action(node, c=1.4):
    best_score, bestA = -999999.0, None
    for a_idx in node.actions:
        N, Q = node.stats[a_idx]
        if N == 0:
            return a_idx  # immediate expansion priority
        uct = Q + c * math.sqrt(math.log(node.visit_sum+1) / N)
        if uct > best_score:
            best_score = uct
            bestA = a_idx
    return bestA

def run_mcts_episode(env, max_iterations=50):
    obs, info = env.reset()
    done = False
    while not done:
        act_idx = mcts_policy(env, max_iterations=max_iterations)
        obs2, rew, term, trunc, inf = env.step(act_idx)
        done = term or trunc

    # store final ep
    if len(env.current_episode) > 0:
        env.all_episodes.append(env.current_episode)
    return env.all_episodes

def rollout(env_copy, depth, max_depth):
    # random rollout
    while depth < max_depth:
        if env_copy.done_forced:
            return env_copy.current_episode[-1]["reward"]
        valid = env_copy.build_action_list()
        if not valid:
            return env_copy.current_episode[-1]["reward"]
        a_idx = random.randint(0, len(valid)-1)
        obs2, rew, term, trunc, _ = env_copy.step(a_idx)
        if term or trunc:
            return env_copy.current_episode[-1]["reward"]
        depth += 1
    return env_copy.current_episode[-1]["reward"]


############################
#  Simple "tree" approach
############################

def tree_select_action(env, depth=1):
    valid_actions = env.build_action_list()
    if not valid_actions:
        return 0  # fallback

    best_score = -99999.0
    best_idx = 0
    for i, (pidx, sub_action) in enumerate(valid_actions):
        env_copy = deepcopy(env)
        obs_next, reward_next, terminated, truncated, info = env_copy.step(i)
        total = reward_next
        if not terminated and not truncated and depth > 1:
            # random next step
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
    # store final ep
    if len(env.current_episode) > 0:
        env.all_episodes.append(env.current_episode)


############################
# PPO
############################

def _run_one_episode(model, env):
    obs = env.reset()
    done, state = False, None
    while not done:
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        done = done


def main():
    parser = argparse.ArgumentParser(description="Hex Puzzle RL with optional randomization.")
    parser.add_argument("--randomize", action="store_true",
                        help="If set, randomize initial piece positions each reset.")
    parser.add_argument("--approach", choices=["ppo", "tree", "mcts"], default="ppo",
                        help="Which approach to use: 'ppo', 'tree', or 'mcts'. Default is ppo.")
    args = parser.parse_args()

    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy = deepcopy(scenario)

    if args.approach == "ppo":
        vec_env = DummyVecEnv([make_env_fn(scenario_copy, randomize=args.randomize)])
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=1e-5,
            n_steps=4096,
            batch_size=512,
            clip_range=0.2,
            ent_coef=0.0,
            max_grad_norm=0.3
        )

        print("Training PPO for up to 1 hour.")
        start_time = time.time()
        time_limit = 3600  # 1 hour

        iteration_count_before = 0
        while True:
            model.learn(total_timesteps=1000)
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print("Time limit => stop training.")
                break

            all_eps = vec_env.envs[0].all_episodes
            for i, ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
                if len(ep) == 0:
                    continue
                final = ep[-1]
                rew = final["reward"]
                side = final["turn_side"]
                if rew >= 30 and side == "player":
                    print(f"Player side apparently wiped out the enemy in iteration {i+1}.")
            iteration_count_before = len(all_eps)

        # gather episodes
        all_episodes = vec_env.envs[0].all_episodes
        if args.randomize:
            print("\nPerforming 1 test iteration on the FIXED scenario.")
            test_env = DummyVecEnv([make_env_fn(scenario_copy, randomize=False)])
            _run_one_episode(model, test_env)
            all_episodes += test_env.envs[0].all_episodes

    elif args.approach == "tree":
        print("Running simple 'tree' approach (no PPO).")
        start_time = time.time()
        time_limit = 60  # 1 minute
        all_episodes = []
        ep_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print("Time limit => stop tree-based approach.")
                break

            env = make_env_fn(scenario_copy, randomize=args.randomize)()
            run_tree_search(env)
            all_episodes.extend(env.all_episodes)
            ep_count += 1
            print(f"Tree-based episode {ep_count} finished, total episodes so far: {len(all_episodes)}")

    elif args.approach == "mcts":
        print("Running MCTS approach. Both Player and Enemy side uses MCTS")
        start_time = time.time()
        time_limit = 60 * 10
        all_episodes = []
        ep_count = 0


        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print("Time limit => stop MCTS approach.")
                break

            env = make_env_fn(scenario_copy, randomize=args.randomize)()
            eps = run_mcts_episode(env, max_iterations=2000)
            # obs, info = env.reset()
            # done = False

            # while not done:
            #     # call MCTS for the current turn_side (could be 'player' or 'enemy')
            #     act_idx = mcts_policy(env, max_iterations=50)
            #     obs2, rew, term, trunc, inf = env.step(act_idx)
            #     done = term or trunc

            # store final ep
            if len(env.current_episode) > 0:
                env.all_episodes.append(env.current_episode)
            all_episodes.extend(env.all_episodes)
            ep_count += 1
            print(f"MCTS-based episode {ep_count} done, total episodes so far: {len(all_episodes)}")

    # Summaries
    iteration_outcomes = []
    for i, episode in enumerate(all_episodes):
        if len(episode) == 0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue
        final = episode[-1]
        rew = final["reward"]
        side = final["turn_side"]
        nbk = final.get("non_bloodwarden_kills", 0)

        if rew >= 30:
            outcome_str = f"Iteration {i+1}: {side.upper()} side WINS (nb_kills={nbk})"
        elif rew <= -30:
            outcome_str = f"Iteration {i+1}: {side.upper()} side LOSES => -30 (nb_kills={nbk})"
        elif rew == -20:
            outcome_str = f"Iteration {i+1}: time-limit penalty => -20"
        else:
            outcome_str = f"Iteration {i+1}: final reward={rew}, side={side}, nb_kills={nbk}"
        iteration_outcomes.append(outcome_str)

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print(f"Saved actions_log.npy with scenario data. Approach = {args.approach}")
    print(f"Collected {len(all_episodes)} total episodes.")


if __name__ == "__main__":
    main()
