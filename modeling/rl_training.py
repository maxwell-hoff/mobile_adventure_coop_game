import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time

from copy import deepcopy
from itertools import combinations

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
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2


def line_of_sight(q1, r1, q2, r2, blocked_hexes, all_pieces):
    """
    True if no blocked hex or piece fully blocks the line from (q1,r1) to (q2,r2).
    We mimic your map.js approach: get all hexes along the line, skip start & end.
    If any are blocked or have a living piece => no LOS.
    """
    if q1 == q2 and r1 == r2:
        return True
    # Build the line
    N = max(abs(q2 - q1), abs(r2 - r1), abs((q1 + r1) - (q2 + r2)))
    if N == 0:
        return True
    s1 = -q1 - r1
    s2 = -q2 - r2

    line_hexes = []
    for i in range(N+1):
        t = i / N
        qf = q1 + (q2 - q1) * t
        rf = r1 + (r2 - r1) * t
        sf = s1 + (s2 - s1) * t
        # Round
        rq = round(qf)
        rr = round(rf)
        rs = round(sf)
        # Fix sum=0 if rounding is off
        qdiff = abs(rq - qf)
        rdiff = abs(rr - rf)
        sdiff = abs(rs - sf)
        if qdiff > rdiff and qdiff > sdiff:
            rq = -rr - rs
        elif rdiff > sdiff:
            rr = -rq - rs
        line_hexes.append((rq, rr))

    # Check mid points for blocking
    # skip the first & last
    for (hq, hr) in line_hexes[1:-1]:
        if (hq, hr) in blocked_hexes:
            return False
        # If any piece is exactly in that hex (and alive)
        # => blocks LOS
        # (In your logic, might want different rules, but let's do it.)
        # We'll check "all_pieces" for living piece
        for p in all_pieces:
            if not p.get("dead", False) and p["q"] == hq and p["r"] == hr:
                return False
    return True


class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment controlling both 'player' & 'enemy'.

    We replicate "map.js" logic more closely by enumerating not only
    "move/pass/necro", but also single-target or multi-target attacks, so each
    distinct target (or subset) is its own discrete action. 

    This way, the agent can actually pick e.g. 'dark_bolt' on a specific enemy's hex.
    
    Key points:
      * If the enemy Priest is killed => we instantly give +20 to current side => end.
      * If there's an enemy in range for some attack, but we skip => small negative (-0.2).
      * We rely on a _get_full_action_list() that enumerates sub-actions:
          - move to each valid hex
          - pass
          - each single_target_attack vs each valid enemy
          - each multi_target_attack vs subsets of enemies
          - necro if available
    """

    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        self._init_pieces_from_scenario(self.scenario)

        # Build all hexes
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # Instead of a fixed "actions_per_piece", we will build a dynamic list
        # of (piece_index, sub_action) in _get_full_action_list().
        # But we do need a consistent action_space size. We'll define a "max_actions"
        # so we do a stable shape. Or we can define it large enough, e.g. 500, to cover worst-case.
        # For subgrid radius=3 => 7x7=49 hexes => each piece might move to ~49 or pass or do single-target
        # If multi-target => can be big. We'll pick some big number, but in practice might do more complex approach.
        self.max_actions_for_side = 500  # hopefully big enough for all enumerations in worst case
        self.action_space = gym.spaces.Discrete(self.max_actions_for_side)

        # Observations => 2 coords per piece
        self.obs_size = 2 * len(self.all_pieces)
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0

        # For delayed spells
        self.delayedAttacks = []

        # We'll store the enumerated actions in a list => self.current_action_list
        # Each step, we rebuild it in _build_action_list() => we produce a mask accordingly.

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

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

        print("\n=== RESET ===")
        for p in self.player_pieces:
            print(f"  Player {p['label']} at ({p['q']}, {p['r']})")
        for e in self.enemy_pieces:
            print(f"  Enemy {e['label']} at ({e['q']}, {e['r']})")
        print("================")

        init_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(init_dict)

        return self._get_obs(), {}

    def step(self, action_index):
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        # Rebuild possible actions:
        valid_actions = self._build_action_list()
        if len(valid_actions) == 0:
            # forcibly end puzzle:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        if action_index < 0 or action_index >= len(valid_actions):
            # invalid => penalize
            return self._finish_step(-1.0, False, False)

        # decode
        piece_index, sub_action = valid_actions[action_index]
        piece = self.all_pieces[piece_index]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, False, False)

        # Check if we skip an attack
        # If piece could have attacked => but we didn't do an "attack" => small negative
        # We'll define "attack" as sub_action["type"] in ["single_target_attack", "multi_target_attack", "aoe"]
        could_attack = self._could_have_attacked(piece)
        do_attack = (sub_action["type"] in ["single_target_attack", "multi_target_attack", "aoe"])
        reward_mod = 0.0
        if could_attack and not do_attack:
            reward_mod -= 0.2

        # Now apply the sub_action
        final_reward = 0.0 + reward_mod
        subtype = sub_action["type"]
        if subtype == "move":
            # move
            q, r = sub_action["dest"]
            piece["q"] = q
            piece["r"] = r
        elif subtype == "pass":
            final_reward -= 0.5
        elif subtype == "aoe" and sub_action["name"] == "necrotizing_consecrate":
            # same logic as _schedule_necro
            self._schedule_necro(piece)
        elif subtype == "single_target_attack":
            # kill that one enemy
            target_idx = sub_action["target_idx"]  # index in enemy array, or we store reference
            # We'll do it by label or direct reference. Let’s store direct piece ref in the sub_action
            target_piece = sub_action["target_piece"]
            # immediate
            if target_piece is not None and not target_piece.get("dead", False):
                # kill
                self._kill_piece(target_piece)
                final_reward += 1.0  # maybe +1 per kill
        elif subtype == "multi_target_attack":
            # kill each in sub_action["targets"]
            for t in sub_action["targets"]:
                if not t.get("dead", False):
                    self._kill_piece(t)
                    final_reward += 1.0
        else:
            # e.g. "dark_bolt" or something
            # we can unify single_target/multi_target. But let's handle if we want a cast_speed
            # If some spells have cast_speed => we do delayedAttacks
            pass

        # Now do normal end checks
        rew, terminated, truncated = self._apply_end_conditions(final_reward)
        return self._finish_step(rew, terminated, truncated)

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

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _build_action_list(self):
        """
        Return a list of (piece_index, sub_action_dict).
        sub_action_dict is e.g. {"type": "move", "dest": (q,r)}
        or {"type":"single_target_attack", "target_piece":..., "range":...}
        We'll produce at most self.max_actions_for_side (some large cap).
        """
        actions = []
        living_side = [(i, pc) for (i, pc) in enumerate(self.all_pieces)
                       if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(living_side) == 0:
            return actions

        # Build blocked hex set
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}

        # For LOS checks, we'll pass in all pieces:
        all_pieces_list = self.all_pieces  # for line_of_sight

        # gather all enemy pieces for single or multi target
        if self.turn_side == "player":
            enemies = [ep for ep in self.enemy_pieces if not ep.get("dead", False)]
        else:
            enemies = [pp for pp in self.player_pieces if not pp.get("dead", False)]

        # We'll build each piece's possible sub-actions
        for (pidx, piece) in living_side:
            pclass = pieces_data["classes"][piece["class"]]
            # 1) Moves
            if "move" in pclass["actions"]:
                max_range = pclass["actions"]["move"]["range"]
                # for each hex
                for (q, r) in self.all_hexes:
                    dist = hex_distance(piece["q"], piece["r"], q, r)
                    if dist <= max_range and not self._occupied_or_blocked(q, r):
                        # also require it's not the same spot
                        if not (q == piece["q"] and r == piece["r"]):
                            actions.append((pidx, {"type": "move", "dest": (q, r)}))

            # 2) pass
            actions.append((pidx, {"type": "pass"}))

            # 3) For each named action besides 'move', we see if it's an "aoe", "single_target_attack",
            #    or "multi_target_attack".
            for aname, adata in pclass["actions"].items():
                if aname == "move":
                    continue
                # If it's necro => treat special
                if aname == "necrotizing_consecrate":
                    # always valid if piece can do it
                    actions.append((pidx, {"type": "aoe", "name": "necrotizing_consecrate"}))
                    continue

                atype = adata["action_type"]
                if atype == "single_target_attack":
                    # for each enemy in range+LOS => sub-action
                    rng = adata["range"]
                    requires_los = adata.get("requires_los", False)
                    # find enemies in range
                    for enemyP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], enemyP["q"], enemyP["r"])
                        if dist <= rng:
                            # check LOS
                            if (not requires_los) or line_of_sight(piece["q"], piece["r"],
                                                                   enemyP["q"], enemyP["r"],
                                                                   blocked_hexes,
                                                                   all_pieces_list):
                                actions.append((pidx, {
                                    "type": "single_target_attack",
                                    "action_name": aname,   # e.g. "dark_bolt"
                                    "target_piece": enemyP,
                                }))

                elif atype == "multi_target_attack":
                    # e.g. 'precise_shot' range=2, max_num_targets=2
                    rng = adata["range"]
                    max_tg = adata.get("max_num_targets", 1)
                    requires_los = adata.get("requires_los", False)
                    # gather all enemies in range+los
                    in_range_enemies = []
                    for enemyP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], enemyP["q"], enemyP["r"])
                        if dist <= rng:
                            if (not requires_los) or line_of_sight(piece["q"], piece["r"],
                                                                   enemyP["q"], enemyP["r"],
                                                                   blocked_hexes,
                                                                   all_pieces_list):
                                in_range_enemies.append(enemyP)
                    # now produce subsets up to max_tg
                    # e.g. if 2, produce all combos of size 1 or 2
                    # or you might want exactly 2. We'll do 1..max_tg
                    for size in range(1, max_tg+1):
                        for combo in combinations(in_range_enemies, size):
                            # sub-action
                            actions.append((pidx, {
                                "type": "multi_target_attack",
                                "action_name": aname,
                                "targets": list(combo),
                            }))

                elif atype == "aoe":
                    # we can do an area-based approach, but let's keep it simpler:
                    # We interpret "range" => center must be in that range from piece
                    # Then we do each center in range => sub-action
                    rng = adata["range"]
                    radius = adata["radius"]
                    requires_los = adata.get("requires_los", False)
                    # for each (q,r) => if dist <= rng => if requiresLOS => check LOS
                    # Then the effect hits enemies in radius => done
                    # This can be many subactions. We'll add them if you want.
                    # But your map.js does "click the center hex." 
                    # We'll replicate that approach:
                    for (hq, hr) in self.all_hexes:
                        dist = hex_distance(piece["q"], piece["r"], hq, hr)
                        if dist <= rng:
                            if not requires_los or line_of_sight(piece["q"], piece["r"], hq, hr,
                                                                 blocked_hexes, all_pieces_list):
                                actions.append((pidx, {
                                    "type": "aoe",
                                    "action_name": aname,
                                    "center": (hq, hr),
                                    "radius": radius
                                }))

                else:
                    # e.g. swap_position or other
                    # skipping for brevity
                    pass

        # Now we have a big list => we cap at max_actions_for_side
        return actions[: self.max_actions_for_side]

    def _occupied_or_blocked(self, q, r):
        """Return True if that hex is blocked or occupied by a living piece."""
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return True
        for p in self.all_pieces:
            if not p.get("dead", False) and p["q"] == q and p["r"] == r:
                return True
        return False

    def _could_have_attacked(self, piece):
        """
        Return True if piece has any single_target_attack, multi_target_attack, or aoe
        that can hit at least one living enemy. We do line_of_sight checks if required,
        etc. Then if the piece chooses not to do one of those actions => small negative.
        """
        if piece["side"] == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]
        if len(enemies) == 0:
            return False

        pclass = pieces_data["classes"][piece["class"]]
        for aname, adata in pclass["actions"].items():
            if aname == "move":
                continue
            atype = adata["action_type"]
            rng = adata.get("range", 0)
            requires_los = adata.get("requires_los", False)
            # single
            if atype == "single_target_attack":
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        # check LOS
                        blocked = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
                        if not requires_los or line_of_sight(piece["q"], piece["r"],
                                                             e["q"], e["r"],
                                                             blocked,
                                                             self.all_pieces):
                            return True
            elif atype == "multi_target_attack":
                # if there's at least 1 enemy in range => True
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        blocked = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
                        if not requires_los or line_of_sight(piece["q"], piece["r"],
                                                             e["q"], e["r"],
                                                             blocked,
                                                             self.all_pieces):
                            return True
            elif atype == "aoe":
                # if there's at least 1 enemy => True
                # e.g. necro range=100 => sure
                if len(enemies) > 0:
                    return True
        return False

    def _schedule_necro(self, piece):
        # basically your code for necrotizing_consecrate
        piece_class = pieces_data["classes"][piece["class"]]
        necro_data = piece_class["actions"]["necrotizing_consecrate"]
        c_speed = necro_data.get("cast_speed", 0)
        if c_speed > 0:
            tturn = self.turn_number + c_speed
            event = {
                "turn_number": self.turn_number,
                "turn_side": piece["side"],
                "reward": 0.0,
                "positions": self._log_positions(),
                "desc": f"{piece['class']} ({piece['label']}) started necro => triggers turn {tturn}"
            }
            self.current_episode.append(event)
            self.delayedAttacks.append({
                "caster_side": piece["side"],
                "caster_label": piece["label"],
                "trigger_turn": tturn,
                "action_name": "necrotizing_consecrate"
            })
        else:
            # immediate
            if piece["side"] == "enemy":
                kills = sum(1 for p in self.player_pieces if not p.get("dead", False))
                for p in self.player_pieces:
                    self._kill_piece(p)
                event = {
                    "turn_number": self.turn_number,
                    "turn_side": "enemy",
                    "reward": 2 * kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by enemy kills {kills} players"
                }
                self.current_episode.append(event)
            else:
                kills = sum(1 for e in self.enemy_pieces if not e.get("dead", False))
                for e in self.enemy_pieces:
                    self._kill_piece(e)
                event = {
                    "turn_number": self.turn_number,
                    "turn_side": "player",
                    "reward": 2 * kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by player kills {kills} enemies"
                }
                self.current_episode.append(event)

    def _check_delayed_attacks(self):
        to_remove = []
        for i, att in enumerate(self.delayedAttacks):
            if att["trigger_turn"] == self.turn_number:
                c_side = att["caster_side"]
                if c_side == "enemy":
                    kills = sum(1 for p in self.player_pieces if not p.get("dead", False))
                    for p in self.player_pieces:
                        self._kill_piece(p)
                    extra = 2 * kills
                    event_dict = {
                        "turn_number": self.turn_number,
                        "turn_side": "enemy",
                        "reward": extra,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by enemy kills {kills} player piece(s)."
                    }
                    self.current_episode.append(event_dict)
                else:
                    kills = sum(1 for e in self.enemy_pieces if not e.get("dead", False))
                    for e in self.enemy_pieces:
                        self._kill_piece(e)
                    extra = 2 * kills
                    event_dict = {
                        "turn_number": self.turn_number,
                        "turn_side": "player",
                        "reward": extra,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by player kills {kills} enemy piece(s)."
                    }
                    self.current_episode.append(event_dict)
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _apply_end_conditions(self, base_reward):
        # same as your code
        rew = base_reward
        term = False
        trunc = False

        p_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        e_alive = [e for e in self.enemy_pieces if not e.get("dead", False)]

        # priest check
        player_priest_alive = any(p["class"] == "Priest" for p in p_alive)
        enemy_priest_alive = any(e["class"] == "Priest" for e in e_alive)

        if len(p_alive) == 0 and len(e_alive) == 0:
            rew -= 10
            term = True
        else:
            if len(p_alive) == 0:
                if self.turn_side == "enemy":
                    rew += 20
                else:
                    rew -= 20
                term = True
            elif len(e_alive) == 0:
                if self.turn_side == "player":
                    rew += 20
                else:
                    rew -= 20
                term = True

        if not term:
            if self.turn_side == "player":
                if not enemy_priest_alive:
                    rew += 20
                    term = True
                elif not player_priest_alive:
                    rew -= 20
                    term = True
            else:
                if not player_priest_alive:
                    rew += 20
                    term = True
                elif not enemy_priest_alive:
                    rew -= 20
                    term = True

        if not term:
            if self.turn_number >= self.max_turns:
                rew -= 10
                trunc = True
        return rew, term, trunc

    def _kill_piece(self, piece):
        if not piece.get("dead", False):
            if piece["class"] != "BloodWarden":
                self.non_bloodwarden_kills += 1
            piece["dead"] = True
            piece["q"] = 9999
            piece["r"] = 9999

    def _get_obs(self):
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
        """
        We'll build self._build_action_list() => we produce a mask of length self.max_actions_for_side,
        with 'True' up to len( that list ) and 'False' afterwards. 
        """
        if self.done_forced:
            dummy = np.zeros(self.max_actions_for_side, dtype=bool)
            dummy[0] = True
            return dummy

        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            # ends puzzle
            dummy = np.zeros(self.max_actions_for_side, dtype=bool)
            dummy[0] = True
            return dummy

        valid_actions = self._build_action_list()
        mask = np.zeros(self.max_actions_for_side, dtype=bool)
        for i in range(len(valid_actions)):
            mask[i] = True
        return mask

    def action_masks(self):
        return self._get_action_mask()


def make_env_fn(scenario_dict):
    def _init():
        env = HexPuzzleEnv(puzzle_scenario=scenario_dict, max_turns=10)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


def main():
    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy = deepcopy(scenario)

    vec_env = DummyVecEnv([make_env_fn(scenario_copy)])
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)

    print("Training until player wins or 20 minutes pass...")

    player_side_has_won = False
    iteration_count_before = 0
    start_time = time.time()
    time_limit = 20 * 60

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
            if rew >= 20 and side == "player":
                print(f"Player side just won iteration {i+1}!")
                player_side_has_won = True
                break
        iteration_count_before = len(all_eps)
        if player_side_has_won:
            break

    # Summaries
    all_episodes = vec_env.envs[0].all_episodes
    iteration_outcomes = []
    for i, episode in enumerate(all_episodes):
        if len(episode) == 0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue
        final = episode[-1]
        rew = final["reward"]
        side = final["turn_side"]
        nbk = final.get("non_bloodwarden_kills", 0)
        if rew >= 20 and side == "player":
            outcome_str = f"Iteration {i+1}: PLAYER side WINS! (nbw_kills={nbk})"
        elif rew >= 20 and side == "enemy":
            outcome_str = f"Iteration {i+1}: ENEMY side WINS! (nbw_kills={nbk})"
        elif rew <= -20:
            outcome_str = f"Iteration {i+1}: {side} side LOSES! (nbw_kills={nbk})"
        elif rew == -10:
            outcome_str = f"Iteration {i+1}: double knockout/time-limit penalty (nbw_kills={nbk})"
        else:
            outcome_str = (f"Iteration {i+1}: final reward={rew}, side={side}, nbw_kills={nbk}")
        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for line in iteration_outcomes:
        print(line)
    print("==========================\n")

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")


if __name__ == "__main__":
    main()
