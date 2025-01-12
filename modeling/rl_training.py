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
    True if no blocked hex or living piece fully blocks the line from (q1,r1) to (q2,r2).
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
    for i in range(N + 1):
        t = i / N
        qf = q1 + (q2 - q1) * t
        rf = r1 + (r2 - r1) * t
        sf = s1 + (s2 - s1) * t
        rq = round(qf)
        rr = round(rf)
        rs = round(sf)

        # Fix rounding so q + r + s = 0
        qdiff = abs(rq - qf)
        rdiff = abs(rr - rf)
        sdiff = abs(rs - sf)
        if qdiff > rdiff and qdiff > sdiff:
            rq = -rr - rs
        elif rdiff > sdiff:
            rr = -rq - rs

        line_hexes.append((rq, rr))

    # skip the first and last
    for (hq, hr) in line_hexes[1:-1]:
        # blocked hex?
        if (hq, hr) in blocked_hexes:
            return False
        # or living piece
        for p in all_pieces:
            if not p.get("dead", False) and (p["q"], p["r"]) == (hq, hr):
                return False
    return True


class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment controlling both 'player' & 'enemy'.

    We replicate "map.js" logic more closely by enumerating:
      - move actions
      - pass
      - necrotizing_consecrate
      - single_target_attack
      - multi_target_attack
      - (optionally AOE)...

    Then in step(), we decode and apply them. We fix the 'target_idx' KeyError by actually
    using 'target_piece' or 'targets' that we stored in sub_action.

    Additional constraints:
      * Killing the enemy's Priest => immediate +20.
      * If there's at least one enemy in range for an attack but we choose no attack => -0.2.
      * If we pass => -0.5
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

        # We'll define a max so stable_baselines3 sees a fixed discrete space:
        self.max_actions_for_side = 500
        self.action_space = gym.spaces.Discrete(self.max_actions_for_side)

        # Observations
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

    def step(self, action_idx):
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        # If the current side has zero living pieces => forcibly end
        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        # Build the valid sub-action list
        valid_actions = self._build_action_list()
        if len(valid_actions) == 0:
            # no possible actions => forcibly end
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        if action_idx < 0 or action_idx >= len(valid_actions):
            # invalid => small penalty
            return self._finish_step(-1.0, False, False)

        (pidx, sub_action) = valid_actions[action_idx]
        piece = self.all_pieces[pidx]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, False, False)

        # If the piece *could* have attacked but didn't => -0.2
        could_attack = self._could_have_attacked(piece)
        is_attack = sub_action["type"] in ["single_target_attack", "multi_target_attack", "aoe"]
        reward_mod = 0.0
        if could_attack and not is_attack:
            reward_mod -= 0.2

        # apply sub_action
        atype = sub_action["type"]
        if atype == "move":
            (q, r) = sub_action["dest"]
            piece["q"] = q
            piece["r"] = r
        elif atype == "pass":
            reward_mod -= 0.5
        elif atype == "aoe" and sub_action.get("name") == "necrotizing_consecrate":
            self._schedule_necro(piece)
        elif atype == "single_target_attack":
            # Here we use "target_piece"
            target_piece = sub_action["target_piece"]  # Not target_idx
            if target_piece is not None and not target_piece.get("dead", False):
                self._kill_piece(target_piece)
                reward_mod += 1.0
        elif atype == "multi_target_attack":
            for tgt in sub_action["targets"]:
                if not tgt.get("dead", False):
                    self._kill_piece(tgt)
                    reward_mod += 1.0
        # else other action types as needed

        final_reward, terminated, truncated = self._apply_end_conditions(reward_mod)
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
            # swap side
            if self.turn_side == "player":
                self.turn_side = "enemy"
            else:
                self.turn_side = "player"
                self.turn_number += 1
            self._check_delayed_attacks()

        return self._get_obs(), reward, terminated, truncated, {}

    def _build_action_list(self):
        """
        Return a list of (piece_index, sub_action_dict).
        This enumerates moves, pass, necro, single_target, multi_target, etc.
        """
        actions = []
        living_side = [(i, pc) for (i, pc) in enumerate(self.all_pieces)
                       if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(living_side) == 0:
            return actions  # none

        # We'll gather references to enemies for attacks
        if self.turn_side == "player":
            enemies = [e for e in self.enemy_pieces if not e.get("dead", False)]
        else:
            enemies = [p for p in self.player_pieces if not p.get("dead", False)]

        # Build blocked set
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}

        for (pidx, piece) in living_side:
            pclass = pieces_data["classes"][piece["class"]]
            # 1) Move
            if "move" in pclass["actions"]:
                mrange = pclass["actions"]["move"]["range"]
                for (q, r) in self.all_hexes:
                    if (q, r) != (piece["q"], piece["r"]):
                        if hex_distance(piece["q"], piece["r"], q, r) <= mrange:
                            if not self._occupied_or_blocked(q, r):
                                actions.append((pidx, {"type": "move", "dest": (q, r)}))
            # 2) pass
            actions.append((pidx, {"type": "pass"}))

            # 3) other actions
            for aname, adata in pclass["actions"].items():
                if aname == "move":
                    continue
                if aname == "necrotizing_consecrate":
                    actions.append((pidx, {"type": "aoe", "name": "necrotizing_consecrate"}))
                    continue

                # single/multi/aoe
                atype = adata["action_type"]
                rng = adata.get("range", 0)
                requires_los = adata.get("requires_los", False)

                # Single
                if atype == "single_target_attack":
                    for enemyP in enemies:
                        dist = hex_distance(piece["q"], piece["r"], enemyP["q"], enemyP["r"])
                        if dist <= rng:
                            # check LOS
                            if (not requires_los) or line_of_sight(piece["q"], piece["r"],
                                                                   enemyP["q"], enemyP["r"],
                                                                   blocked_hexes, self.all_pieces):
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
                            if (not requires_los) or line_of_sight(piece["q"], piece["r"],
                                                                   eP["q"], eP["r"],
                                                                   blocked_hexes, self.all_pieces):
                                in_range_enemies.append(eP)
                    # produce combos
                    for size in range(1, max_tg + 1):
                        for combo in combinations(in_range_enemies, size):
                            actions.append((pidx, {
                                "type": "multi_target_attack",
                                "action_name": aname,
                                "targets": list(combo)
                            }))
                elif atype == "aoe":
                    # e.g. 'sweep', 'elemental_blast' if you want to do center-based. For brevity,
                    # let's skip enumerating all center hexes. Or do so if you prefer:
                    pass

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
        """Return True if piece has an actual single/multi/aoe attack that can hit at least one living enemy."""
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

            # single
            if atype == "single_target_attack":
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"],
                                                               blocked_hexes, self.all_pieces):
                            return True
            elif atype == "multi_target_attack":
                for e in enemies:
                    dist = hex_distance(piece["q"], piece["r"], e["q"], e["r"])
                    if dist <= rng:
                        if (not requires_los) or line_of_sight(piece["q"], piece["r"], e["q"], e["r"],
                                                               blocked_hexes, self.all_pieces):
                            return True
            elif atype == "aoe":
                # e.g. necro => if there's any living enemy => can be attacked
                if len(enemies) > 0:
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
            # immediate
            if piece["side"] == "enemy":
                kills = sum(1 for p in self.player_pieces if not p.get("dead", False))
                for p in self.player_pieces:
                    self._kill_piece(p)
                evt = {
                    "turn_number": self.turn_number,
                    "turn_side": "enemy",
                    "reward": 2 * kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by enemy kills {kills}"
                }
                self.current_episode.append(evt)
            else:
                kills = sum(1 for e in self.enemy_pieces if not e.get("dead", False))
                for e in self.enemy_pieces:
                    self._kill_piece(e)
                evt = {
                    "turn_number": self.turn_number,
                    "turn_side": "player",
                    "reward": 2 * kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by player kills {kills}"
                }
                self.current_episode.append(evt)

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
                    ed = {
                        "turn_number": self.turn_number,
                        "turn_side": "enemy",
                        "reward": extra,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by enemy kills {kills} players"
                    }
                    self.current_episode.append(ed)
                else:
                    kills = sum(1 for e in self.enemy_pieces if not e.get("dead", False))
                    for e in self.enemy_pieces:
                        self._kill_piece(e)
                    extra = 2 * kills
                    ed = {
                        "turn_number": self.turn_number,
                        "turn_side": "player",
                        "reward": extra,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by player kills {kills} enemies"
                    }
                    self.current_episode.append(ed)
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _apply_end_conditions(self, base_reward):
        rew = base_reward
        term = False
        trunc = False

        p_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        e_alive = [e for e in self.enemy_pieces if not e.get("dead", False)]

        # Priest checks
        player_priest_alive = any(p["class"] == "Priest" for p in p_alive)
        enemy_priest_alive = any(e["class"] == "Priest" for e in e_alive)

        # Both sides wiped?
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
            # If player's turn, if enemy priest is dead => +20, or if player's priest is dead => -20
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
        We produce a mask of shape (max_actions_for_side,). The first len(valid_actions) are True,
        the rest are False.
        """
        if self.done_forced:
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            # forcibly end in step => just dummy
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        val_acts = self._build_action_list()
        mask = np.zeros(self.max_actions_for_side, dtype=bool)
        for i in range(len(val_acts)):
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
    time_limit = 2 * 60

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
                # break
        iteration_count_before = len(all_eps)
        # if player_side_has_won:
        #     break

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
