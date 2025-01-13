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
    A single-agent environment controlling both sides (player & enemy).
    Includes:
      - move
      - pass
      - necrotizing_consecrate (aoe)
      - single_target_attack
      - multi_target_attack
      - swap_position
      - symmetrical +5/-5 kills
      - +30/-30 end-of-iteration logic
      - line_of_sight checks
      - negative penalty if a Priest is successfully attacked
    """
    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

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
        self.player_pieces = [
            p for p in scenario_dict["pieces"] if p["side"] == "player"
        ]
        self.enemy_pieces = [
            p for p in scenario_dict["pieces"] if p["side"] == "enemy"
        ]

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

        # If no living pieces => forcibly end
        side_living = [
            pc for pc in self.all_pieces
            if pc["side"] == self.turn_side and not pc.get("dead", False)
        ]
        if len(side_living) == 0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        valid_actions = self._build_action_list()
        if len(valid_actions) == 0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward, term, False)

        if action_idx < 0 or action_idx >= len(valid_actions):
            # invalid => small penalty
            return self._finish_step(-1.0, False, False)

        (pidx, sub_action) = valid_actions[action_idx]
        piece = self.all_pieces[pidx]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, False, False)

        # If we *could* have attacked but didn't => mild negative
        could_attack = self._could_have_attacked(piece)
        is_attack = sub_action["type"] in [
            "single_target_attack",
            "multi_target_attack",
            "aoe"
        ]
        step_mod = 0.0
        if could_attack and not is_attack:
            # Example: -4 if you skip an attack
            step_mod -= 4.0

        # apply sub_action
        atype = sub_action["type"]
        if atype == "move":
            (q, r) = sub_action["dest"]
            piece["q"] = q
            piece["r"] = r
        elif atype == "pass":
            step_mod -= 1.0
        elif atype == "aoe" and sub_action.get("name") == "necrotizing_consecrate":
            self._schedule_necro(piece)
        elif atype == "single_target_attack":
            target_piece = sub_action["target_piece"]
            if target_piece is not None and not target_piece.get("dead", False):
                self._kill_piece(target_piece)
        elif atype == "multi_target_attack":
            for tgt in sub_action["targets"]:
                if not tgt.get("dead", False):
                    self._kill_piece(tgt)
        elif atype == "swap_position":
            # We'll swap our 'piece' with sub_action["target_piece"].
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

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _build_action_list(self):
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
            # (1) Move
            if "move" in pclass["actions"]:
                mrange = pclass["actions"]["move"]["range"]
                for (q, r) in self.all_hexes:
                    if (q, r) != (piece["q"], piece["r"]):
                        if hex_distance(piece["q"], piece["r"], q, r) <= mrange:
                            if not self._occupied_or_blocked(q, r):
                                actions.append((pidx, {"type": "move", "dest": (q, r)}))

            # (2) pass
            actions.append((pidx, {"type": "pass"}))

            # (3) other (attack, swap, necro)
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
                    for size in range(1, max_tg + 1):
                        for combo in combinations(in_range_enemies, size):
                            actions.append((pidx, {
                                "type": "multi_target_attack",
                                "action_name": aname,
                                "targets": list(combo)
                            }))

                elif atype == "swap_position":
                    # check all possible targets
                    if ally_only:
                        possible_targets = allies
                    else:
                        possible_targets = self.all_pieces  # can swap with anyone
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
                    # e.g. other aoe logic
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
        """
        Return True if piece can do a single/multi/aoe attack that would hit at least one enemy in range.
        We only check feasibility, not whether the piece eventually chooses it.
        """
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
                if len(enemies) > 0:
                    return True
        return False

    def _schedule_necro(self, piece):
        """
        If necro has cast_speed > 0 => schedule it;
        else apply immediate effect.
        """
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
            # immediate effect
            if piece["side"] == "enemy":
                for p in self.player_pieces:
                    if not p.get("dead", False):
                        self._kill_piece(p)
            else:
                for e in self.enemy_pieces:
                    if not e.get("dead", False):
                        self._kill_piece(e)

    def _check_delayed_attacks(self):
        """
        If a necro triggers now => kill the opposing side => each kill => +5 or -5
        """
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
        """
        If one side is wiped => +30 or -30,
        if both sides => -30,
        if time limit => -20,
        if a side's Priest is dead => +30 or -30,
        else accumulate base.
        """
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
        """
        Actually kill the piece => if piece.side == self.turn_side => -5, else => +5.
        Then mark dead, record stats, etc.
        
        ADDITION: if the piece is a Priest, impose an extra negative penalty on that piece's side
        to reflect that "the Priest was successfully attacked."
        """
        if not piece.get("dead", False):
            # Normal symmetrical logic
            if piece["side"] == self.turn_side:
                # we just killed our own piece => -5
                self.current_episode[-1]["reward"] += -5
            else:
                # we killed the enemy piece => +5
                self.current_episode[-1]["reward"] += +5

            # If the piece is a Priest => impose an additional negative on that piece's side
            if piece["class"] == "Priest":
                # The side that owned this Priest experiences a further penalty
                # We'll say -3 (tweak as you see fit).
                # If the Priest's side is 'player' but we are 'enemy',
                # we *deduct* from the environment's reward. It's a single-agent environment controlling both,
                # so we simply do `-3`.
                self.current_episode[-1]["reward"] -= 3

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
        We produce a mask of shape (max_actions_for_side,). True for each valid action, False otherwise.
        If forced done => dummy 1-hot, etc.
        """
        if self.done_forced:
            mask = np.zeros(self.max_actions_for_side, dtype=bool)
            mask[0] = True
            return mask

        side_living = [
            pc for pc in self.all_pieces
            if pc["side"] == self.turn_side and not pc.get("dead", False)
        ]
        if len(side_living) == 0:
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

    print("Training for 2 minutes (demo). Now includes an extra penalty if a Priest is successfully attacked.")
    iteration_count_before = 0
    start_time = time.time()
    time_limit = 2 * 60  # 2 minutes

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

        if rew >= 30:
            outcome_str = f"Iteration {i+1}: {side.upper()} side WINS by total wipe or priest kill (nb_kills={nbk})"
        elif rew <= -30:
            outcome_str = f"Iteration {i+1}: {side.upper()} side LOSES or tie => -30 (nb_kills={nbk})"
        elif rew == -20:
            outcome_str = f"Iteration {i+1}: time-limit penalty => -20"
        else:
            outcome_str = (
                f"Iteration {i+1}: final reward={rew}, side={side}, nb_kills={nbk}"
            )
        iteration_outcomes.append(outcome_str)

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")


if __name__ == "__main__":
    main()
