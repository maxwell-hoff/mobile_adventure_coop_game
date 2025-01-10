import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time

from copy import deepcopy

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


class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment controlling both 'player' & 'enemy'.

    * We incorporate 'cast_speed' for spells like 'necrotizing_consecrate'
      via self.delayedAttacks. The spell effect only happens after X turns.

    * We ensure we never return an all-False mask unless we also
      set a 'dummy' 1-hot so MaskablePPO doesn't crash.
    """

    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        # Initialize pieces from scenario
        self._init_pieces_from_scenario(self.scenario)

        # Precompute all hexes
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q+r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # Each piece => can do up to (num_positions) moves, +1 pass, +1 necro => total actions
        self.actions_per_piece = self.num_positions + 2
        self.total_pieces = len(self.all_pieces)  # e.g. 8 for 3 player + 5 enemy
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observations = (q,r) for each piece
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Turn variables
        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        # Logging
        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0

        # For delayed spells
        self.delayedAttacks = []  # e.g. necrotizing_consecrate with cast_speed

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # If we had a running episode, store it
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

        # Step 0 log
        init_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(init_dict)

        return self._get_obs(), {}

    def step(self, action):
        # If forcibly done, instantly return
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece
        valid_reward = 0.0

        # Basic checks
        if piece_index < 0 or piece_index >= len(self.all_pieces):
            return self._finish_step(-1.0, terminated=False, truncated=False)

        piece = self.all_pieces[piece_index]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, terminated=False, truncated=False)

        # Actually interpret the local_action
        if local_action < self.num_positions:
            # Move
            q, r = self.all_hexes[local_action]
            if self._valid_move(piece, q, r):
                piece["q"] = q
                piece["r"] = r
            else:
                valid_reward -= 1.0
        elif local_action == self.num_positions:
            # pass => minor penalty
            valid_reward -= 0.5
        else:
            # necro => see if caster can do it
            if self._can_necro(piece):
                self._schedule_necro(piece)
            else:
                valid_reward -= 1.0

        # Now do normal end-of-turn checks
        reward, terminated, truncated = self._apply_end_conditions(valid_reward)
        return self._finish_step(reward, terminated, truncated)

    def _finish_step(self, reward, terminated, truncated):
        """
        Concludes the side's step, logs the step_data, possibly flips side,
        triggers delayedAttacks if we just completed a full round (enemy->player).
        """
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
            # Switch side
            if self.turn_side == "player":
                self.turn_side = "enemy"
            else:
                self.turn_side = "player"
                self.turn_number += 1

            # After side swap, we check delayedAttacks
            self._check_delayed_attacks()

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _check_delayed_attacks(self):
        """
        For each delayed spell, if trigger_turn == self.turn_number => apply effect.
        Then remove it from self.delayedAttacks.
        """
        to_remove = []
        for i, att in enumerate(self.delayedAttacks):
            if att["trigger_turn"] == self.turn_number:
                # Attack triggers now
                caster_side = att["caster_side"]
                # If you want to check if caster is alive or not => do it here.
                # We'll do a direct effect.

                # We'll handle necro. If in future there are more spells, handle them similarly.
                # e.g. if att["action_name"] == "necrotizing_consecrate":

                if caster_side == "enemy":
                    kills = sum(1 for p in self.player_pieces if not p.get("dead",False))
                    for p in self.player_pieces:
                        self._kill_piece(p)
                    extra_reward = 2*kills
                    event_dict = {
                        "turn_number": self.turn_number,
                        "turn_side": "enemy",
                        "reward": extra_reward,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by enemy killed {kills} player piece(s)."
                    }
                    self.current_episode.append(event_dict)
                else:
                    kills = sum(1 for e in self.enemy_pieces if not e.get("dead",False))
                    for e in self.enemy_pieces:
                        self._kill_piece(e)
                    extra_reward = 2*kills
                    event_dict = {
                        "turn_number": self.turn_number,
                        "turn_side": "player",
                        "reward": extra_reward,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by player killed {kills} enemy piece(s)."
                    }
                    self.current_episode.append(event_dict)

                to_remove.append(i)

        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _schedule_necro(self, piece):
        """
        If cast_speed>0 => delayed,
        else => immediate effect
        """
        piece_class = pieces_data["classes"][piece["class"]]
        necro_data = piece_class["actions"]["necrotizing_consecrate"]
        cast_speed = necro_data.get("cast_speed", 0)

        if cast_speed > 0:
            # Delayed
            trigger_turn = self.turn_number + cast_speed
            schedule_event = {
                "turn_number": self.turn_number,
                "turn_side": piece["side"],
                "reward": 0.0,
                "positions": self._log_positions(),
                "desc": f"{piece['class']} ({piece['label']}) started necro, triggers on turn {trigger_turn}"
            }
            self.current_episode.append(schedule_event)

            self.delayedAttacks.append({
                "caster_side": piece["side"],
                "caster_label": piece["label"],
                "trigger_turn": trigger_turn,
                "action_name": "necrotizing_consecrate"
            })
        else:
            # Immediate effect
            if piece["side"] == "enemy":
                kills = sum(1 for p in self.player_pieces if not p.get("dead",False))
                for p in self.player_pieces:
                    self._kill_piece(p)
                event_dict = {
                    "turn_number": self.turn_number,
                    "turn_side": "enemy",
                    "reward": 2*kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by enemy killed {kills} players."
                }
                self.current_episode.append(event_dict)
            else:
                kills = sum(1 for e in self.enemy_pieces if not e.get("dead",False))
                for e in self.enemy_pieces:
                    self._kill_piece(e)
                event_dict = {
                    "turn_number": self.turn_number,
                    "turn_side": "player",
                    "reward": 2*kills,
                    "positions": self._log_positions(),
                    "desc": f"Immediate necro by player killed {kills} enemies."
                }
                self.current_episode.append(event_dict)

    def _apply_end_conditions(self, base_reward):
        """
        Normal end-of-episode checks: if either side is wiped => end,
        or if turn_number > max_turns => truncated => -10
        """
        player_alive = [p for p in self.player_pieces if not p.get("dead",False)]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead",False)]
        reward = base_reward
        terminated = False
        truncated = False

        if len(player_alive) == 0 and len(enemy_alive)==0:
            reward -= 10
            terminated = True
        elif len(player_alive)==0:
            if self.turn_side == "enemy":
                reward += 20
            else:
                reward -= 20
            terminated = True
        elif len(enemy_alive)==0:
            if self.turn_side == "player":
                reward += 20
            else:
                reward -= 20
            terminated = True

        if not terminated:
            if self.turn_number >= self.max_turns:
                reward -= 10
                truncated = True

        return reward, terminated, truncated

    def _valid_move(self, piece, q, r):
        """Return True if piece can move to (q,r)."""
        piece_class = pieces_data["classes"][piece["class"]]
        move_def = piece_class["actions"].get("move", None)
        if not move_def:
            return False
        max_range = move_def["range"]

        dist = self._hex_distance(piece["q"], piece["r"], q, r)
        if dist > max_range:
            return False

        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return False

        # If another piece is living there
        for pc in self.all_pieces:
            if not pc.get("dead",False) and pc["q"]==q and pc["r"]==r:
                return False

        return True

    def _can_necro(self, piece):
        return piece["class"] == "BloodWarden"

    def _kill_piece(self, piece):
        if not piece.get("dead",False):
            # If not BloodWarden => track kills
            if piece["class"]!="BloodWarden":
                self.non_bloodwarden_kills+=1
            piece["dead"]=True
            piece["q"]=9999
            piece["r"]=9999

    def _hex_distance(self,q1,r1,q2,r2):
        return (abs(q1-q2) + abs(r1-r2) + abs((q1+r1)-(q2+r2)))//2

    def _get_obs(self):
        coords=[]
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords,dtype=np.float32)

    def _log_positions(self):
        pa = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        ea = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": pa, "enemy": ea}

    def _get_action_mask(self):
        """
        We'll never produce an all-False mask while done=False.
        If we detect no valid actions => done_forced = True + dummy 1-hot.
        """
        if self.done_forced:
            # Return dummy
            dummy = np.zeros(self.total_actions, dtype=bool)
            dummy[0]=True
            return dummy

        mask = np.zeros(self.total_actions, dtype=bool)

        # Build for living pieces of current side
        side_living = [pc for pc in self.all_pieces if pc["side"]==self.turn_side and not pc.get("dead",False)]

        # If no living => we do not forcibly end here; 
        # Instead we only do 'dummy mask' => next step => environment sees side is dead => ends
        if len(side_living)==0:
            print(f"No living pieces for {self.turn_side}, no valid actions => dummy mask.")
            # We do *not* forcibly end puzzle here. We'll let step check and do so. 
            dummy = np.zeros(self.total_actions,dtype=bool)
            dummy[0]=True
            return dummy

        # Build normal
        for i, pc in enumerate(self.all_pieces):
            if pc.get("dead",False) or pc["side"]!=self.turn_side:
                continue

            base = i*self.actions_per_piece
            # for each hex => if valid move => True
            for idx,(q,r) in enumerate(self.all_hexes):
                if self._valid_move(pc,q,r):
                    mask[base+idx]=True
            # pass
            mask[base+self.num_positions] = True

            # necro
            if self._can_necro(pc):
                mask[base+self.num_positions+1]=True

        if not mask.any():
            print(f"No valid actions => dummy mask.")
            # do not forcibly end => let _apply_end_conditions handle
            dummy = np.zeros(self.total_actions,dtype=bool)
            dummy[0]=True
            return dummy

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
    time_limit = 20*60

    while True:
        model.learn(total_timesteps=1000)

        elapsed = time.time()-start_time
        if elapsed>=time_limit:
            print("Time limit reached, stopping.")
            break

        all_eps = vec_env.envs[0].all_episodes
        # Check if any newly-finished ep has final_reward>=20, side=player
        for i,ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
            if len(ep)==0: 
                continue
            final_step=ep[-1]
            if final_step["reward"]>=20 and final_step["turn_side"]=="player":
                print(f"Player side just won on iteration {i+1}!")
                player_side_has_won=True
                break
        iteration_count_before=len(all_eps)
        if player_side_has_won:
            break

    # Summarize
    all_episodes = vec_env.envs[0].all_episodes
    iteration_outcomes = []
    for i,episode in enumerate(all_episodes):
        if len(episode)==0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue
        final = episode[-1]
        rew=final["reward"]
        side=final["turn_side"]
        nbk=final.get("non_bloodwarden_kills",0)
        if rew>=20 and side=="player":
            outcome_str=f"Iteration {i+1}: PLAYER side WINS! (nbw_kills={nbk})"
        elif rew>=20 and side=="enemy":
            outcome_str=f"Iteration {i+1}: ENEMY side WINS! (nbw_kills={nbk})"
        elif rew<=-20:
            outcome_str=f"Iteration {i+1}: {side} side LOSES! (nbw_kills={nbk})"
        elif rew==-10:
            outcome_str=f"Iteration {i+1}: double knockout/time-limit penalty (nbw_kills={nbk})"
        else:
            outcome_str=(f"Iteration {i+1}: final reward={rew}, turn_side={side}, "
                         f"non_bloodwarden_kills={nbk}")
        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for o in iteration_outcomes:
        print(o)
    print("==========================\n")

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")


if __name__=="__main__":
    main()
