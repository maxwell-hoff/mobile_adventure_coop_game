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
    Single-agent environment controlling both 'player' & 'enemy' in a turn-based manner.
    
    The critical fix: we never return an all-false mask in _get_action_mask(). If no valid
    actions remain, we set `self.done_forced = True` *and* return a minimal dummy mask 
    (exactly one True). Then on the next .step() call, we see done_forced=True and return done.
    """

    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        # Build piece arrays
        self._init_pieces_from_scenario(self.scenario)

        # Build hex coordinate list
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # +1 for pass, +1 for necro
        self.actions_per_piece = self.num_positions + 2
        self.total_pieces = len(self.all_pieces)
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observations = (q, r) for each piece
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        # Logging
        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0

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
        # If done_forced, end immediately
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece
        valid_reward = 0.0

        # Check piece index
        if piece_index < 0 or piece_index >= len(self.all_pieces):
            return self._finish_step(-1.0, terminated=False, truncated=False)

        piece = self.all_pieces[piece_index]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, terminated=False, truncated=False)

        # Interpret local_action
        if local_action < self.num_positions:
            q, r = self.all_hexes[local_action]
            if self._valid_move(piece, q, r):
                piece["q"] = q
                piece["r"] = r
            else:
                valid_reward -= 1.0
        elif local_action == self.num_positions:
            # pass => small penalty
            valid_reward -= 0.5
        else:
            # necro
            if self._can_necro(piece):
                self._do_necro(piece)
            else:
                valid_reward -= 1.0

        reward, terminated, truncated = self._apply_end_conditions(valid_reward)
        return self._finish_step(reward, terminated, truncated)

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

        return self._get_obs(), reward, terminated, truncated, {}

    def _apply_end_conditions(self, base_reward):
        player_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead", False)]
        reward = base_reward
        terminated = False
        truncated = False

        # Both sides wiped => big negative
        if len(player_alive) == 0 and len(enemy_alive) == 0:
            reward -= 10
            terminated = True
        elif len(player_alive) == 0:
            if self.turn_side == "enemy":
                reward += 20
            else:
                reward -= 20
            terminated = True
        elif len(enemy_alive) == 0:
            if self.turn_side == "player":
                reward += 20
            else:
                reward -= 20
            terminated = True

        # turn limit
        if not terminated:
            if self.turn_number >= self.max_turns:
                reward -= 10
                truncated = True

        return reward, terminated, truncated

    def _valid_move(self, piece, q, r):
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

        living_positions = [(p["q"], p["r"]) for p in self.all_pieces if not p.get("dead", False)]
        if (q, r) in living_positions:
            return False

        return True

    def _can_necro(self, piece):
        return (piece["class"] == "BloodWarden")

    def _do_necro(self, piece):
        # If enemy necro => kill all players. If player necro => kill all enemies.
        if piece["side"] == "enemy":
            for p in self.player_pieces:
                self._kill_piece(p)
        else:
            for e in self.enemy_pieces:
                self._kill_piece(e)

    def _kill_piece(self, piece):
        if not piece.get("dead", False):
            if piece["class"] != "BloodWarden":
                self.non_bloodwarden_kills += 1
            piece["dead"] = True
            piece["q"] = 9999
            piece["r"] = 9999

    def _hex_distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2)
              + abs(r1 - r2)
              + abs((q1 + r1) - (q2 + r2))) / 2

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
        player_arr = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        enemy_arr = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": player_arr, "enemy": enemy_arr}

    def _get_action_mask(self):
        """
        If we produce an all-False mask but the env is not done, MaskablePPO meltdown.
        So we ensure we never produce an all-False mask while done=False.
        - If no living pieces => forcibly end => return a 1-hot "dummy" mask.
        - If after building is all-False => forcibly end => return a 1-hot "dummy" mask.
        """
        # If the env is already forcibly done, just return a 1-hot mask so distribution is valid
        if self.done_forced:
            mask = np.zeros(self.total_actions, dtype=bool)
            mask[0] = True  # dummy
            return mask

        # 1) If the side has no living pieces => forcibly end
        side_living = [pc for pc in self.all_pieces if pc["side"] == self.turn_side and not pc.get("dead", False)]
        if len(side_living) == 0:
            print(f"No living pieces for {self.turn_side} => forcibly end puzzle!")
            self.done_forced = True
            mask = np.zeros(self.total_actions, dtype=bool)
            mask[0] = True  # dummy
            return mask

        # 2) Build the mask normally
        mask = np.zeros(self.total_actions, dtype=bool)
        for i, pc in enumerate(self.all_pieces):
            if pc.get("dead", False) or pc["side"] != self.turn_side:
                continue

            base = i * self.actions_per_piece
            # moves
            for h_idx, (q, r) in enumerate(self.all_hexes):
                if self._valid_move(pc, q, r):
                    mask[base + h_idx] = True
            # pass
            mask[base + self.num_positions] = True
            # necro
            if self._can_necro(pc):
                mask[base + self.num_positions + 1] = True

        # 3) If it's all-False => forcibly end + dummy 1-hot
        if not mask.any():
            print(f"No valid actions for side={self.turn_side} => forcibly end puzzle!")
            self.done_forced = True
            mask = np.zeros(self.total_actions, dtype=bool)
            mask[0] = True  # dummy
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
    time_limit = 20 * 60  # 20 minutes

    while True:
        model.learn(total_timesteps=1000)

        # Time check
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            print("Time limit reached, stopping.")
            break

        all_eps = vec_env.envs[0].all_episodes
        # Check for player-side win
        for i, ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
            if len(ep) == 0:
                continue
            final_step = ep[-1]
            if final_step["reward"] >= 20 and final_step["turn_side"] == "player":
                print(f"Player side just won on iteration {i+1}!")
                player_side_has_won = True
                break

        iteration_count_before = len(all_eps)
        if player_side_has_won:
            break

    # Gather all episodes
    all_episodes = vec_env.envs[0].all_episodes

    # Summarize
    iteration_outcomes = []
    for i, ep in enumerate(all_episodes):
        if len(ep) == 0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue

        final_step = ep[-1]
        rew = final_step["reward"]
        side = final_step["turn_side"]
        nbw_kills = final_step.get("non_bloodwarden_kills", 0)

        if rew >= 20 and side == "player":
            outcome_str = f"Iteration {i+1}: PLAYER side WINS! (nbw_kills={nbw_kills})"
        elif rew >= 20 and side == "enemy":
            outcome_str = f"Iteration {i+1}: ENEMY side WINS! (nbw_kills={nbw_kills})"
        elif rew <= -20:
            outcome_str = f"Iteration {i+1}: {side} side LOSES! (nbw_kills={nbw_kills})"
        elif rew == -10:
            outcome_str = f"Iteration {i+1}: double knockout/time-limit penalty (nbw_kills={nbw_kills})"
        else:
            outcome_str = (f"Iteration {i+1}: final reward={rew}, turn_side={side}, "
                           f"non_bloodwarden_kills={nbw_kills}")

        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for out in iteration_outcomes:
        print(out)
    print("==========================\n")

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")


if __name__ == "__main__":
    main()
