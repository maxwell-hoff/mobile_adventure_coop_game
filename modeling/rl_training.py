import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time  # ← for measuring real time

from copy import deepcopy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

# Force a fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)
with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)

class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment controlling both 'player' & 'enemy' in a turn-based manner.
    We'll unify variables so there's only one place to track turn_number, reward, etc.
    
    Logging logic:
      - We'll store self.current_episode as a list of dicts. Each dict has:
          {
            "turn_number": <int>,
            "turn_side": <"player" or "enemy" or None>,
            "reward": <float>,
            "positions": {"player": np.array, "enemy": np.array},
            "non_bloodwarden_kills": <int>  # only in final step
          }
      - We'll add one dict at the start of each episode (turn_number=0, turn_side=None, reward=0)
        to record the initial state (the "reset" state).

    There's no separate _perform_action or _end_step method now—just one 'step()'.
    """

    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        # Keep an original scenario for resets
        self.original_scenario = deepcopy(puzzle_scenario)
        # We'll copy it into self.scenario to track changes during an episode
        self.scenario = deepcopy(puzzle_scenario)

        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        # Initialize pieces from scenario
        self._init_pieces_from_scenario(self.scenario)

        # Build list of hexes
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q+r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # Action space: for each piece, we can move to any of these hex positions or pass or do necro
        self.actions_per_piece = self.num_positions + 2  # move + pass + necro
        self.total_pieces = len(self.all_pieces)  # e.g. 3 player + 5 enemy
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observation space = (q, r) for each piece, in order of player then enemy
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,), dtype=np.float32
        )

        # Turn-based variables
        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        # Logging
        self.all_episodes = []
        self.current_episode = []

        # For tracking kills of non-BloodWarden
        self.non_bloodwarden_kills = 0

    @property
    def all_pieces(self):
        """Returns the combined player + enemy piece list."""
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

        # Debug
        print("\n=== RESET ===")
        for p in self.player_pieces:
            print(f"  Player {p['label']} at ({p['q']}, {p['r']})")
        for e in self.enemy_pieces:
            print(f"  Enemy {e['label']} at ({e['q']}, {e['r']})")
        print("================")

        # Log initial state
        initial_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(initial_dict)

        return self._get_obs(), {}

    def step(self, action):
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece
        valid_reward = 0.0

        if piece_index < 0 or piece_index >= len(self.all_pieces):
            return self._finish_step(-1.0, False, False)

        piece = self.all_pieces[piece_index]
        if piece.get("dead", False) or piece["side"] != self.turn_side:
            return self._finish_step(-1.0, False, False)

        # interpret the local_action
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

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _apply_end_conditions(self, base_reward):
        player_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead", False)]
        reward = base_reward
        terminated = False
        truncated = False

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
        return piece["class"] == "BloodWarden"

    def _do_necro(self, piece):
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
        mask = np.zeros(self.total_actions, dtype=bool)
        for i, pc in enumerate(self.all_pieces):
            if pc.get("dead", False) or pc["side"] != self.turn_side:
                continue

            base = i * self.actions_per_piece
            for h_idx, (q, r) in enumerate(self.all_hexes):
                if self._valid_move(pc, q, r):
                    mask[base + h_idx] = True
            mask[base + self.num_positions] = True  # pass
            if self._can_necro(pc):
                mask[base + self.num_positions + 1] = True
        if not mask.any():
            # This can be a debug or forcibly ended. E.g.:
            print("No valid moves found; environment will end the puzzle!")
            self.done_forced = True
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
    # 1) Load scenario
    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy = deepcopy(scenario)

    vec_env = DummyVecEnv([make_env_fn(scenario_copy)])
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)

    print("Starting training: Will end if player side wins OR if 20 minutes pass.")
    player_side_has_won = False
    iteration_count_before = 0
    start_time = time.time()
    training_time_limit = 20 * 60  # 20 minutes in seconds

    while True:
        # Train for another chunk of timesteps
        model.learn(total_timesteps=1000)

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= training_time_limit:
            print("20 minutes have passed with no forced player-side victory. Stopping training now.")
            break

        # Grab episodes
        all_episodes = vec_env.envs[0].all_episodes

        # Check if player side has won in newly-finished episodes
        for i, ep in enumerate(all_episodes[iteration_count_before:], start=iteration_count_before):
            if len(ep) == 0:
                continue
            final_step = ep[-1]
            final_reward = final_step["reward"]
            side = final_step["turn_side"]
            if final_reward >= 20 and side == "player":
                print(f"Player side just won in iteration {i+1}!")
                player_side_has_won = True
                break

        iteration_count_before = len(all_episodes)

        if player_side_has_won:
            break

    # Done training: either time is up or player side won
    all_episodes = vec_env.envs[0].all_episodes

    # Summarize
    iteration_outcomes = []
    for i, episode in enumerate(all_episodes):
        if len(episode) == 0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue

        final_step = episode[-1]
        final_reward = final_step["reward"]
        side = final_step["turn_side"]
        nbw_kills = final_step.get("non_bloodwarden_kills", 0)

        if final_reward >= 20 and side == "player":
            out_str = f"Iteration {i+1}: PLAYER side WINS! (nbw_kills={nbw_kills})"
        elif final_reward >= 20 and side == "enemy":
            out_str = f"Iteration {i+1}: ENEMY side WINS! (nbw_kills={nbw_kills})"
        elif final_reward <= -20:
            out_str = f"Iteration {i+1}: {side} side LOSES! (nbw_kills={nbw_kills})"
        elif final_reward == -10:
            out_str = f"Iteration {i+1}: double knockout or time-limit penalty (nbw_kills={nbw_kills})"
        else:
            out_str = (f"Iteration {i+1}: final reward={final_reward}, "
                       f"turn_side={side}, nbw_kills={nbw_kills}")

        iteration_outcomes.append(out_str)

    print("\n=== Iteration Outcomes ===")
    for msg in iteration_outcomes:
        print(msg)
    print("==========================\n")

    # Save the episodes log
    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with turn-based scenario.")

if __name__ == "__main__":
    main()
