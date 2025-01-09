import gymnasium as gym
import numpy as np
import random
import yaml
import os
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
    See docstring for reward structure details, and the necrotizing_consecrate ability.
    """
    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns
        self.turn_number = 1

        self._init_pieces_from_scenario(self.scenario)

        # Build a big list of all hex coords so we can index them
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q+r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # We'll define a single action space for all pieces
        self.actions_per_piece = self.num_positions + 2  # move to N hexes + pass + necro
        self.total_pieces = len(self.all_pieces)  # e.g. 3 player + 5 enemy
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observations = (q, r) for all pieces, in order: player pieces then enemy pieces
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,), dtype=np.float32
        )

        self.turn_side = "player"  # which side is currently acting?
        self.done_forced = False
        self.all_episodes = []  # store all episodes here
        self.current_episode = []  # the current episode's steps

    @property
    def all_pieces(self):
        # convenience for player + enemy
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.current_episode) > 0:
            self.all_episodes.append(self.current_episode)
            self.current_episode = []

        # restore puzzle scenario
        self.scenario = deepcopy(self.original_scenario)
        self._init_pieces_from_scenario(self.scenario)

        # DEBUG PRINT: show the scenario's initial positions
        print("=== RESET ===")
        for p in self.player_pieces:
            print(f"Player {p['label']} at ({p['q']}, {p['r']})")
        for e in self.enemy_pieces:
            print(f"Enemy {e['label']} at ({e['q']}, {e['r']})")
        print("================")

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        return self._get_obs(), {}

    def step(self, action):
        obs, rew, term, trunc, info = self._perform_action(action)
        return obs, rew, term, trunc, info

    def _perform_action(self, action):
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece
        all_pcs = self.all_pieces

        # side mismatch or dead => invalid
        if piece_index < 0 or piece_index >= len(all_pcs):
            return self._end_step(-1.0, False, False, {})

        piece = all_pcs[piece_index]
        if piece["side"] != self.turn_side or piece.get("dead", False):
            return self._end_step(-1.0, False, False, {})

        reward = 0.0
        terminated = False
        truncated = False

        # decode local_action
        if local_action < self.num_positions:
            # move
            q, r = self.all_hexes[local_action]
            if self._valid_move(piece, q, r):
                piece["q"] = q
                piece["r"] = r
            else:
                reward -= 1.0
        elif local_action == self.num_positions:
            # pass
            pass
        else:
            # necro
            if self._can_necro(piece):
                self._do_necro(piece)
            else:
                reward -= 1.0

        # Check if we fully wiped out a side
        player_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead", False)]

        if len(player_alive) == 0 and len(enemy_alive) == 0:
            # double knockout => big negative
            reward -= 10
            terminated = True
        elif len(player_alive) == 0:
            # player is dead => enemy wins
            if piece["side"] == "enemy":
                reward += 20
            else:
                reward -= 20
            terminated = True
        elif len(enemy_alive) == 0:
            # enemy is dead => player wins
            if piece["side"] == "player":
                reward += 20
            else:
                reward -= 20
            terminated = True

        if not terminated:
            # check turn limit
            if self.turn_number >= self.max_turns:
                reward -= 10
                truncated = True

        return self._end_step(reward, terminated, truncated, {})

    def _end_step(self, reward, terminated, truncated, info):
        # store step
        step_dict = {
            "turn_number": self.turn_number,
            "turn": self.turn_side,
            "reward": reward,
            "positions": self._log_positions()
        }
        self.current_episode.append(step_dict)

        # if ended
        if terminated or truncated:
            self.done_forced = True
        else:
            # swap turn
            self.turn_side = "enemy" if self.turn_side == "player" else "player"
            if self.turn_side == "player":
                self.turn_number += 1

        return self._get_obs(), reward, terminated, truncated, info

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

        all_living = [(p["q"], p["r"]) for p in self.all_pieces if not p.get("dead")]
        if (q, r) in all_living:
            return False

        return True

    def _can_necro(self, piece):
        return (piece["class"] == "BloodWarden")

    def _do_necro(self, piece):
        # kills all pieces on the opposing side
        if piece["side"] == "enemy":
            for p in self.player_pieces:
                self._kill_piece(p)
        else:
            for e in self.enemy_pieces:
                self._kill_piece(e)

    def _kill_piece(self, piece):
        if not piece.get("dead", False):
            piece["dead"] = True
            piece["q"] = 9999
            piece["r"] = 9999

    def _hex_distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2)
              + abs(r1 - r2)
              + abs((q1 + r1) - (q2 + r2))) / 2

    def _get_action_mask(self):
        mask = np.zeros(self.total_actions, dtype=bool)
        all_pcs = self.all_pieces

        for i, pc in enumerate(all_pcs):
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

        return mask

    # For SB3 MaskablePPO
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

    # quick test
    test_env = HexPuzzleEnv(puzzle_scenario=scenario_copy, max_turns=10)
    print("Observation size:", test_env.obs_size)

    def factory():
        sc = deepcopy(scenario)
        return make_env_fn(sc)

    vec_env = DummyVecEnv([factory()])

    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)
    print("Training single-agent that controls both player & enemy...")

    model.learn(total_timesteps=5000)

    # Save episodes
    all_episodes = vec_env.envs[0].all_episodes

    # -----------------------------------------------------------
    # 1) Print out each iteration's result in the terminal
    # We'll look at the final step in each episode to see who won,
    # or if it was a timeout/draw.  (Simplistic logic just using reward.)
    # -----------------------------------------------------------
    iteration_outcomes = []
    for i, episode in enumerate(all_episodes):
        if len(episode) == 0:
            outcome_str = f"Iteration {i+1}: No steps taken?"
        else:
            final_step = episode[-1]
            final_reward = final_step.get("reward", 0.0)
            side = final_step.get("turn", "?")

            # Heuristics:
            #   If final_reward >= 20 => that side presumably won
            #   If final_reward <= -20 => that side presumably lost
            #   If final_reward == -10 => probably double knockout or time limit
            #   If final_reward < 0 but > -20 => maybe partial penalty
            # (You can refine this as needed.)
            if final_reward >= 20:
                outcome_str = f"Iteration {i+1}: {side} side WINS!"
            elif final_reward <= -20:
                outcome_str = f"Iteration {i+1}: {side} side LOSES!"
            elif final_reward == -10:
                outcome_str = f"Iteration {i+1}: double knockout or time-limit penalty"
            else:
                outcome_str = f"Iteration {i+1}: final reward={final_reward}, side={side}"

        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for out in iteration_outcomes:
        print(out)
    print("==========================\n")

    # Finally, save
    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with turn-based scenario.")

if __name__ == "__main__":
    main()
