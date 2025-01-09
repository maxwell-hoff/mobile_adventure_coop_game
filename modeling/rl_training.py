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

with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)
with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)

class HexPuzzleEnv(gym.Env):
    def __init__(self, puzzle_scenario, cast_speed=2, max_turns=5, render_mode=None):
        super().__init__()
        # Keep a copy for resets
        self.original_scenario = deepcopy(puzzle_scenario)
        # Also keep a 'live' scenario pointer so we can reference blockedHexes
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns
        self.cast_speed = cast_speed
        self.turn_number = 1

        # We build self.player_pieces / self.enemy_pieces from the puzzle scenario
        # but let's store the original version in self.original_scenario for resets
        self._init_pieces_from_scenario(puzzle_scenario)

        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q+r) <= self.grid_radius:
                    self.all_hexes.append((q,r))
        self.num_positions = len(self.all_hexes)

        self.n_player_pieces = len(self.player_pieces)
        self.actions_per_piece = self.num_positions + 1  # +1 for pass
        self.total_actions = self.n_player_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # obs size = 2*(n_player + n_enemy)
        self.obs_size = 2*(len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,), dtype=np.float32
        )

        self.delayedAttacks = []
        self.all_episodes = []
        self.current_episode = []

    def _init_pieces_from_scenario(self, puzzle_scenario):
        # read from puzzle_scenario
        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"]=="player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"]=="enemy"]
        self.bloodwarden = None
        for epiece in self.enemy_pieces:
            if epiece["class"] == "BloodWarden":
                self.bloodwarden = epiece
                break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # If we have an ongoing episode, store it
        if len(self.current_episode)>0:
            self.all_episodes.append(self.current_episode)
            self.current_episode = []

        # restore puzzle scenario from self.original_scenario
        puzzle_scenario_copy = deepcopy(self.original_scenario)
        self.scenario = puzzle_scenario_copy

        self._init_pieces_from_scenario(puzzle_scenario_copy)

        self.delayedAttacks.clear()
        self.turn_number = 1
        self.is_player_turn = True
        self.done_forced = False

        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        # if done_forced, return
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, info

        if not self.is_player_turn:
            # enemy turn (similar to your code) ...
            # we do necrotizing logic or idle
            step_dict = {
                "turn_number": self.turn_number,
                "turn": "enemy",
                "piece_label": None,
                "action": "enemy_ai_not_implemented",
                "move": None,
                "reward": 0.0,
                "positions": self._log_positions()
            }
            self.current_episode.append(step_dict)

            self.is_player_turn = True
            self.turn_number += 1
            extra = self._handle_delayed_attacks()  # see below
            reward += extra
            # Check if one side is gone
            if len(self.player_pieces)==0:
                reward -= 10
                terminated = True
            elif len(self.enemy_pieces)==0:
                reward += 10
                terminated = True

            if self.turn_number > self.max_turns and not terminated:
                reward -= 5
                truncated = True

            if terminated or truncated:
                self.done_forced=True
            return self._get_obs(), reward, terminated, truncated, info

        # else if it's player turn:
        piece_idx, target_idx = divmod(action, self.actions_per_piece)
        if piece_idx < 0 or piece_idx >= len(self.player_pieces):
            # Should never happen if we mask invalid actions
            reward -= 2
            step_dict = {
                "turn_number": self.turn_number,
                "turn": "player",
                "piece_label": "invalid_piece_idx",
                "action": "invalid",
                "move": None,
                "reward": reward,
                "positions": self._log_positions()
            }
            self.current_episode.append(step_dict)
        else:
            piece = self.player_pieces[piece_idx]
            if target_idx == self.num_positions:
                # pass
                step_dict = {
                    "turn_number": self.turn_number,
                    "turn": "player",
                    "piece_label": piece["label"],
                    "action": "pass",
                    "move": None,
                    "reward": reward,
                    "positions": self._log_positions()
                }
                self.current_episode.append(step_dict)
            else:
                # Move
                q, r = self.all_hexes[target_idx]
                # we assume it's valid because we masked out invalid
                piece["q"] = q
                piece["r"] = r
                reward += 0.1

                # Check if puzzle is solved
                if self._is_win(piece):
                    reward += 10
                    terminated = True

                step_dict = {
                    "turn_number": self.turn_number,
                    "turn": "player",
                    "piece_label": piece["label"],
                    "action": "move",
                    "move": (q, r),
                    "reward": reward,
                    "positions": self._log_positions()
                }
                self.current_episode.append(step_dict)

        # Now see if the player has used all pieces this turn
        # We'll define "1 environment step" = "1 piece action." 
        # Once the player has used all their pieces => switch to enemy
        used_player_moves_this_turn = len([p for p in self.current_episode if p["turn_number"] == self.turn_number and p["turn"] == "player"])
        if used_player_moves_this_turn >= len(self.player_pieces):
            # Player done => enemy turn
            self.is_player_turn = False

        # Possibly check if we ended the puzzle
        if self.turn_number > self.max_turns and not terminated:
            reward -= 5
            truncated = True

        if terminated or truncated:
            self.done_forced = True

        return self._get_obs(), reward, terminated, truncated, info

    def _get_action_mask(self, env=None):
        """
        Returns a boolean mask of length self.total_actions = n_player_pieces * (num_positions + 1).
        We'll set 'True' for valid actions, 'False' for invalid.

        If it's enemy turn, we can't pick any action => all False
        If it's player turn, we mark only the valid piece/target combos as True.

        Args:
            env: The environment instance (unused, but required by ActionMasker)
        """
        mask = np.zeros((self.total_actions,), dtype=bool)

        if not self.is_player_turn:
            # no valid actions on enemy turn
            return mask

        # For each piece index in [0..n_player_pieces), 
        # For each target index in [0..num_positions], 
        # check if it's valid:
        for p_idx in range(len(self.player_pieces)):
            piece = self.player_pieces[p_idx]
            # If we haven't used that piece yet this turn
            used_this_piece = any(
                stepd["turn_number"] == self.turn_number
                and stepd["turn"] == "player"
                and stepd["piece_label"] == piece["label"]
                for stepd in self.current_episode
            )
            if used_this_piece:
                # piece already used => skip
                continue

            # The pass action is always valid
            pass_action_idx = p_idx * self.actions_per_piece + self.num_positions
            mask[pass_action_idx] = True

            # For each hex => check if it's a valid move
            for t_idx in range(self.num_positions):
                q, r = self.all_hexes[t_idx]
                if self._valid_move(piece, q, r):
                    a_idx = p_idx * self.actions_per_piece + t_idx
                    mask[a_idx] = True

        return mask

    def _handle_delayed_attacks(self):
        extra_reward = 0
        new_delayed = []
        for att in self.delayedAttacks:
            if att["turn_to_trigger"] == self.turn_number:
                if att["type"] == "necrotizing_consecrate":
                    # kill all players
                    for piece in self.player_pieces:
                        self._kill_player_piece(piece)
                    extra_reward -= 2  # e.g. if you want a penalty
            else:
                new_delayed.append(att)
        self.delayedAttacks = new_delayed
        return extra_reward

    def _is_win(self, piece):
        """
        Suppose a 'win' if the player piece is at some special location,
        or if all enemy are destroyed. For example:
        """
        if len(self.enemy_pieces) == 0:
            return True
        # check location-based
        if (piece["q"], piece["r"]) == (1, -3):
            return True
        return False

    def _valid_move(self, piece, q, r):
        """
        Only used to build the action mask. If it's invalid, 
        we won't let the agent pick it.
        """
        # 1) check if piece is already used this turn
        #    => we skip that in get_action_mask, see logic there.

        # 2) range check
        pieceClass = pieces_data["classes"][piece["class"]]
        move_def = pieceClass["actions"].get("move", None)
        if not move_def:
            return False
        max_range = move_def["range"]

        dist = self._hex_distance(piece["q"], piece["r"], q, r)
        if dist > max_range:
            return False

        # 3) blocked
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return False

        # 4) occupied
        all_positions = [(p["q"], p["r"]) for p in (self.player_pieces + self.enemy_pieces)]
        if (q, r) in all_positions:
            return False

        return True

    def _log_positions(self):
        # Return dict with positions for the visualization
        player_array = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        enemy_array = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": player_array, "enemy": enemy_array}

    def _kill_player_piece(self, piece):
        # Instead of removing from self.player_pieces, mark dead
        piece["dead"] = True
        # optionally store a sentinel position
        piece["q"] = 9999
        piece["r"] = 9999

    def _get_obs(self):
        coords = []
        # We always have 3 'player_pieces' in the list
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        # We always have 5 'enemy_pieces'
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords, dtype=np.float32)
    
    def _hex_distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2)
              + abs(r1 - r2)
              + abs((q1 + r1) - (q2 + r2))) / 2


###############################
#  TRAINING SCRIPT
###############################
def main():
    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    
    # Create a single environment first to get the correct observation size
    test_env = HexPuzzleEnv(puzzle_scenario=scenario, cast_speed=2, max_turns=5)
    obs_size = test_env.obs_size
    
    # Now create the vectorized environment with the same scenario
    def make_env():
        def _init():
            # Create a deep copy of the scenario to avoid sharing state
            scenario_copy = {
                "subGridRadius": scenario["subGridRadius"],
                "pieces": [piece.copy() for piece in scenario["pieces"]],
                "blockedHexes": [hex.copy() for hex in scenario["blockedHexes"]]
            }
            print(f"Original scenario pieces: {len(scenario['pieces'])}")
            print(f"Copied scenario pieces: {len(scenario_copy['pieces'])}")
            env_instance = HexPuzzleEnv(puzzle_scenario=scenario_copy, cast_speed=2, max_turns=5)
            print(f"Environment pieces - player: {len(env_instance.player_pieces)}, enemy: {len(env_instance.enemy_pieces)}")
            # Verify observation size matches
            if env_instance.obs_size != obs_size:
                raise ValueError(f"Observation size mismatch: got {env_instance.obs_size} but expected {obs_size}. This likely means the number of pieces changed.")
            env_instance = ActionMasker(env_instance, "_get_action_mask")
            return env_instance
        return _init

    vec_env = DummyVecEnv([make_env()])

    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)

    print("Training with action masking so only valid moves are chosen...")
    model.learn(total_timesteps=5000)

    # Save the episodes from the vectorized environment
    all_episodes = vec_env.envs[0].all_episodes
    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with turn-based scenario.")

if __name__ == "__main__":
    main()
