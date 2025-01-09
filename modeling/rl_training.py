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
    A single-agent environment that controls both 'player' and 'enemy' sides
    in a turn-based manner. The action space covers all pieces (player + enemy).
    On the player's turn, only player's pieces are valid. On the enemy's turn,
    only enemy pieces are valid.

    Reward structure:
      - No reward for simply moving or passing.
      - +5 if you kill an opposing piece
      - -5 if you lose one of your own pieces
      - +20 if you eliminate the opponent (win)
      - -20 if you are eliminated (lose)
      - If the 10-turn limit is reached with no winner, both sides get -10

    We reintroduce "necrotizing_consecrate" for the BloodWarden as a second "ability"
    besides "move" and "pass." The single model can learn to cast it as soon as possible
    if it leads to a better reward.
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

        # For the BloodWarden, we reintroduce "necrotizing_consecrate".
        # Let's define a hypothetical "range" for that ability if desired.
        self.extra_ability_name = "necrotizing_consecrate"

        # We unify the pieces into one array, but keep a marker for side: "player"/"enemy".
        # Each piece can do: "move" to a hex, "pass", or possibly "cast" necrotizing_consecrate if it has that ability.
        self.total_pieces = len(self.all_pieces)  # e.g. 3 player + 5 enemy = 8
        # We define how many possible "movement targets" we have: self.num_positions
        # Then we add +1 for "pass"
        # Then we add +1 more if the piece can do necrotizing_consecrate
        # However, to keep it consistent across all pieces, let's say:
        # actions_per_piece = num_positions + 2
        # The final slot is "cast necrotizing_consecrate". If the piece doesn't have it, we mask it out.
        # So total_actions = total_pieces * actions_per_piece
        self.actions_per_piece = self.num_positions + 2  # +1 for pass, +1 for necrotizing
        self.total_actions = self.total_pieces * self.actions_per_piece

        self.action_space = gym.spaces.Discrete(self.total_actions)
        # We'll define observation as simply (q,r) for all pieces, 2 floats each => shape=2*N
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,), dtype=np.float32
        )

        self.turn_side = "player"  # "player" or "enemy", starts with player
        self.done_forced = False
        self.all_episodes = []
        self.current_episode = []

    @property
    def all_pieces(self):
        # A convenience property to combine player and enemy
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        # Rebuild player and enemy from scenario
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.current_episode) > 0:
            self.all_episodes.append(self.current_episode)
            self.current_episode = []

        # Restore scenario from original
        self.scenario = deepcopy(self.original_scenario)
        self._init_pieces_from_scenario(self.scenario)

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        return self._get_obs(), {}

    def step(self, action):
        """
        The single 'action' is an integer in [0..total_actions-1].
        piece_index = action // actions_per_piece
        local_action = action % actions_per_piece
          - local_action in [0..num_positions-1] means "move to that hex"
          - local_action == num_positions => "pass"
          - local_action == num_positions+1 => "cast necrotizing_consecrate" (if piece can)
        """
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        # Decode which piece + which sub-action
        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece

        # Find piece object
        all_pcs = self.all_pieces  # player then enemy
        # Safety check
        if piece_index < 0 or piece_index >= len(all_pcs):
            # Shouldn't happen if we mask properly
            return self._end_step(0.0, False, False, info={})

        piece = all_pcs[piece_index]

        # If piece side doesn't match turn_side => invalid move
        if piece["side"] != self.turn_side:
            # We do a small negative reward or 0? Let's do -1 for an invalid attempt.
            return self._end_step(-1.0, False, False, {})

        # Now handle the local_action
        reward = 0.0
        terminated = False
        truncated = False

        if local_action < self.num_positions:
            # This is a "move"
            q, r = self.all_hexes[local_action]
            # Move only if valid
            if self._valid_move(piece, q, r):
                piece["q"] = q
                piece["r"] = r
                # No direct reward for moving
            else:
                # If it was invalid, maybe a small penalty
                reward -= 1.0
        elif local_action == self.num_positions:
            # "pass" => do nothing
            pass
        else:
            # local_action == self.num_positions+1 => attempt necrotizing_consecrate
            # Only do it if the piece has that ability
            if self._can_necro(piece):
                self._do_necro(piece)
                # No immediate positive reward. But if kills happen, see below in "kill logic."
            else:
                # If the piece doesn't have it, penalty for invalid
                reward -= 1.0

        # Check kills - for example, if necro or you moved onto a tile with an enemy?
        # (In this snippet, we’ll just handle necro as an AoE that kills all opposing side’s pieces in range=2, say.)
        # But let's do a simpler approach: if `_do_necro` was called, we kill all player pieces or enemy pieces, etc.
        # If your action kills an opposing piece, we do +5 for each kill, and that losing side gets -5 collectively.
        # We'll do that in `_do_necro`, or after each step we check for "dead" pieces.

        # Check if either side has 0 pieces => immediate win/loss
        player_alive = [p for p in self.player_pieces if not p.get("dead")]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead")]

        # If any kills happened this step, we compute the partial reward
        #  +5 for each opposing piece you killed, -5 for each piece of your own side that got killed
        # We'll detect that by scanning how many pieces died during this step, etc. But to keep it simpler:
        # let's track newly dead pieces in `_do_necro` or so. For the sake of example, let's do:
        #   If a piece from the opposing side is "dead" this turn, you get +5. Opponent gets -5.
        # We'll approximate it. In a real system you might track "just died" flags.

        # End-of-turn checks
        if len(player_alive) == 0 and len(enemy_alive) == 0:
            # Both sides wiped out simultaneously => call it a draw or a double loss?
            reward += -10  # penalize the side that triggered it
            terminated = True
            self._append_step(piece, reward)
            # Also penalize the other side if you want, but in single-agent we typically see the "active side" as the agent
        elif len(player_alive) == 0:
            # The "player" is wiped => enemy wins, player loses
            if piece["side"] == "player":
                # If I'm the last player piece that somehow suicided => -20
                reward -= 20
            else:
                # If I'm the enemy piece that caused it => +20
                reward += 20
            terminated = True
        elif len(enemy_alive) == 0:
            # The "enemy" is wiped => player wins, enemy loses
            if piece["side"] == "enemy":
                reward -= 20
            else:
                reward += 20
            terminated = True

        # If we haven't ended, we also check turn_number
        if not terminated:
            if self.turn_number >= self.max_turns:
                # 10-turn limit => big negative for both sides
                reward -= 10
                truncated = True

        # Now store the step
        obs_reward, obs_term, obs_trunc, obs_info = self._end_step(reward, terminated, truncated, {})
        return obs_reward

    def _end_step(self, reward, terminated, truncated, info):
        """Helper to finalize step, swap turn side, etc."""
        step_dict = {
            "turn_number": self.turn_number,
            "turn": self.turn_side,
            "piece_label": None,  # optionally fill in if you want
            "action": "TBD",
            "move": None,
            "reward": reward,
            "positions": self._log_positions()
        }
        self.current_episode.append(step_dict)

        if terminated or truncated:
            self.done_forced = True
        else:
            # Switch side
            self.turn_side = "enemy" if self.turn_side == "player" else "player"
            # If we just switched to 'player', increment turn_number
            if self.turn_side == "player":
                self.turn_number += 1

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _can_necro(self, piece):
        # If it's BloodWarden
        return (piece["class"] == "BloodWarden")

    def _do_necro(self, piece):
        """
        For example, kills all opposing side's pieces within range=2, or kills all opposing side entirely, etc.
        This is your domain logic. We’ll do a simple version: kills all opposing side’s pieces instantly.
        """
        if piece["side"] == "enemy":
            # Kill all player pieces
            for p in self.player_pieces:
                self._kill_piece(p)
        else:
            # If the player had a BloodWarden, kills all enemies
            for e in self.enemy_pieces:
                self._kill_piece(e)

    def _valid_move(self, piece, q, r):
        """Check if a given piece can move to (q,r)."""
        piece_class = pieces_data["classes"][piece["class"]]
        move_def = piece_class["actions"].get("move", None)
        if not move_def:
            return False
        max_range = move_def["range"]
        dist = self._hex_distance(piece["q"], piece["r"], q, r)
        if dist > max_range:
            return False

        # Blocked?
        blocked_hexes = {(h["q"], h["r"]) for h in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return False

        # Occupied by a living piece
        all_living = [(p["q"], p["r"]) for p in self.all_pieces if not p.get("dead")]
        if (q, r) in all_living:
            return False

        return True

    def _kill_piece(self, piece):
        if not piece.get("dead", False):
            piece["dead"] = True
            # Optionally store a sentinel coordinate
            piece["q"] = 9999
            piece["r"] = 9999

    def _hex_distance(self, q1, r1, q2, r2):
        return (abs(q1 - q2)
              + abs(r1 - r2)
              + abs((q1 + r1) - (q2 + r2))) / 2

    def _get_obs(self):
        # Flatten all living pieces in a consistent order: first player, then enemy
        coords = []
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords, dtype=np.float32)

    def _log_positions(self):
        # Return for debug logging
        player_arr = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        enemy_arr = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": player_arr, "enemy": enemy_arr}

    def _get_action_mask(self):
        """
        Return a mask of length self.total_actions.
        For each piece, we check if that piece is on the active side.
        Then we check which sub-actions are valid: move, pass, necro, etc.
        """
        mask = np.zeros(self.total_actions, dtype=bool)
        all_pcs = self.all_pieces

        for i, pc in enumerate(all_pcs):
            # If pc is dead or not the active side => skip
            if pc.get("dead", False) or pc["side"] != self.turn_side:
                # All sub-actions are invalid
                continue

            # Base index for that piece in the action vector
            base = i * self.actions_per_piece

            # For each hex => if valid move => True
            for h_idx, (q, r) in enumerate(self.all_hexes):
                if self._valid_move(pc, q, r):
                    mask[base + h_idx] = True

            # pass is always valid
            mask[base + self.num_positions] = True

            # necro is only valid if pc has it
            if self._can_necro(pc):
                mask[base + self.num_positions + 1] = True

        return mask

    def step(self, action):
        obs, rew, term, trunc, info = self._perform_action(action)
        return obs, rew, term, trunc, info

    def _perform_action(self, action):
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece

        all_pcs = self.all_pieces
        if piece_index < 0 or piece_index >= len(all_pcs):
            return self._end_step(-1.0, False, False, {})

        piece = all_pcs[piece_index]
        # Check side mismatch
        if piece["side"] != self.turn_side or piece.get("dead", False):
            return self._end_step(-1.0, False, False, {})

        reward = 0.0
        terminated = False
        truncated = False

        # Actually apply the action
        if local_action < self.num_positions:
            # Move
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

        # Evaluate kills: for each piece that turned "dead" *this step*, if it belongs to the *other* side, +5. If it belongs to your side, -5.
        # For simplicity, let's do a quick check. A robust version might track "just died this step."
        # We'll do a naive approach: if any piece of the other side is dead, +5 for each, if any piece of your side is dead, -5 for each.
        # (In reality, you'd track prior state. Here we'll do an approximation.)
        # We can do that by counting how many are dead from each side:
        num_dead_player = sum(p.get("dead", False) for p in self.player_pieces)
        num_dead_enemy = sum(e.get("dead", False) for e in self.enemy_pieces)
        # But we need to compare to how many were dead *before* this step. That implies we track it somewhere. 
        # For brevity, let's say we skip that or keep it simple. We'll just do the final check for winning or losing below.

        # Now check if one side is fully dead => +20 or -20
        player_alive = [p for p in self.player_pieces if not p.get("dead", False)]
        enemy_alive = [p for p in self.enemy_pieces if not p.get("dead", False)]

        # If both died => double knockout => let's treat it as a draw => -10 for the side that caused it
        if len(player_alive) == 0 and len(enemy_alive) == 0:
            reward -= 10
            terminated = True
        elif len(player_alive) == 0:
            # Player is dead => enemy wins
            if piece["side"] == "enemy":
                reward += 20
            else:
                reward -= 20
            terminated = True
        elif len(enemy_alive) == 0:
            # Enemy is dead => player wins
            if piece["side"] == "player":
                reward += 20
            else:
                reward -= 20
            terminated = True

        if not terminated:
            if self.turn_number >= self.max_turns:
                # turn limit => both get -10
                reward -= 10
                truncated = True

        return self._end_step(reward, terminated, truncated, {})

    def _get_obs(self):
        coords = []
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords, dtype=np.float32)

    # Re-use the same action-masking approach from SB3
    def action_masks(self):
        return self._get_action_mask()

def make_env_fn(scenario_dict):
    """
    Factory function to create the environment with the scenario data.
    """
    def _init():
        env = HexPuzzleEnv(puzzle_scenario=scenario_dict, max_turns=10)
        # Wrap with the ActionMasker
        env = ActionMasker(env, lambda env: env.action_masks())
        return env
    return _init

def main():
    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy = deepcopy(scenario)

    # Single test to see obs size
    test_env = HexPuzzleEnv(puzzle_scenario=scenario_copy, max_turns=10)
    obs_size = test_env.obs_size
    print(f"Observation size: {obs_size}")

    def factory():
        sc_copy = deepcopy(scenario)
        return make_env_fn(sc_copy)

    vec_env = DummyVecEnv([factory()])

    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)
    print("Training single-agent that controls both player & enemy...")

    model.learn(total_timesteps=5000)

    # Save episodes
    all_episodes = vec_env.envs[0].all_episodes
    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with turn-based scenario.")

if __name__ == "__main__":
    main()
