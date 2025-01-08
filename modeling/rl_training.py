import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yaml
import os

# Load world and pieces YAML
with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)


class HexPuzzleEnv(gym.Env):
    def __init__(self, puzzle_scenario, max_steps=50):
        super(HexPuzzleEnv, self).__init__()
        self.puzzle_scenario = puzzle_scenario
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_steps = max_steps

        # Identify hexes
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # Distinguish sides
        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "enemy"]

        # Simple approach: move the first player piece
        self.action_space = gym.spaces.Discrete(self.num_positions)

        # Observations: positions for all player + enemy
        obs_size = 2 * (len(self.player_pieces) + len(self.enemy_pieces))
        # e.g. [p1_q, p1_r, p2_q, p2_r, e1_q, e1_r, e2_q, e2_r, ...]
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(obs_size,), dtype=np.float32
        )

        # For logging moves
        self.actions_log = []
        self.reset()

    def reset(self):
        self.steps_taken = 0
        # Could randomize piece positions or just load from scenario
        self.player_positions = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        self.enemy_positions = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        self.actions_log.clear()
        return self._get_obs()

    def step(self, action):
        self.steps_taken += 1
        done = False
        reward = 0

        # Convert action index -> (q, r)
        q, r = self.all_hexes[action]

        # Check if it's valid or leads to checkmate
        if self._is_valid_move(q, r):
            # Move the first player piece
            self.player_positions[0] = np.array([q, r], dtype=np.float32)

            if self._is_checkmate(q, r):
                reward += 10
                done = True
            else:
                reward += 0.1
        else:
            reward -= 1

        # For demonstration: we do a trivial enemy move or do nothing
        # If you want to handle a real enemy turn, you'd do it here.

        # If max steps or game over
        if self.steps_taken >= self.max_steps or self._check_game_over():
            done = True
        
        all_positions = {
            "player": self.player_positions.copy(),
            "enemy": self.enemy_positions.copy()
        }
        
        # Log the action
        self.actions_log.append({
            "turn": "player",
            "move": (q, r),
            "reward": reward,
            "positions": all_positions 
        })

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Flatten player + enemy positions
        flat_player = self.player_positions.flatten()
        flat_enemy = self.enemy_positions.flatten()
        return np.concatenate([flat_player, flat_enemy], axis=0)

    def _is_valid_move(self, q, r):
        # Example: within 1 hex from current position
        cur_q, cur_r = self.player_positions[0]
        dist = self._hex_distance(cur_q, cur_r, q, r)
        if dist > 1:
            return False

        # Check if blocked
        blocked_hexes = {(b["q"], b["r"]) for b in self.puzzle_scenario.get("blockedHexes", [])}
        if (q, r) in blocked_hexes:
            return False

        return True

    def _is_checkmate(self, q, r):
        # Placeholder logic
        # e.g. if (q, r) in some special list
        # or if it captures an enemy piece, etc.
        # For now just check if we stepped on (0,3) or (1,-3)
        check_positions = [(0,3), (1,-3)]
        return (q, r) in check_positions

    def _check_game_over(self):
        # Could check if no enemies remain, or no player pieces remain, etc.
        return False

    @staticmethod
    def _hex_distance(q1, r1, q2, r2):
        return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) / 2


scenario = world_data["regions"][0]["puzzleScenarios"][0]
env = DummyVecEnv([lambda: HexPuzzleEnv(scenario, max_steps=50)])

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")

print("Training PPO RL model...")
model.learn(total_timesteps=20000)
model.save("ppo_redwood_vale")

# Save actions log for visualization
actions_log_file = "actions_log.npy"
# We only need to save the log from env.envs[0]
np.save(actions_log_file, env.envs[0].actions_log)
print("Training complete. Actions log saved!")
