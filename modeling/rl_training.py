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


# Custom Environment for Puzzle Scenarios
class HexPuzzleEnv(gym.Env):
    def __init__(self, puzzle_scenario):
        super(HexPuzzleEnv, self).__init__()
        self.puzzle_scenario = puzzle_scenario
        self.grid_radius = puzzle_scenario["subGridRadius"]

        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "enemy"]
        self.actions_log = []  # Stores each action taken

        self.observation_space = gym.spaces.Dict({
            "player_positions": gym.spaces.Box(low=-3, high=3, shape=(len(self.player_pieces), 2), dtype=np.int32),
            "enemy_positions": gym.spaces.Box(low=-3, high=3, shape=(len(self.enemy_pieces), 2), dtype=np.int32),
        })

        num_positions = (2 * self.grid_radius + 1) ** 2
        self.action_space = gym.spaces.MultiDiscrete([num_positions, num_positions])

    def reset(self):
        self.state = {
            "player_positions": np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.int32),
            "enemy_positions": np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.int32),
        }
        self.actions_log.clear()  # Reset action log for new episode
        return self.state

    def step(self, action):
        q, r = action
        reward = 0
        done = False

        if self._is_checkmate(q, r):
            reward = 10
            done = True
        else:
            reward = -1  # Penalty for non-optimal move

        if len(self.state["player_positions"]) > 0:
            self.state["player_positions"][0] = np.array([q, r], dtype=np.int32)

        self.actions_log.append({"player_move": (q, r), "reward": reward})
        return self.state, reward, done, {}

    def _is_checkmate(self, q, r):
        return (q, r) in [(0, 3), (1, -3)]


scenario = world_data["regions"][0]["puzzleScenarios"][0]
env = DummyVecEnv([lambda: HexPuzzleEnv(scenario)])

# Create PPO Model
model = PPO("MultiInputPolicy", env, verbose=1)

# Training and logging loop
print("Training PPO RL model...")
model.learn(total_timesteps=20000)
model.save("ppo_redwood_vale")

# Save actions log for debugging
actions_log_file = "actions_log.npy"
np.save(actions_log_file, env.envs[0].actions_log)
