import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yaml
import os
import pygame

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

        # Count player and enemy pieces separately
        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "enemy"]
        
        # Create observation space with correct shapes
        self.observation_space = gym.spaces.Dict({
            "player_positions": gym.spaces.Box(
                low=-3, 
                high=3, 
                shape=(len(self.player_pieces), 2), 
                dtype=np.int32
            ),
            "enemy_positions": gym.spaces.Box(
                low=-3, 
                high=3, 
                shape=(len(self.enemy_pieces), 2), 
                dtype=np.int32
            ),
        })

        # Actions: select a hex coordinate (q, r) for each move
        num_positions = (2 * self.grid_radius + 1) ** 2
        self.action_space = gym.spaces.MultiDiscrete([num_positions, num_positions])

    def reset(self):
        # Convert positions to numpy arrays with correct shapes
        self.state = {
            "player_positions": np.array([
                [piece["q"], piece["r"]] for piece in self.player_pieces
            ], dtype=np.int32),
            "enemy_positions": np.array([
                [piece["q"], piece["r"]] for piece in self.enemy_pieces
            ], dtype=np.int32)
        }
        return self.state

    def step(self, action):
        q, r = action
        # Reward logic: positive for checkmate, negative for wrong moves
        reward = 0
        done = False

        # Example: check if action leads to checkmate
        if self._is_checkmate(q, r):
            reward = 10
            done = True
        else:
            reward = -1  # Penalty for non-optimal move

        # Update only the first player piece position
        if len(self.state["player_positions"]) > 0:
            self.state["player_positions"][0] = np.array([q, r], dtype=np.int32)
            
        return self.state, reward, done, {}

    def render(self):
        for player in self.state["player_positions"]:
            print(f"Player piece at {player}")
        for enemy in self.state["enemy_positions"]:
            print(f"Enemy piece at {enemy}")

    def _is_checkmate(self, q, r):
        # Placeholder logic: implement based on game rules
        return (q, r) in [(0, 3), (1, -3)]  # Example checkmate points


# Instantiate Environment
scenario = world_data["regions"][0]["puzzleScenarios"][0]
env = DummyVecEnv([lambda: HexPuzzleEnv(scenario)])

# Create PPO Model
model = PPO("MultiInputPolicy", env, verbose=1)

# Training loop
print("Training PPO RL model...")
model.learn(total_timesteps=20000)
model.save("ppo_redwood_vale")