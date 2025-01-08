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
        self.actions_log = []  # Initialize the actions log
        self.is_player_turn = True  # Track whose turn it is

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
        self.actions_log.clear()  # Clear the actions log on reset
        self.is_player_turn = True  # Start with player's turn
        return self.state

    def step(self, action):
        q, r = action
        # Store the current turn information
        turn_info = {
            "turn": "player" if self.is_player_turn else "enemy",
            "move": (int(q), int(r)),  # Convert to regular integers for serialization
            "reward": 0  # Will be updated below
        }
        
        # Reward logic: positive for checkmate, negative for wrong moves
        reward = 0
        done = False

        # Example: check if action leads to checkmate
        if self._is_checkmate(q, r):
            reward = 10
            done = True
        else:
            reward = -1  # Penalty for non-optimal move

        # Update the reward in our turn info
        turn_info["reward"] = reward
        
        # Update only the first player piece position
        if len(self.state["player_positions"]) > 0:
            self.state["player_positions"][0] = np.array([q, r], dtype=np.int32)
        
        # Add the turn info to our actions log
        self.actions_log.append(turn_info)
        
        # Toggle the turn
        self.is_player_turn = not self.is_player_turn
        
        return self.state, reward, done, {}

    def _calculate_reward(self, side_str):
        # Example placeholder reward logic
        if side_str == "player" and self._is_checkmate():
            return 10
        elif side_str == "enemy" and self._is_checkmate():
            return -10
        return -1  # Negative reward for non-optimal moves

    def _is_checkmate(self, q, r):
        # Placeholder checkmate logic
        return any((q, r) in [(0, 3), (1, -3)] for q, r in self.state["player_positions"])
    
    def _check_game_over(self):
        # Placeholder game over logic
        return len(self.state["player_positions"]) == 0 or len(self.state["enemy_positions"]) == 0


scenario = world_data["regions"][0]["puzzleScenarios"][0]
env = DummyVecEnv([lambda: HexPuzzleEnv(scenario)])

# Create PPO Model
model = PPO("MultiInputPolicy", env, verbose=1)

# Training PPO model with modified logging
print("Training PPO RL model...")
model.learn(total_timesteps=20000)
model.save("ppo_redwood_vale")

# Save actions log for visualization
actions_log_file = "actions_log.npy"
np.save(actions_log_file, env.envs[0].actions_log)
