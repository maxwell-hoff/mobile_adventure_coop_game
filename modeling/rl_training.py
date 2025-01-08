import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml
import os
import random

# Load world and pieces YAML
with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)

class HexPuzzleEnv(gym.Env):
    def __init__(self, puzzle_scenario, max_steps=50, n_episodes=3):
        super(HexPuzzleEnv, self).__init__()
        self.puzzle_scenario = puzzle_scenario
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_steps = max_steps
        self.n_episodes = n_episodes  # how many episodes we want to run

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

        # Simple approach: discrete action picks which hex to move the first player piece
        self.action_space = gym.spaces.Discrete(self.num_positions)

        # Observations: positions for all player + enemy
        obs_size = 2 * (len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(obs_size,), dtype=np.float32
        )

        # We’ll store multiple episodes in self.all_episodes
        # each episode is a list of steps => step: { turn, move, reward, positions }
        self.all_episodes = []   # final shape: [episode_0, episode_1, ...]

        self.current_episode_data = []  # steps in the active episode
        self.episodes_run = 0
        self.reset()

    def reset(self):
        # If we just finished an episode, push its data to self.all_episodes
        if len(self.current_episode_data) > 0:
            self.all_episodes.append(self.current_episode_data)
            self.current_episode_data = []

            self.episodes_run += 1
            if self.episodes_run >= self.n_episodes:
                # We’ve run all episodes we intended, but Gym wants an obs anyway
                # Return something valid to not break training.
                return np.zeros(self.observation_space.shape, dtype=np.float32)

        self.steps_taken = 0

        # Restore initial scenario positions
        self.player_positions = np.array([[p["q"], p["r"]] 
                                          for p in self.player_pieces], dtype=np.float32)
        self.enemy_positions = np.array([[p["q"], p["r"]] 
                                         for p in self.enemy_pieces], dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        self.steps_taken += 1
        done = False
        reward = 0.0

        # Convert action index -> (q, r)
        q, r = self.all_hexes[action]

        # Player move
        if self._is_valid_move(q, r):
            self.player_positions[0] = np.array([q, r], dtype=np.float32)
            reward += 0.1
            if self._is_checkmate(q, r):
                reward += 10
                done = True
        else:
            reward -= 1

        # Enemy move (simple random for demonstration)
        self._enemy_random_move()

        if self.steps_taken >= self.max_steps or self._check_game_over():
            done = True

        # Log the *entire state* after both player & enemy move
        all_positions = {
            "player": self.player_positions.copy(),
            "enemy": self.enemy_positions.copy()
        }
        step_dict = {
            "turn": "player",  # your environment is single-agent, but you can store "player/enemy"
            "move": (q, r),
            "reward": reward,
            "positions": all_positions
        }
        self.current_episode_data.append(step_dict)

        obs = self._get_obs()
        return obs, reward, done, {}

    def _enemy_random_move(self):
        """
        Move the first enemy piece by a random valid step of up to 1 hex away,
        ignoring blocked hexes for brevity. 
        """
        if len(self.enemy_positions) == 0:
            return
        cur_q, cur_r = self.enemy_positions[0]

        # random delta in [-1, 0, 1] for q and r
        tries = 10
        while tries > 0:
            tries -= 1
            dq = random.randint(-1, 1)
            dr = random.randint(-1, 1)
            new_q = cur_q + dq
            new_r = cur_r + dr
            if self._is_valid_enemy_move(new_q, new_r):
                self.enemy_positions[0] = [new_q, new_r]
                break

    def _is_valid_enemy_move(self, q, r):
        # Must be within board
        if (q, r) not in self.all_hexes:
            return False
        # Could also check blocked or distance
        return True

    def _get_obs(self):
        flat_player = self.player_positions.flatten()
        flat_enemy = self.enemy_positions.flatten()
        return np.concatenate([flat_player, flat_enemy], axis=0).astype(np.float32)

    def _is_valid_move(self, q, r):
        cur_q, cur_r = self.player_positions[0]
        dist = self._hex_distance(cur_q, cur_r, q, r)
        if dist > 1:
            return False
        blocked_hexes = {(b["q"], b["r"]) for b in self.puzzle_scenario.get("blockedHexes", [])}
        if (q, r) in blocked_hexes:
            return False
        return True

    def _is_checkmate(self, q, r):
        # For demonstration
        check_positions = [(0,3), (1,-3)]
        return (q, r) in check_positions

    def _check_game_over(self):
        # If no enemies remain or no players remain, etc.
        return False

    @staticmethod
    def _hex_distance(q1, r1, q2, r2):
        return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) / 2


# ---------------------------------------------------
# Create and train
# ---------------------------------------------------
def main():
    scenario = world_data["regions"][0]["puzzleScenarios"][0]
    # We will run 3 episodes for demonstration, so we can see multiple "iterations"
    env = DummyVecEnv([lambda: HexPuzzleEnv(scenario, max_steps=15, n_episodes=3)])

    model = PPO("MlpPolicy", env, verbose=1)
    print("Training PPO RL model...")
    model.learn(total_timesteps=2000)  # short training

    # After training, retrieve the environment's episodes
    # Each env.envs[0].all_episodes is a list of episodes.
    all_episodes = env.envs[0].all_episodes

    # Save it to a .npy so we can visualize multiple episodes
    # each episode is a list of step dicts
    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Training complete. Actions log saved!")

if __name__ == "__main__":
    main()
