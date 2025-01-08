import gym
import numpy as np
import random
import yaml
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load world and piece definitions
with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)

with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)

class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment:
      - The agent controls the 'player' side.
      - The 'enemy' side is scripted: specifically, the BloodWarden 
        will cast 'necrotizing_consecrate' on its first turn, killing 
        all player pieces if they haven't already 'won'.

    Each episode:
      - Always starts with the puzzle in the same initial layout.
      - The player has up to 3 moves to solve the puzzle 
        (some "checkmate" or "win" condition).
      - If the puzzle isn't solved by the time the enemy acts 
        (which happens after the player's first move), 
        the BloodWarden kills everyone => puzzle lost => done.

    ACTION SPACE (simplified):
      - The agent picks which piece to act (among player pieces).
      - Then picks a target hex or "pass."

    We'll keep this environment short:
      - Maximum 3 player moves total. If not solved => done (lose).
      - On the enemy's first turn, they do necrotizing_consecrate => done (player loses) 
        unless the puzzle was already solved.

    REWARDS:
      - Big +10 if solve the puzzle (some checkmate condition).
      - -10 if the puzzle is lost (BloodWarden kills you or you run out of moves).
      - +0.1 for a valid move, -1 for an invalid move, etc.
    """

    def __init__(self, puzzle_scenario, max_player_moves=3):
        super().__init__()
        self.scenario = puzzle_scenario
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_player_moves = max_player_moves  # how many player moves we allow

        # Separate out pieces
        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "enemy"]

        # Identify the BloodWarden (if present)
        self.bloodwarden = None
        for epiece in self.enemy_pieces:
            if epiece["class"] == "BloodWarden":
                self.bloodwarden = epiece
                break

        # Build a list of all hex coords
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # We'll store an action space like:
        #   - piece_index in [0..(n_player_pieces-1)]
        #   - target_index in [0..num_positions] where "num_positions" means pass
        self.n_player_pieces = len(self.player_pieces)
        self.action_space = gym.spaces.Discrete(self.n_player_pieces * (self.num_positions + 1))

        # Observations: positions of all player + enemy pieces
        obs_size = 2*(len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(obs_size,), dtype=np.float32
        )

        # Logging
        self.all_episodes = []        # entire set of episodes
        self.current_episode_steps = []  # steps in the current episode
        self.reset()

    def reset(self):
        # If we just finished an episode, store it
        if len(self.current_episode_steps) > 0:
            self.all_episodes.append(self.current_episode_steps)
            self.current_episode_steps = []

        # Reset puzzle to the initial configuration each time
        # so that each iteration starts from the same puzzle layout
        for p in self.player_pieces:
            # optionally store original q/r in puzzle or do something like:
            p["q"] = p.get("q_init", p["q"])  
            p["r"] = p.get("r_init", p["r"])
        for e in self.enemy_pieces:
            e["q"] = e.get("q_init", e["q"])
            e["r"] = e.get("r_init", e["r"])

        # Track how many moves the player has used
        self.player_moves_used = 0
        # It's always the player's turn first
        self.is_player_turn = True
        # Whether the enemy has used necrotizing_consecrate
        self.enemy_has_cast = False

        return self._get_obs()

    def step(self, action):
        """
        Each call to step() => one piece on the active side uses one action 
        (which might be pass or move).
        Then we check if the puzzle is solved or if the enemy kills everyone.
        """
        done = False
        reward = 0.0

        piece_action_space = self.num_positions + 1
        piece_idx = action // piece_action_space
        target_idx = action % piece_action_space

        # We'll assume single-agent => controlling 'player' side
        # We only step if it's the player's turn. If it's the enemy's turn,
        # we do the forced "necrotizing_consecrate."
        # In your final approach, you might do multi-step logic to handle 
        # each piece's turn in sequence. But here's a simpler approach:
        if not self.is_player_turn:
            # It's enemy turn => always cast necrotizing_consecrate
            self._enemy_necrotizing_consecrate()
            # Log that
            step_dict = {
                "turn": "enemy",
                "piece_label": (self.bloodwarden["label"] if self.bloodwarden else "Unknown"),
                "action": "necrotizing_consecrate",
                "move": None,
                "reward": reward,
                "positions": self._log_positions()
            }
            self.current_episode_steps.append(step_dict)

            # If any player pieces remain after that, let's allow next step. 
            # But typically that kills them => done.
            if len(self.player_pieces) == 0:
                reward -= 10  # big penalty: player lost
                done = True
            else:
                # If for some reason some remain, we can let puzzle continue or end
                done = True  # typically you'd end, because the puzzle scenario is designed that it kills everyone
            obs = self._get_obs()
            return obs, reward, done, {}

        # If we are here => it's the player's turn
        if piece_idx < 0 or piece_idx >= len(self.player_pieces):
            # invalid piece index
            reward -= 2
            step_dict = {
                "turn": "player",
                "piece_label": "invalid_piece_index",
                "action": "invalid",
                "move": None,
                "reward": reward,
                "positions": self._log_positions()
            }
            self.current_episode_steps.append(step_dict)

        else:
            piece = self.player_pieces[piece_idx]
            if target_idx == self.num_positions:
                # pass
                step_dict = {
                    "turn": "player",
                    "piece_label": piece["label"],
                    "action": "pass",
                    "move": None,
                    "reward": reward,
                    "positions": self._log_positions()
                }
                self.current_episode_steps.append(step_dict)
            else:
                # Try to move to that hex
                q, r = self.all_hexes[target_idx]
                if self._valid_move(piece, q, r):
                    old_q, old_r = piece["q"], piece["r"]
                    piece["q"], piece["r"] = q, r

                    # small reward for valid move
                    reward += 0.1
                    # check if puzzle is solved => +10
                    if self._is_checkmate(piece, q, r):
                        reward += 10
                        done = True
                    step_dict = {
                        "turn": "player",
                        "piece_label": piece["label"],
                        "action": "move",
                        "move": (q, r),
                        "reward": reward,
                        "positions": self._log_positions()
                    }
                    self.current_episode_steps.append(step_dict)
                else:
                    # invalid move
                    reward -= 1
                    step_dict = {
                        "turn": "player",
                        "piece_label": piece["label"],
                        "action": "invalid_move",
                        "move": (q, r),
                        "reward": reward,
                        "positions": self._log_positions()
                    }
                    self.current_episode_steps.append(step_dict)

        # Now we've used 1 player move
        self.player_moves_used += 1

        # If the puzzle wasn't solved => check if we used up all 3 moves
        if not done and self.player_moves_used >= self.max_player_moves:
            # now it's the enemy turn => they cast necrotizing => end
            self.is_player_turn = False

        obs = self._get_obs()
        info = {}

        return obs, reward, done, info

    # ---------------------------------------------------------
    # HELPER METHODS
    # ---------------------------------------------------------

    def _enemy_necrotizing_consecrate(self):
        """
        Enemy's lethal AOE that kills all players instantly.
        We'll remove all player pieces from self.player_pieces.
        """
        if not self.bloodwarden:
            return
        # kills everything
        self.player_pieces.clear()  # so length is now 0
        self.enemy_has_cast = True

    def _valid_move(self, piece, q, r):
        """Check constraints from pieces.yaml for 'move' action."""
        pclass = pieces_data["classes"][piece["class"]]
        move_def = pclass["actions"].get("move", None)
        if not move_def:
            return False

        max_range = move_def["range"]
        dist = self._hex_distance(piece["q"], piece["r"], q, r)
        if dist > max_range:
            return False

        # check blocked
        blocked_hexes = {(bh["q"], bh["r"]) for bh in self.scenario["blockedHexes"]}
        if (q, r) in blocked_hexes:
            return False

        # check occupied
        all_positions = [(p["q"], p["r"]) for p in (self.player_pieces + self.enemy_pieces)]
        if (q, r) in all_positions:
            return False

        return True

    def _is_checkmate(self, piece, q, r):
        """
        Suppose 'checkmate' = we stepped onto (1, -3), or 
        we captured an enemy piece, or some puzzle logic, etc.
        Here is just a placeholder.
        """
        check_positions = [(0, 3), (1, -3)]
        return (q, r) in check_positions

    def _log_positions(self):
        """
        Return dict with current positions of player and enemy 
        so the visualization can update.
        """
        player_array = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        enemy_array = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": player_array, "enemy": enemy_array}

    def _get_obs(self):
        # Flatten player + enemy positions
        coords = []
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords, dtype=np.float32)

    @staticmethod
    def _hex_distance(q1, r1, q2, r2):
        return (abs(q1 - q2)
              + abs(r1 - r2)
              + abs((q1 + r1) - (q2 + r2))) / 2

    def close(self):
        # If we want to store final steps for the last episode
        if len(self.current_episode_steps) > 0:
            self.all_episodes.append(self.current_episode_steps)

def main():
    scenario = world_data["regions"][0]["puzzleScenarios"][0]

    # Single environment
    env = DummyVecEnv([lambda: HexPuzzleEnv(scenario, max_player_moves=3)])
    model = PPO("MlpPolicy", env, verbose=1)

    # We'll do a short training
    model.learn(total_timesteps=2000)

    # Save episodes
    episodes = env.envs[0].all_episodes
    np.save("actions_log.npy", np.array(episodes, dtype=object), allow_pickle=True)
    print("Done training. Saved actions log for visualization!")

if __name__ == "__main__":
    main()
