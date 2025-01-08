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
    Example environment with alternating turns:
      1) Player side moves each of its pieces once.
      2) Enemy side moves each of its pieces once.
    That constitutes one full round.

    We'll store both sides' moves in the log. We assume each piece 
    has exactly 1 action per round.

    Because your map.js has more thorough puzzle logic, here we just
    show how to enforce constraints from pieces.yaml (range, LOS, etc.).
    """
    def __init__(self, puzzle_scenario, max_rounds=5):
        super().__init__()
        self.scenario = puzzle_scenario
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_rounds = max_rounds  # how many full "rounds" of player+enemy
        self.rounds_done = 0

        # Separate out pieces
        self.player_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in puzzle_scenario["pieces"] if p["side"] == "enemy"]

        # Build a list of all hex coordinates in the puzzle
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius + 1):
            for r in range(-self.grid_radius, self.grid_radius + 1):
                if abs(q + r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # For a single-step environment approach, let's say the agent picks
        # one "piece index" + one "action" + one "target hex" all in one.
        # This is just a simplistic approach; you can do a better multi-discrete or dict space.
        # But to keep it short, let's define:
        #   - piece_index in [0..Nplayer-1]
        #   - hex_index in [0..(board_size)-1]
        #   => total possible actions = (Nplayer * board_size)
        # We'll assume "no-op" or "pass" is also included, or we can store that in a separate discrete dimension.
        
        self.n_player_pieces = len(self.player_pieces)
        # For now, define a single Discrete space that = n_player_pieces * (num_positions + 1 for pass).
        # If the agent picks an action that is "invalid", we give negative reward, for example.
        # This is simplified; typically you'd do a multi-discrete or a custom approach.
        self.action_space = gym.spaces.Discrete(self.n_player_pieces * (self.num_positions + 1))

        # Observations: positions of all player + enemy pieces
        # shape = 2*(Nplayer + Nenemy)
        obs_size = 2*(len(self.player_pieces) + len(self.enemy_pieces))
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(obs_size,), dtype=np.float32
        )

        # Logging
        self.all_episodes = []
        self.current_episode_steps = []
        self.reset()

    def reset(self):
        # If we just finished an episode, store its steps
        if len(self.current_episode_steps) > 0:
            self.all_episodes.append(self.current_episode_steps)
            self.current_episode_steps = []

        self.rounds_done = 0

        # Reset piece positions to scenario initial
        for p in self.player_pieces:
            p["q"] = p.get("q_init", p["q"])  # or use p["q"] directly
            p["r"] = p.get("r_init", p["r"])
        for p in self.enemy_pieces:
            p["q"] = p.get("q_init", p["q"])
            p["r"] = p.get("r_init", p["r"])

        self.steps_in_current_round = 0  # how many piece-moves done so far
        self.is_player_turn = True
        return self._get_obs()

    def step(self, action):
        """
        The agent picks an action representing which piece to move + where to move it (or pass).
        Then the environment will also move the enemy's piece(s). We only do 1 piece per environment step 
        or multiple pieces? 
        Because you said each piece on that side acts once, we might do:
         - step() => picks next piece on the active side
         - once all pieces on that side have acted, we switch sides
         - once both sides have acted => 1 round is complete
        """
        done = False
        reward = 0.0

        # 1) Decode the action
        #    passIndex = num_positions => means "pass"
        piece_action_space = self.num_positions + 1  # for each of the n_player_pieces
        piece_idx = action // piece_action_space
        target_idx = action % piece_action_space

        # piece to act
        if self.is_player_turn:
            piece_list = self.player_pieces
            side_str = "player"
        else:
            piece_list = self.enemy_pieces
            side_str = "enemy"

        if piece_idx < 0 or piece_idx >= len(piece_list):
            # invalid piece index
            reward -= 2
            done = False
        else:
            piece = piece_list[piece_idx]
            # parse target
            if target_idx == self.num_positions:
                # pass
                reward += 0  # or small negative or zero
                # log pass
                step_dict = {
                    "turn": side_str,
                    "piece_label": piece["label"],
                    "action": "pass",
                    "move": None,
                    "reward": reward,
                    "positions": self._log_positions()
                }
                self.current_episode_steps.append(step_dict)
            else:
                # The agent tries to move piece to self.all_hexes[target_idx]
                q, r = self.all_hexes[target_idx]
                
                # Check constraints from pieces.yaml for piece's "move" action:
                # We'll assume each piece can only do the "move" action for demonstration.
                # If you want to incorporate "attack" or "aoe", you'd parse that from the action as well.
                
                if self._valid_move(piece, q, r):
                    # do move
                    old_q, old_r = piece["q"], piece["r"]
                    piece["q"], piece["r"] = q, r
                    reward += 0.1
                    # check if checkmate
                    if self._is_checkmate(piece, q, r):
                        reward += 10
                        done = True
                    # log
                    step_dict = {
                        "turn": side_str,
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
                        "turn": side_str,
                        "piece_label": piece["label"],
                        "action": "invalid_move",
                        "move": (q, r),
                        "reward": reward,
                        "positions": self._log_positions()
                    }
                    self.current_episode_steps.append(step_dict)

        self.steps_in_current_round += 1

        # if all pieces in this side have acted, switch to the other side
        # note: you have as many steps in a side's turn as pieces on that side
        # If each side has the same # of pieces => total steps in a round is sum(...) etc.
        # We do a simpler approach here:
        n_pieces_side = len(piece_list)
        if self.steps_in_current_round >= n_pieces_side:
            # switch
            self.steps_in_current_round = 0
            self.is_player_turn = not self.is_player_turn
            # if we just switched from enemy->player => that means 1 full round is done
            if self.is_player_turn:  
                self.rounds_done += 1
                if self.rounds_done >= self.max_rounds:
                    done = True

        # done condition: max rounds or no pieces left, etc.
        if self._check_end_conditions():
            done = True

        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    # --------------- Helper functions --------------------
    def _check_end_conditions(self):
        # For instance, if all enemy pieces are gone or all player pieces are gone
        # or we reached round limit, etc.
        if len(self.player_pieces) == 0 or len(self.enemy_pieces) == 0:
            return True
        return False

    def _valid_move(self, piece, q, r):
        """
        Check constraints from pieces.yaml and map.js:
         - range <= pieceClass.actions["move"].range
         - not blocked
         - not occupied
         etc.
        """
        # find pieceClass
        pclass = pieces_data["classes"][piece["class"]]
        move_def = pclass["actions"].get("move", None)
        if not move_def:
            # no "move" action => invalid
            return False

        # check range
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

    def _log_positions(self):
        """
        Return a dict {player: np.array(...), enemy: np.array(...)} 
        so the visualization can see the final positions 
        after the action.
        """
        player_pos = np.array([[p["q"], p["r"]] for p in self.player_pieces], dtype=np.float32)
        enemy_pos = np.array([[p["q"], p["r"]] for p in self.enemy_pieces], dtype=np.float32)
        return {"player": player_pos, "enemy": enemy_pos}

    def _is_checkmate(self, piece, q, r):
        """
        Example: if the piece steps on (0,3) or (1,-3) 
        or if an enemy piece is captured, etc.
        """
        check_positions = [(0,3), (1,-3)]
        return (q, r) in check_positions

    def _get_obs(self):
        # Flatten positions
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
        # if we want to store final episode steps
        if len(self.current_episode_steps) > 0:
            self.all_episodes.append(self.current_episode_steps)

# -----------------------------------------
#  Example usage: train + save logs
# -----------------------------------------
def main():
    # Load puzzle scenario from YAML
    scenario = world_data["regions"][0]["puzzleScenarios"][0]

    env = DummyVecEnv([lambda: HexPuzzleEnv(scenario, max_rounds=5)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)

    # after training, gather episodes
    episodes = env.envs[0].all_episodes
    np.save("actions_log.npy", np.array(episodes, dtype=object), allow_pickle=True)
    print("Done training, saved logs with multi-piece turn structure.")

if __name__ == "__main__":
    main()
