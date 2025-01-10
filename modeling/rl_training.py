import gymnasium as gym
import numpy as np
import random
import yaml
import os
import time

from copy import deepcopy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

# Force a fixed seed
np.random.seed(42)
random.seed(42)

with open(os.path.join("data", "world.yaml"), "r") as f:
    world_data = yaml.safe_load(f)
with open(os.path.join("data", "pieces.yaml"), "r") as f:
    pieces_data = yaml.safe_load(f)


class HexPuzzleEnv(gym.Env):
    """
    Single-agent environment controlling both 'player' & 'enemy' in a turn-based manner.

    * If the current side has no living pieces at the start of step(), we forcibly end the puzzle.
      => We never produce an all-False or dummy mask in that scenario.
    * We incorporate 'cast_speed' for spells like 'necrotizing_consecrate' by storing
      them in self.delayedAttacks, triggered after X turns.
    """

    def __init__(self, puzzle_scenario, max_turns=10, render_mode=None):
        super().__init__()
        self.original_scenario = deepcopy(puzzle_scenario)
        self.scenario = deepcopy(puzzle_scenario)
        self.grid_radius = puzzle_scenario["subGridRadius"]
        self.max_turns = max_turns

        # Initialize pieces
        self._init_pieces_from_scenario(self.scenario)

        # Build all hexes
        self.all_hexes = []
        for q in range(-self.grid_radius, self.grid_radius+1):
            for r in range(-self.grid_radius, self.grid_radius+1):
                if abs(q+r) <= self.grid_radius:
                    self.all_hexes.append((q, r))
        self.num_positions = len(self.all_hexes)

        # Each piece => num_positions moves + pass + necro
        self.actions_per_piece = self.num_positions + 2
        self.total_pieces = len(self.all_pieces)  # e.g. 8
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observations => (q,r) for each piece
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius, high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        # Logging
        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0

        # Delayed spells
        self.delayedAttacks = []

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"] == "enemy"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if len(self.current_episode) > 0:
            self.all_episodes.append(self.current_episode)
        self.current_episode = []

        self.scenario = deepcopy(self.original_scenario)
        self._init_pieces_from_scenario(self.scenario)

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False
        self.non_bloodwarden_kills = 0
        self.delayedAttacks.clear()

        print("\n=== RESET ===")
        for p in self.player_pieces:
            print(f"  Player {p['label']} at ({p['q']}, {p['r']})")
        for e in self.enemy_pieces:
            print(f"  Enemy {e['label']} at ({e['q']}, {e['r']})")
        print("================")

        # Step 0
        init_dict = {
            "turn_number": 0,
            "turn_side": None,
            "reward": 0.0,
            "positions": self._log_positions()
        }
        self.current_episode.append(init_dict)

        return self._get_obs(), {}

    def step(self, action):
        """
        1) If no living pieces on current side => forcibly end puzzle.
        2) Otherwise, interpret 'action' => check if necro or move or pass.
        3) End-of-turn => apply _apply_end_conditions + delayedAttacks checks.
        """
        # If forcibly ended
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        # If no living pieces => forcibly end puzzle immediately
        side_living = [pc for pc in self.all_pieces if pc["side"]==self.turn_side and not pc.get("dead",False)]
        if len(side_living)==0:
            # e.g. If it's player side's turn but we have 0 living players => end puzzle
            end_reward, term, _ = self._apply_end_conditions(0.0)
            # We force the puzzle to end
            return self._finish_step(end_reward, terminated=term, truncated=False)

        # Normal logic
        piece_index = action // self.actions_per_piece
        local_action = action % self.actions_per_piece
        valid_reward = 0.0

        if piece_index<0 or piece_index>=len(self.all_pieces):
            return self._finish_step(-1.0, terminated=False, truncated=False)

        piece = self.all_pieces[piece_index]
        if piece.get("dead",False) or piece["side"]!=self.turn_side:
            return self._finish_step(-1.0, terminated=False, truncated=False)

        if local_action<self.num_positions:
            # Move
            q, r = self.all_hexes[local_action]
            if self._valid_move(piece,q,r):
                piece["q"]=q
                piece["r"]=r
            else:
                valid_reward-=1.0
        elif local_action==self.num_positions:
            # pass
            valid_reward-=0.5
        else:
            # necro
            if self._can_necro(piece):
                self._schedule_necro(piece)
            else:
                valid_reward-=1.0

        reward, terminated, truncated = self._apply_end_conditions(valid_reward)
        return self._finish_step(reward,terminated,truncated)

    def _finish_step(self, reward, terminated, truncated):
        """
        Conclude side's step, log, possibly swap side, check delayedAttacks if we just ended enemy->player cycle
        """
        step_data = {
            "turn_number": self.turn_number,
            "turn_side": self.turn_side,
            "reward": reward,
            "positions": self._log_positions()
        }
        if terminated or truncated:
            step_data["non_bloodwarden_kills"]=self.non_bloodwarden_kills
            self.done_forced=True

        self.current_episode.append(step_data)

        if not(terminated or truncated):
            # switch side
            if self.turn_side=="player":
                self.turn_side="enemy"
            else:
                self.turn_side="player"
                self.turn_number+=1

            # check delayedAttacks
            self._check_delayed_attacks()

        return self._get_obs(), reward, terminated, truncated, {}

    def _check_delayed_attacks(self):
        to_remove=[]
        for i,att in enumerate(self.delayedAttacks):
            if att["trigger_turn"]==self.turn_number:
                # trigger
                caster_side=att["caster_side"]
                if caster_side=="enemy":
                    kills=sum(1 for p in self.player_pieces if not p.get("dead",False))
                    for p in self.player_pieces:
                        self._kill_piece(p)
                    extra_reward=2*kills
                    event_dict={
                        "turn_number": self.turn_number,
                        "turn_side":"enemy",
                        "reward": extra_reward,
                        "positions": self._log_positions(),
                        "desc":f"Delayed necro by enemy kills {kills} player piece(s)."
                    }
                    self.current_episode.append(event_dict)
                else:
                    kills=sum(1 for e in self.enemy_pieces if not e.get("dead",False))
                    for e in self.enemy_pieces:
                        self._kill_piece(e)
                    extra_reward=2*kills
                    event_dict={
                        "turn_number": self.turn_number,
                        "turn_side":"player",
                        "reward":extra_reward,
                        "positions":self._log_positions(),
                        "desc":f"Delayed necro by player kills {kills} enemy piece(s)."
                    }
                    self.current_episode.append(event_dict)
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.delayedAttacks.pop(idx)

    def _schedule_necro(self, piece):
        piece_class=pieces_data["classes"][piece["class"]]
        necro_data=piece_class["actions"]["necrotizing_consecrate"]
        cast_speed=necro_data.get("cast_speed",0)
        if cast_speed>0:
            trig_turn=self.turn_number+cast_speed
            log_event={
                "turn_number":self.turn_number,
                "turn_side":piece["side"],
                "reward":0.0,
                "positions":self._log_positions(),
                "desc":f"{piece['class']} ({piece['label']}) started necro, triggers on turn {trig_turn}"
            }
            self.current_episode.append(log_event)
            self.delayedAttacks.append({
                "caster_side": piece["side"],
                "caster_label": piece["label"],
                "trigger_turn": trig_turn,
                "action_name":"necrotizing_consecrate"
            })
        else:
            # immediate
            if piece["side"]=="enemy":
                kills=sum(1 for p in self.player_pieces if not p.get("dead",False))
                for p in self.player_pieces:
                    self._kill_piece(p)
                event_dict={
                    "turn_number": self.turn_number,
                    "turn_side":"enemy",
                    "reward":2*kills,
                    "positions":self._log_positions(),
                    "desc":f"Immediate necro by enemy kills {kills} players"
                }
                self.current_episode.append(event_dict)
            else:
                kills=sum(1 for e in self.enemy_pieces if not e.get("dead",False))
                for e in self.enemy_pieces:
                    self._kill_piece(e)
                event_dict={
                    "turn_number": self.turn_number,
                    "turn_side":"player",
                    "reward":2*kills,
                    "positions":self._log_positions(),
                    "desc":f"Immediate necro by player kills {kills} enemies"
                }
                self.current_episode.append(event_dict)

    def _apply_end_conditions(self, base_reward):
        """
        If a side is wiped => end. If turn_number>max_turns => truncated => -10.
        """
        p_alive=[p for p in self.player_pieces if not p.get("dead",False)]
        e_alive=[p for p in self.enemy_pieces if not p.get("dead",False)]
        reward=base_reward
        term=False
        trunc=False

        if len(p_alive)==0 and len(e_alive)==0:
            reward-=10
            term=True
        elif len(p_alive)==0:
            if self.turn_side=="enemy":
                reward+=20
            else:
                reward-=20
            term=True
        elif len(e_alive)==0:
            if self.turn_side=="player":
                reward+=20
            else:
                reward-=20
            term=True

        if not term:
            if self.turn_number>=self.max_turns:
                reward-=10
                trunc=True
        return reward, term, trunc

    def _valid_move(self, piece,q,r):
        piece_class=pieces_data["classes"][piece["class"]]
        move_def=piece_class["actions"].get("move",None)
        if not move_def:
            return False
        max_range=move_def["range"]
        dist=self._hex_distance(piece["q"],piece["r"],q,r)
        if dist>max_range:
            return False

        blocked={(h["q"],h["r"]) for h in self.scenario["blockedHexes"]}
        if (q,r) in blocked:
            return False

        for pc in self.all_pieces:
            if not pc.get("dead",False) and pc["q"]==q and pc["r"]==r:
                return False
        return True

    def _can_necro(self,piece):
        return piece["class"]=="BloodWarden"

    def _kill_piece(self,piece):
        if not piece.get("dead",False):
            if piece["class"]!="BloodWarden":
                self.non_bloodwarden_kills+=1
            piece["dead"]=True
            piece["q"]=9999
            piece["r"]=9999

    def _hex_distance(self,q1,r1,q2,r2):
        return (abs(q1-q2)+abs(r1-r2)+abs((q1+r1)-(q2+r2)))//2

    def _get_obs(self):
        coords=[]
        for p in self.player_pieces:
            coords.append(p["q"])
            coords.append(p["r"])
        for e in self.enemy_pieces:
            coords.append(e["q"])
            coords.append(e["r"])
        return np.array(coords,dtype=np.float32)

    def _log_positions(self):
        pa=np.array([[p["q"],p["r"]] for p in self.player_pieces],dtype=np.float32)
        ea=np.array([[p["q"],p["r"]] for p in self.enemy_pieces],dtype=np.float32)
        return {"player":pa,"enemy":ea}

    def _get_action_mask(self):
        """
        If the current side has living pieces => normal mask.
        If not => we won't produce a mask because step() forcibly ends. 
        So we never produce all-False => meltdown avoided.
        """
        if self.done_forced:
            # Return a 1-hot dummy so stable_baselines doesn't meltdown
            mask = np.zeros(self.total_actions,dtype=bool)
            mask[0]=True
            return mask

        side_living=[pc for pc in self.all_pieces if pc["side"]==self.turn_side and not pc.get("dead",False)]
        if len(side_living)==0:
            # In step() we forcibly end => so presumably this won't be used
            # But if SB3 calls it anyway => return a 1-hot
            print(f"DEBUG: side={self.turn_side} no living => 1-hot mask, puzzle ends in step()")
            mask=np.zeros(self.total_actions,dtype=bool)
            mask[0]=True
            return mask

        mask=np.zeros(self.total_actions,dtype=bool)
        for i,pc in enumerate(self.all_pieces):
            if pc.get("dead",False) or pc["side"]!=self.turn_side:
                continue
            base=i*self.actions_per_piece
            for idx,(q,r) in enumerate(self.all_hexes):
                if self._valid_move(pc,q,r):
                    mask[base+idx]=True
            # pass
            mask[base+self.num_positions]=True
            # necro
            if self._can_necro(pc):
                mask[base+self.num_positions+1]=True

        # If after that we have no True => just in case => 1-hot
        if not mask.any():
            print("DEBUG: no valid moves => 1-hot mask. We'll end puzzle in step() anyway.")
            mask[0]=True
        return mask

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

    vec_env = DummyVecEnv([make_env_fn(scenario_copy)])
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)

    print("Training until player wins or 20 minutes pass...")

    player_side_has_won=False
    iteration_count_before=0
    start_time=time.time()
    time_limit=20*60

    while True:
        model.learn(total_timesteps=1000)

        elapsed=time.time()-start_time
        if elapsed>=time_limit:
            print("Time limit reached => stop training.")
            break

        all_eps=vec_env.envs[0].all_episodes
        for i,ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
            if len(ep)==0: 
                continue
            final=ep[-1]
            if final["reward"]>=20 and final["turn_side"]=="player":
                print(f"Player side just won in iteration {i+1}!")
                player_side_has_won=True
                break
        iteration_count_before=len(all_eps)
        if player_side_has_won:
            break

    # Summaries
    all_episodes=vec_env.envs[0].all_episodes
    iteration_outcomes=[]
    for i,episode in enumerate(all_episodes):
        if len(episode)==0:
            iteration_outcomes.append(f"Iteration {i+1}: No steps?")
            continue
        final=episode[-1]
        rew=final["reward"]
        side=final["turn_side"]
        nbk=final.get("non_bloodwarden_kills",0)
        if rew>=20 and side=="player":
            outcome_str=f"Iteration {i+1}: PLAYER side WINS (nb_kills={nbk})"
        elif rew>=20 and side=="enemy":
            outcome_str=f"Iteration {i+1}: ENEMY side WINS (nb_kills={nbk})"
        elif rew<=-20:
            outcome_str=f"Iteration {i+1}: {side} side LOSES (nb_kills={nbk})"
        elif rew==-10:
            outcome_str=f"Iteration {i+1}: double knockout/time-limit penalty (nb_kills={nbk})"
        else:
            outcome_str=(f"Iteration {i+1}: final reward={rew}, side={side}, nb_kills={nbk}")
        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for line in iteration_outcomes:
        print(line)
    print("==========================\n")

    np.save("actions_log.npy", np.array(all_episodes,dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")

if __name__=="__main__":
    main()
