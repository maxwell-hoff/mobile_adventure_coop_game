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
    Single-agent environment controlling both 'player' & 'enemy'.

    Extra constraints added:
      1) If a side's Priest is killed, the opposing side instantly wins (+20).
      2) If there's an enemy in range of one of your attacks, but you do not pick an attack action,
         you get a small negative reward (e.g. -0.2).

    * If the current side has no living pieces at the start of step(), puzzle ends immediately.
    * We keep your delayedAttacks logic for spells with cast_speed (like necrotizing_consecrate).
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

        # actions_per_piece = N moves + pass + necro => total_actions = num_pieces * actions_per_piece
        self.actions_per_piece = self.num_positions + 2
        self.total_pieces = len(self.all_pieces)
        self.total_actions = self.total_pieces * self.actions_per_piece
        self.action_space = gym.spaces.Discrete(self.total_actions)

        # Observations => 2 coords per piece
        self.obs_size = 2 * self.total_pieces
        self.observation_space = gym.spaces.Box(
            low=-self.grid_radius,
            high=self.grid_radius,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        self.turn_number = 1
        self.turn_side = "player"
        self.done_forced = False

        self.all_episodes = []
        self.current_episode = []
        self.non_bloodwarden_kills = 0

        # For delayed spells
        self.delayedAttacks = []

    @property
    def all_pieces(self):
        return self.player_pieces + self.enemy_pieces

    def _init_pieces_from_scenario(self, scenario_dict):
        self.player_pieces = [p for p in scenario_dict["pieces"] if p["side"]=="player"]
        self.enemy_pieces = [p for p in scenario_dict["pieces"] if p["side"]=="enemy"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if len(self.current_episode)>0:
            self.all_episodes.append(self.current_episode)
        self.current_episode=[]

        self.scenario = deepcopy(self.original_scenario)
        self._init_pieces_from_scenario(self.scenario)
        self.turn_number=1
        self.turn_side="player"
        self.done_forced=False
        self.non_bloodwarden_kills=0
        self.delayedAttacks.clear()

        print("\n=== RESET ===")
        for p in self.player_pieces:
            print(f"  Player {p['label']} at ({p['q']}, {p['r']})")
        for e in self.enemy_pieces:
            print(f"  Enemy {e['label']} at ({e['q']}, {e['r']})")
        print("================")

        init_dict={
            "turn_number":0,
            "turn_side":None,
            "reward":0.0,
            "positions":self._log_positions()
        }
        self.current_episode.append(init_dict)

        return self._get_obs(), {}

    def step(self, action):
        # forcibly ended?
        if self.done_forced:
            return self._get_obs(), 0.0, True, False, {}

        # If side has no living => forcibly end puzzle
        side_living=[pc for pc in self.all_pieces if pc["side"]==self.turn_side and not pc.get("dead",False)]
        if len(side_living)==0:
            end_reward, term, _ = self._apply_end_conditions(0.0)
            return self._finish_step(end_reward,terminated=term,truncated=False)

        piece_index = action//self.actions_per_piece
        local_action = action%self.actions_per_piece
        valid_reward=0.0

        # Basic checks
        if piece_index<0 or piece_index>=len(self.all_pieces):
            return self._finish_step(-1.0,terminated=False,truncated=False)

        piece=self.all_pieces[piece_index]
        if piece.get("dead",False) or piece["side"]!=self.turn_side:
            return self._finish_step(-1.0,terminated=False,truncated=False)

        # We'll do a small function that checks "did we skip an attack even though an enemy is in range?"
        # We'll call it after we see what action we took.
        could_have_attacked = self._could_have_attacked(piece)

        # interpret local_action
        if local_action<self.num_positions:
            # move
            q,r=self.all_hexes[local_action]
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

        # If the piece "could have attacked" but didn't do an attacking action => small negative
        # We'll define "attacking action" as the local_action that is not "move"/"pass"/"necro" in your code.
        # But you only have "move", "pass", or "necrotizing_consecrate" currently. 
        # So let's say: if the piece *could have used necro or some single_target_attack if that existed* but didn't => -0.2
        # For now, let's interpret "necro" as an AOE "attack" => if they "could_have_attacked" but didn't pick "necro", -0.2
        # In the future, if you add more attacks to the piece, you'd adapt the code in _could_have_attacked() to check them.
        if could_have_attacked:
            # local_action >= self.num_positions+1 => necro => that's an attack => no penalty
            # else => penalty
            if local_action < (self.num_positions+1):
                # didn't cast necro => penalty
                valid_reward -= 0.2

        # Now apply normal end checks
        reward,term,trunc = self._apply_end_conditions(valid_reward)
        return self._finish_step(reward,term,trunc)

    def _finish_step(self, reward, terminated, truncated):
        step_data={
            "turn_number":self.turn_number,
            "turn_side":self.turn_side,
            "reward":reward,
            "positions":self._log_positions()
        }
        if terminated or truncated:
            step_data["non_bloodwarden_kills"]=self.non_bloodwarden_kills
            self.done_forced=True

        self.current_episode.append(step_data)

        if not(terminated or truncated):
            if self.turn_side=="player":
                self.turn_side="enemy"
            else:
                self.turn_side="player"
                self.turn_number+=1
            self._check_delayed_attacks()

        obs=self._get_obs()
        return obs,reward,terminated,truncated,{}

    def _check_delayed_attacks(self):
        to_remove=[]
        for i,att in enumerate(self.delayedAttacks):
            if att["trigger_turn"]==self.turn_number:
                caster_side=att["caster_side"]
                # if you want to check if caster is still alive => do so
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
                        "desc": f"Delayed necro by enemy kills {kills} player piece(s)."
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
                        "reward": extra_reward,
                        "positions": self._log_positions(),
                        "desc": f"Delayed necro by player kills {kills} enemy piece(s)."
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
                "desc":f"{piece['class']} ({piece['label']}) started necro => triggers on turn {trig_turn}"
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
        1) If the enemy Priest is dead => you immediately win (+20).
           If your Priest is dead => you lose (-20).
           (We also keep the old logic: if entire side is wiped => end.)
        2) If turn_number>max_turns => truncated => -10
        """
        reward=base_reward
        term=False
        trunc=False

        p_alive=[p for p in self.player_pieces if not p.get("dead",False)]
        e_alive=[e for e in self.enemy_pieces if not e.get("dead",False)]

        # Check for priests
        player_priest_alive = any((p for p in p_alive if p["class"]=="Priest"))
        enemy_priest_alive = any((e for e in e_alive if e["class"]=="Priest"))

        # If the *enemy's* priest is dead => we (the current side) immediately win => +20
        # If *our* priest is dead => we lose => -20
        # We'll define "our side" as self.turn_side => "enemy side" is the opposite
        # But we also have the old logic: if entire side is wiped => +20 or -20
        # We'll handle them in a certain order to avoid conflict:

        # 1) Check if both sides are out of pieces => double knockout
        if len(p_alive)==0 and len(e_alive)==0:
            reward-=10
            term=True
        else:
            # 2) Check if one side has no living pieces => old logic
            if len(p_alive)==0:
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

        # 3) If not ended, check priests specifically
        #    e.g. if the enemy priest is dead => we get +20 => end
        #    if our priest is dead => -20 => end
        if not term:
            if self.turn_side=="player":
                # if enemy's priest is dead => +20
                if not enemy_priest_alive:
                    reward+=20
                    term=True
                # if player's priest is dead => -20
                elif not player_priest_alive:
                    reward-=20
                    term=True
            else:
                # turn_side="enemy"
                # if player's priest is dead => +20
                if not player_priest_alive:
                    reward+=20
                    term=True
                # if enemy's priest is dead => -20
                elif not enemy_priest_alive:
                    reward-=20
                    term=True

        # 4) if still not ended => check turn limit
        if not term:
            if self.turn_number>=self.max_turns:
                reward-=10
                trunc=True

        return reward,term,trunc

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

    def _could_have_attacked(self,piece):
        """
        Return True if 'piece' has at least one attack action with an enemy in range.
        For now, we only check if piece can do necro (if it's BloodWarden).
        If you add more 'single_target_attack' or 'multi_target_attack', you'd expand logic here.

        We'll interpret "in range" as range <= the action's 'range' and not blocked.
        For necro, range=100 => basically entire map. If there's at least 1 enemy alive => could cast => True
        """
        piece_class=pieces_data["classes"][piece["class"]]
        # We'll scan all possible "attack" or "xxx_attack" or 'necrotizing_consecrate'.
        # If there's an enemy in range => return True
        for actionName, actionData in piece_class["actions"].items():
            # We'll consider it an "attack" if action_name is not "move" or "pass"
            # But you only have "move" or "necrotizing_consecrate" in your example
            if actionName=="move":
                continue
            # else => assume it's an attack or necro
            rangeDist=actionData.get("range",0)
            if rangeDist>0:
                # Check if there's an enemy in that range
                # If it's necro => entire map => as soon as there's any enemy, piece *could* do that
                # We'll do a simpler approach: if there's at least 1 living enemy => True
                # Because necro in your data => range=100 => entire map
                # If you had single_target => we'd check if an enemy is within 'rangeDist' hex distance
                # For brevity, let's do: if there's any living enemy => True, done
                if piece["side"]=="player":
                    # see if there's an enemy piece not dead
                    livingEnemies=[e for e in self.enemy_pieces if not e.get("dead",False)]
                    if len(livingEnemies)>0:
                        return True
                else:
                    # side=enemy => see if there's a living player
                    livingPlayers=[p for p in self.player_pieces if not p.get("dead",False)]
                    if len(livingPlayers)>0:
                        return True
        return False

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
        If side has no living => step() forcibly ends => we won't produce an all-False => meltdown avoided.
        If forced done => 1-hot. If all valid => normal.
        """
        if self.done_forced:
            dummy=np.zeros(self.total_actions,dtype=bool)
            dummy[0]=True
            return dummy

        side_living=[pc for pc in self.all_pieces if pc["side"]==self.turn_side and not pc.get("dead",False)]
        if len(side_living)==0:
            print(f"DEBUG: side={self.turn_side} no living => puzzle ends in step() => 1-hot mask.")
            dummy=np.zeros(self.total_actions,dtype=bool)
            dummy[0]=True
            return dummy

        mask=np.zeros(self.total_actions,dtype=bool)
        for i,pc in enumerate(self.all_pieces):
            if pc.get("dead",False) or pc["side"]!=self.turn_side:
                continue
            base=i*self.actions_per_piece
            # moves
            for idx,(q,r) in enumerate(self.all_hexes):
                if self._valid_move(pc,q,r):
                    mask[base+idx]=True
            # pass
            mask[base+self.num_positions]=True
            # necro
            if self._can_necro(pc):
                mask[base+self.num_positions+1]=True

        if not mask.any():
            print("DEBUG: no valid => 1-hot => puzzle ends in step()")
            mask[0]=True
        return mask

    def action_masks(self):
        return self._get_action_mask()

def make_env_fn(scenario_dict):
    def _init():
        env=HexPuzzleEnv(puzzle_scenario=scenario_dict, max_turns=10)
        env=ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

def main():
    scenario=world_data["regions"][0]["puzzleScenarios"][0]
    scenario_copy=deepcopy(scenario)

    vec_env=DummyVecEnv([make_env_fn(scenario_copy)])
    model=MaskablePPO("MlpPolicy",vec_env,verbose=1)

    print("Training until player wins or 20 minutes pass...")

    player_side_has_won=False
    iteration_count_before=0
    start_time=time.time()
    time_limit=20*60

    while True:
        model.learn(total_timesteps=1000)

        elapsed=time.time()-start_time
        if elapsed>=time_limit:
            print("Time limit => stop training.")
            break

        all_eps=vec_env.envs[0].all_episodes
        for i,ep in enumerate(all_eps[iteration_count_before:], start=iteration_count_before):
            if len(ep)==0:
                continue
            final=ep[-1]
            if final["reward"]>=20 and final["turn_side"]=="player":
                print(f"Player side just won iteration {i+1}!")
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
            iteration_outcomes.append(f"Iteration {i+1}: No steps taken?")
            continue
        final=episode[-1]
        rew=final["reward"]
        side=final["turn_side"]
        nbk=final.get("non_bloodwarden_kills",0)
        if rew>=20 and side=="player":
            outcome_str=f"Iteration {i+1}: PLAYER side WINS! (nbw_kills={nbk})"
        elif rew>=20 and side=="enemy":
            outcome_str=f"Iteration {i+1}: ENEMY side WINS! (nbw_kills={nbk})"
        elif rew<=-20:
            outcome_str=f"Iteration {i+1}: {side} side LOSES! (nbw_kills={nbk})"
        elif rew==-10:
            outcome_str=f"Iteration {i+1}: double knockout/time-limit penalty (nbw_kills={nbk})"
        else:
            outcome_str=(f"Iteration {i+1}: final reward={rew}, side={side}, nbw_kills={nbk}")
        iteration_outcomes.append(outcome_str)

    print("\n=== Iteration Outcomes ===")
    for line in iteration_outcomes:
        print(line)
    print("==========================\n")

    np.save("actions_log.npy", np.array(all_episodes, dtype=object), allow_pickle=True)
    print("Saved actions_log.npy with scenario.")

if __name__=="__main__":
    main()
