import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import random

try:
    import gym
    from gym import spaces as gym_spaces
except ImportError:
    gym = None
    gym_spaces = None

from src.minecraft.mc_perception import MCWorldState


@dataclass
class MCObservation:
    state_vector: np.ndarray
    reward: float = 0.0
    done: bool = False
    info: Dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class MCEnvironment:
    def __init__(self, state_dim: int = 141, history_len: int = 4):
        self.state_dim = state_dim
        self.history_len = history_len
        self.history = deque(maxlen=history_len)
        self.current_goal = 'NONE'
        self.step_count = 0
        self.max_steps = 1000
        self.last_health = 20.0
        self.last_xp = 0
        self.last_inventory_count = 0
        self.reward_history = deque(maxlen=100)
        self.milestones = set()

        self._action_space_shape = (40,)
        self._observation_space_shape = (state_dim * history_len + 16,)

    def reset(self, initial_state: Optional[MCWorldState] = None) -> np.ndarray:
        self.step_count = 0
        self.milestones = set()
        self.reward_history.clear()

        if initial_state is not None:
            self.last_health = initial_state.health
            self.last_xp = initial_state.experience
            self.last_inventory_count = len(initial_state.inventory)
        else:
            self.last_health = 20.0
            self.last_xp = 0
            self.last_inventory_count = 0

        self.history.clear()
        if initial_state is not None:
            self.history.append(initial_state.to_state_vector())

        return self._get_observation()

    def step(self, action: np.ndarray, world_state: MCWorldState) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        state_vector = world_state.to_state_vector()
        self.history.append(state_vector)

        reward = self._calculate_reward(world_state)
        self.reward_history.append(reward)

        done = world_state.health <= 0 or self.step_count >= self.max_steps

        if world_state.health <= 0:
            reward -= 20.0

        info = {
            'health': world_state.health,
            'hunger': world_state.hunger,
            'inventory_size': len(world_state.inventory),
            'nearby_hostiles': world_state.nearby_hostiles,
            'milestones': list(self.milestones),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'current_goal': self.current_goal,
        }

        obs = self._get_observation()
        return obs, reward, done, info

    def _calculate_reward(self, state: MCWorldState) -> float:
        reward = 0.0

        health_change = state.health - self.last_health
        if health_change < 0 and state.nearby_hostiles > 0:
            reward += health_change * 0.1
        elif health_change < 0:
            reward += health_change * 2.0
        self.last_health = state.health

        xp_change = state.experience - self.last_xp
        if xp_change > 0:
            reward += xp_change * 5.0
            self.last_xp = state.experience

        inv_count = len(state.inventory)
        if inv_count > self.last_inventory_count:
            reward += (inv_count - self.last_inventory_count) * 2.0
            self.last_inventory_count = inv_count

        if state.hunger < 5:
            reward -= 1.0
        elif state.hunger > 15:
            reward += 0.5

        if state.nearby_hostiles > 0 and state.health > 15:
            reward += 1.0
        elif state.nearby_hostiles > 3 and state.health < 10:
            reward -= 2.0

        if state.armor_protection > 0:
            reward += 0.5 * state.armor_protection / 20.0

        milestone_reward = self._check_milestones(state)
        reward += milestone_reward

        reward += 0.01

        return reward

    def _check_milestones(self, state: MCWorldState) -> float:
        reward = 0.0
        milestones = {
            'has_wood': self._has_item(state, ['log', 'planks'], 1),
            'has_stone_tools': self._has_item(state, ['stone_pickaxe', 'stone_sword', 'stone_axe'], 1),
            'has_iron': self._has_item(state, ['iron_ingot'], 1),
            'has_furnace': self._has_item(state, ['furnace'], 1) or
                           self._has_item(state, ['crafting_table', 'planks'], 1),
            'level_5': state.experience >= 5,
            'has_diamond': self._has_item(state, ['diamond'], 1),
            'in_nether': state.dimension == 'minecraft:the_nether',
            'near_fortress': 'fortress' in state.biome.lower() if state.biome else False,
            'has_blaze_rod': self._has_item(state, ['blaze_rod'], 1),
            'near_end_portal': state.block_looking_at and 'end_portal' in state.block_looking_at,
            'in_end': state.dimension == 'minecraft:the_end',
            'dragon_defeated': state.current_goal == 'DONE',
        }

        milestone_values = {
            'has_wood': 5.0,
            'has_stone_tools': 10.0,
            'has_iron': 15.0,
            'has_furnace': 5.0,
            'level_5': 8.0,
            'has_diamond': 20.0,
            'in_nether': 30.0,
            'near_fortress': 10.0,
            'has_blaze_rod': 15.0,
            'near_end_portal': 25.0,
            'in_end': 50.0,
            'dragon_defeated': 100.0,
        }

        for name, achieved in milestones.items():
            if achieved and name not in self.milestones:
                self.milestones.add(name)
                reward += milestone_values.get(name, 5.0)
                from src.minecraft.mc_goals import MCGoalPlanner
                goal_map = {
                    'has_wood': 'CRAFT_TOOLS',
                    'has_stone_tools': 'MINE_IRON',
                    'has_iron': 'MINE_DIAMOND',
                    'has_diamond': 'FIND_LAVA',
                    'in_nether': 'FIND_FORTRESS',
                    'has_blaze_rod': 'FIND_END_PORTAL',
                    'in_end': 'KILL_ENDER_DRAGON',
                }
                if name in goal_map:
                    self.current_goal = goal_map[name]

        return reward

    def _has_item(self, state: MCWorldState, item_names: List[str], count: int = 1) -> bool:
        all_items = state.hotbar + state.inventory
        for entry in all_items:
            item = entry.get('item', '')
            cnt = entry.get('count', 0)
            if cnt >= count:
                for name in item_names:
                    if name in item.lower():
                        return True
        return False

    def _get_observation(self) -> np.ndarray:
        obs = []
        for state_vec in list(self.history):
            obs.extend(state_vec)

        while len(obs) < self.state_dim * self.history_len:
            obs.extend([0.0] * self.state_dim)

        goal_encoding = self._encode_goal()
        obs.extend(goal_encoding)

        return np.array(obs, dtype=np.float32)

    def _encode_goal(self) -> List[float]:
        goals = [
            'GATHER_WOOD', 'CRAFT_TOOLS', 'MINE_STONE', 'FIND_FOOD',
            'MINE_IRON', 'CRAFT_ARMOR', 'MINE_DIAMOND', 'FIND_LAVA',
            'BUILD_PORTAL', 'ENTER_NETHER', 'FIND_FORTRESS', 'KILL_BLAZE',
            'FIND_END_PORTAL', 'KILL_ENDER_DRAGON', 'EXPLORE', 'NONE'
        ]
        encoding = [0.0] * len(goals)
        if self.current_goal in goals:
            encoding[goals.index(self.current_goal)] = 1.0
        return encoding

    def set_goal(self, goal: str):
        self.current_goal = goal
