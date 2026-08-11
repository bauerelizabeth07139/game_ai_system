import numpy as np
import torch
from typing import Dict, Optional, Tuple
import logging
from collections import deque

from src.minecraft.mc_perception import MCWorldState
from src.minecraft.mc_environment import MCEnvironment
from src.minecraft.mc_policy import MCPolicy
from src.minecraft.mc_goals import MCGoalPlanner, MCGoal

logger = logging.getLogger(__name__)


class TrainingMinecraftEnv:
    def __init__(self, headless: bool = True, max_steps: int = 1000):
        self.environment = MCEnvironment()
        self.goal_planner = MCGoalPlanner()
        self.max_steps = max_steps
        self.headless = headless
        self.policy = MCPolicy()
        self.current_state: Optional[MCWorldState] = None
        self.total_steps = 0

    def reset(self) -> np.ndarray:
        state = self._create_initial_state()
        self.current_state = state
        self.goal_planner.current_goal = MCGoal.GATHER_WOOD
        self.goal_planner.completed_goals = []
        self.environment.set_goal('GATHER_WOOD')

        obs = self.environment.reset(state)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self._simulate_step(action)
        self.current_state = next_state

        new_goal = self.goal_planner.update_goal(next_state)

        obs, reward, done, info = self.environment.step(action, next_state)
        return obs, reward, done, info

    def _create_initial_state(self) -> MCWorldState:
        state = MCWorldState()
        state.health = 20.0
        state.max_health = 20.0
        state.hunger = 20.0
        state.pos_x = 0.0
        state.pos_y = 64.0
        state.pos_z = 0.0
        state.dimension = "minecraft:overworld"
        state.time_of_day = 0
        state.nearby_hostiles = 0

        state.hotbar = []
        state.inventory = []
        return state

    def _simulate_step(self, action: np.ndarray) -> MCWorldState:
        state = self.current_state
        if state is None:
            return self._create_initial_state()

        new_state = MCWorldState()
        new_state.__dict__.update(state.__dict__)

        move_x = (float(action[2]) - 0.5) * 2.0
        move_y = (float(action[3]) - 0.5) * 2.0
        new_state.pos_x += move_x * 0.5
        new_state.pos_z += move_y * 0.5

        if self.goal_planner.current_goal in [MCGoal.GATHER_WOOD, MCGoal.MINE_STONE,
                                               MCGoal.MINE_IRON, MCGoal.MINE_DIAMOND]:
            if float(action[4]) > 0.5:
                self._add_resource(new_state)
                new_state.hunger = max(0.0, new_state.hunger - 0.1)

        new_state.hunger = max(0.0, new_state.hunger - 0.05)

        if new_state.hunger < 2:
            new_state.health = max(0.0, new_state.health - 0.2)

        if new_state.nearby_hostiles > 0:
            if float(action[4]) > 0.5:
                new_state.nearby_hostiles = max(0, new_state.nearby_hostiles - 1)
            else:
                new_state.health = max(0.0, new_state.health - 1.0)

        if float(action[5]) > 0.5 and self._has_food(new_state):
            new_state.hunger = min(20.0, new_state.hunger + 5.0)

        new_state.time_of_day = (new_state.time_of_day + 50) % 24000
        new_state.is_night = 13000 <= new_state.time_of_day <= 23000

        self.total_steps += 1
        return new_state

    def _add_resource(self, state: MCWorldState):
        goal = self.goal_planner.current_goal
        resource_map = {
            MCGoal.GATHER_WOOD: ('minecraft:oak_log', 1),
            MCGoal.MINE_STONE: ('minecraft:cobblestone', 1),
            MCGoal.MINE_IRON: ('minecraft:raw_iron', 1),
            MCGoal.MINE_DIAMOND: ('minecraft:diamond', 1),
        }

        if goal in resource_map:
            item_name, count = resource_map[goal]
            found = False
            for entry in state.inventory:
                if entry.get('item') == item_name:
                    entry['count'] = entry.get('count', 0) + count
                    found = True
                    break
            if not found:
                state.inventory.append({'slot': len(state.inventory), 'item': item_name, 'count': count})

            if goal == MCGoal.GATHER_WOOD and self._count_item(state, 'minecraft:oak_log') >= 4:
                state.inventory.append({'slot': len(state.inventory), 'item': 'minecraft:oak_planks', 'count': 4})
            elif goal == MCGoal.MINE_IRON and self._count_item(state, 'minecraft:raw_iron') >= 3:
                state.inventory.append({'slot': len(state.inventory), 'item': 'minecraft:iron_ingot', 'count': 3})
            elif goal == MCGoal.MINE_DIAMOND and self._count_item(state, 'minecraft:diamond') >= 3:
                state.inventory.append({'slot': len(state.inventory), 'item': 'minecraft:diamond_pickaxe', 'count': 1})

    def _count_item(self, state: MCWorldState, item_name: str) -> int:
        return sum(e.get('count', 0) for e in state.inventory if e.get('item') == item_name)

    def _has_food(self, state: MCWorldState) -> bool:
        return any('food' in e.get('item', '') or 'bread' in e.get('item', '') or
                   'apple' in e.get('item', '') for e in state.inventory)

    def run_episode(self, agent, max_steps: int = None) -> Dict:
        if max_steps is None:
            max_steps = self.max_steps
        obs = self.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action, log_prob, value = agent.select_action(obs)
            obs, reward, done, info = self.step(action)
            total_reward += reward
            steps += 1

            agent.store_transition(obs, action, log_prob, value, reward, done)

            if done:
                break

        learn_info = agent.learn()

        return {
            'episode_reward': total_reward,
            'steps': steps,
            'milestones': info.get('milestones', []),
            'learn_info': learn_info,
        }
