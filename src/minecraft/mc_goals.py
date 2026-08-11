from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

from src.minecraft.mc_perception import MCWorldState


class MCGoal(str, Enum):
    NONE = "NONE"
    GATHER_WOOD = "GATHER_WOOD"
    CRAFT_TOOLS = "CRAFT_TOOLS"
    BUILD_SHELTER = "BUILD_SHELTER"
    MINE_STONE = "MINE_STONE"
    FIND_FOOD = "FIND_FOOD"
    MINE_IRON = "MINE_IRON"
    CRAFT_ARMOR = "CRAFT_ARMOR"
    MINE_DIAMOND = "MINE_DIAMOND"
    EXPLORE = "EXPLORE"
    FIND_LAVA = "FIND_LAVA"
    BUILD_PORTAL = "BUILD_PORTAL"
    ENTER_NETHER = "ENTER_NETHER"
    FIND_FORTRESS = "FIND_FORTRESS"
    KILL_BLAZE = "KILL_BLAZE"
    FIND_END_PORTAL = "FIND_END_PORTAL"
    KILL_ENDER_DRAGON = "KILL_ENDER_DRAGON"
    DONE = "DONE"


PROGRESSION_CHAIN = [
    (MCGoal.GATHER_WOOD, ["log", "planks"], 4),
    (MCGoal.CRAFT_TOOLS, ["pickaxe", "axe", "sword"], 1),
    (MCGoal.MINE_STONE, ["cobblestone", "stone"], 8),
    (MCGoal.FIND_FOOD, ["bread", "apple", "cooked_beef", "cooked_porkchop"], 3),
    (MCGoal.MINE_IRON, ["iron_ingot", "raw_iron"], 3),
    (MCGoal.CRAFT_ARMOR, ["iron_chestplate", "iron_leggings", "iron_boots", "iron_helmet"], 1),
    (MCGoal.MINE_DIAMOND, ["diamond"], 3),
    (MCGoal.FIND_LAVA, ["lava_bucket"], 1),
    (MCGoal.BUILD_PORTAL, ["obsidian"], 10),
    (MCGoal.ENTER_NETHER, ["the_nether"], 0),
    (MCGoal.FIND_FORTRESS, ["nether_bricks", "blaze_powder"], 1),
    (MCGoal.KILL_BLAZE, ["blaze_rod"], 3),
    (MCGoal.FIND_END_PORTAL, ["ender_pearl"], 12),
    (MCGoal.KILL_ENDER_DRAGON, ["dragon_head", "dragon_egg"], 1),
]


@dataclass
class GoalPlan:
    goal: MCGoal
    priority: int = 0
    target_block: Optional[Tuple[int, int, int]] = None
    target_entity_id: Optional[str] = None
    required_items: List[str] = field(default_factory=list)
    subgoals: List['GoalPlan'] = field(default_factory=list)
    completed: bool = False


class MCGoalPlanner:
    def __init__(self):
        self.current_goal: MCGoal = MCGoal.GATHER_WOOD
        self.goal_history: List[MCGoal] = []
        self.completed_goals: List[MCGoal] = []
        self.goal_stack: List[GoalPlan] = []
        self._last_check_step = 0
        self._stuck_counter = 0

    def select_next_goal(self, state: MCWorldState) -> MCGoal:
        for goal, items, count in PROGRESSION_CHAIN:
            if goal in self.completed_goals:
                continue
            if self._check_progression(state, items, count, goal):
                return goal

        env_score = self._evaluate_environment(state)
        if state.nearby_hostiles > 0 and state.health < 10:
            return MCGoal.MINE_STONE

        if state.hunger < 8 and not self._has_food(state):
            return MCGoal.FIND_FOOD

        return self.current_goal

    def update_goal(self, state: MCWorldState) -> MCGoal:
        next_goal = self.select_next_goal(state)

        if next_goal != self.current_goal:
            if self._is_goal_completed(self.current_goal, state):
                self.completed_goals.append(self.current_goal)

            self.goal_history.append(self.current_goal)
            self.current_goal = next_goal

        return self.current_goal

    def get_goal_actions(self, state: MCWorldState) -> List[str]:
        goal_actions = {
            MCGoal.GATHER_WOOD: self._actions_gather_wood,
            MCGoal.CRAFT_TOOLS: self._actions_craft_tools,
            MCGoal.MINE_STONE: self._actions_mine_stone,
            MCGoal.FIND_FOOD: self._actions_find_food,
            MCGoal.MINE_IRON: self._actions_mine_iron,
            MCGoal.CRAFT_ARMOR: self._actions_craft_armor,
            MCGoal.MINE_DIAMOND: self._actions_mine_diamond,
            MCGoal.FIND_LAVA: self._actions_find_lava,
            MCGoal.BUILD_PORTAL: self._actions_build_portal,
            MCGoal.ENTER_NETHER: self._actions_enter_nether,
            MCGoal.FIND_FORTRESS: self._actions_find_fortress,
            MCGoal.KILL_BLAZE: self._actions_kill_blaze,
            MCGoal.FIND_END_PORTAL: self._actions_find_end_portal,
            MCGoal.KILL_ENDER_DRAGON: self._actions_kill_dragon,
        }

        action_func = goal_actions.get(self.current_goal, self._actions_explore)
        return action_func(state)

    def _check_progression(self, state: MCWorldState, items: List[str],
                           count: int, goal: MCGoal) -> bool:
        if goal == MCGoal.ENTER_NETHER:
            return not self._has_item(state, ["the_nether"], 0)
        if goal == MCGoal.GATHER_WOOD:
            return True
        return not self._has_any_item(state, items, count)

    def _has_item(self, state: MCWorldState, item_names: List[str], count: int) -> bool:
        all_items = state.hotbar + state.inventory
        for entry in all_items:
            item = entry.get('item', '').lower()
            cnt = entry.get('count', 0)
            if cnt >= count:
                for name in item_names:
                    if name.lower() in item:
                        return True
        return False

    def _has_any_item(self, state: MCWorldState, item_names: List[str], count: int) -> bool:
        all_items = state.hotbar + state.inventory
        total = 0
        for entry in all_items:
            item = entry.get('item', '').lower()
            cnt = entry.get('count', 0)
            for name in item_names:
                if name.lower() in item:
                    total += cnt
        return total >= count

    def _has_food(self, state: MCWorldState) -> bool:
        food_items = ['bread', 'apple', 'cooked_beef', 'cooked_porkchop',
                       'cooked_chicken', 'cooked_mutton', 'steak', 'carrot',
                       'potato', 'beetroot', 'cookie', 'melon', 'pumpkin_pie']
        return self._has_any_item(state, food_items, 1)

    def _is_goal_completed(self, goal: MCGoal, state: MCWorldState) -> bool:
        if goal == MCGoal.DONE:
            return True
        for g, items, count in PROGRESSION_CHAIN:
            if g == goal:
                if goal == MCGoal.ENTER_NETHER:
                    return state.dimension == 'minecraft:the_nether'
                return self._has_any_item(state, items, count)
        return False

    def _evaluate_environment(self, state: MCWorldState) -> float:
        score = 0.0
        if state.is_night:
            score -= 3
        if state.nearby_hostiles > 2:
            score -= 5
        if state.hunger < 5:
            score -= 8
        if state.is_raining:
            score -= 1
        return score

    def _actions_gather_wood(self, state: MCWorldState) -> List[str]:
        if state.block_looking_at and 'log' in state.block_looking_at:
            return ["LMB"]
        return ["MINE", "LMB"]

    def _actions_craft_tools(self, state: MCWorldState) -> List[str]:
        return ["CRAFT:pickaxe", "CRAFT:axe", "CRAFT:sword"]

    def _actions_mine_stone(self, state: MCWorldState) -> List[str]:
        return ["MINE", "LMB"]

    def _actions_find_food(self, state: MCWorldState) -> List[str]:
        return ["EAT", "EXPLORE"]

    def _actions_mine_iron(self, state: MCWorldState) -> List[str]:
        return ["MINE", "LMB"]

    def _actions_craft_armor(self, state: MCWorldState) -> List[str]:
        return ["CRAFT:chestplate", "CRAFT:leggings", "CRAFT:boots"]

    def _actions_mine_diamond(self, state: MCWorldState) -> List[str]:
        return ["MINE", "LMB"]

    def _actions_find_lava(self, state: MCWorldState) -> List[str]:
        return ["EXPLORE", "MINE"]

    def _actions_build_portal(self, state: MCWorldState) -> List[str]:
        return ["PLACE", "RMB"]

    def _actions_enter_nether(self, state: MCWorldState) -> List[str]:
        return ["USE", "RMB"]

    def _actions_find_fortress(self, state: MCWorldState) -> List[str]:
        return ["EXPLORE"]

    def _actions_kill_blaze(self, state: MCWorldState) -> List[str]:
        return ["ATTACK", "RMB"]

    def _actions_find_end_portal(self, state: MCWorldState) -> List[str]:
        return ["EXPLORE", "USE"]

    def _actions_kill_dragon(self, state: MCWorldState) -> List[str]:
        return ["ATTACK", "RMB", "ATTACK"]

    def _actions_explore(self, state: MCWorldState) -> List[str]:
        return ["EXPLORE"]

    def get_status_string(self) -> str:
        completed = len(self.completed_goals)
        total = len(PROGRESSION_CHAIN)
        return f"Progress: {completed}/{total} | Current: {self.current_goal.name}"
