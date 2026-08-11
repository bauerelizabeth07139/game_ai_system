import time
import threading
import logging
import json
from typing import Optional, Dict, List
from collections import deque

from src.minecraft.mc_controller import MCController, MCCommandTranslator
from src.minecraft.mc_perception import MCPerception, MCWorldState
from src.minecraft.mc_policy import MCPolicy, MCActionSpace
from src.minecraft.mc_goals import MCGoalPlanner, MCGoal
from src.minecraft.mc_environment import MCEnvironment
from src.minecraft.mc_prompt import MCPromptBuilder

logger = logging.getLogger(__name__)


class MinecraftAIBot:
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 25575,
                 use_llm: bool = False,
                 policy_path: Optional[str] = None,
                 config: Optional[Dict] = None):
        self.controller = MCController(host, port)
        self.translator = MCCommandTranslator()
        self.perception = MCPerception()
        self.goal_planner = MCGoalPlanner()
        self.action_space = MCActionSpace()
        self.prompt_builder = MCPromptBuilder()
        self.environment = MCEnvironment()

        self.policy = MCPolicy()
        if policy_path:
            self.policy.load_pretrained(policy_path)

        self.use_llm = use_llm
        self.config = config or {}
        self.fps = self.config.get('fps', 10)
        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0.1

        self._running = False
        self._paused = False
        self._brain_thread: Optional[threading.Thread] = None
        self._latest_state: Optional[MCWorldState] = None
        self._state_lock = threading.Lock()
        self._action_queue = deque(maxlen=50)
        self._frame_count = 0
        self._stats = {
            'frames': 0,
            'actions_sent': 0,
            'blocks_broken': 0,
            'hostiles_killed': 0,
            'milestones': [],
            'start_time': None,
        }

        if self.use_llm:
            from src.decision.llm_client import LLMClient
            self.llm_client = LLMClient(self.config.get('decision', {}))
        else:
            self.llm_client = None

        self.controller.add_state_listener(self._on_state_update)

    def start(self):
        if not self.controller.connect():
            logger.error("Failed to connect to Minecraft AI mod")
            return False

        logger.info("Bot connected. Spawning AI player...")
        time.sleep(2)

        self.controller.spawn_ai_player()
        time.sleep(3)

        self.controller.start_ai()
        time.sleep(1)

        self._running = True
        self._paused = False
        self._stats['start_time'] = time.time()

        self._brain_thread = threading.Thread(target=self._brain_loop, daemon=True)
        self._brain_thread.start()

        logger.info("Minecraft AI Bot started successfully")
        return True

    def stop(self):
        self._running = False
        if self.controller:
            self.controller.stop_ai()
            self.controller.disconnect()
        logger.info("Minecraft AI Bot stopped")
        self._print_stats()

    def pause(self):
        self._paused = True
        self.controller.stop_ai()

    def resume(self):
        self._paused = False
        self.controller.start_ai()

    def _on_state_update(self, state: MCWorldState):
        with self._state_lock:
            self._latest_state = state

    def _brain_loop(self):
        logger.info("Brain loop started")
        last_goal_update = 0

        while self._running:
            start_time = time.time()

            if self._paused:
                time.sleep(0.1)
                continue

            with self._state_lock:
                state = self._latest_state

            if state is None:
                self.controller.request_state()
                time.sleep(0.1)
                continue

            self.perception.process_state(state)

            if self._frame_count - last_goal_update >= 50:
                new_goal = self.goal_planner.update_goal(state)
                self.controller.set_goal(new_goal.value)
                last_goal_update = self._frame_count

            action = self._decide_action(state)

            if action is not None:
                commands = self.action_space.to_minecraft_command(action)
                for cmd in commands:
                    self.controller.execute_action(cmd)
                    self._stats['actions_sent'] += 1

            self._frame_count += 1
            self._stats['frames'] += 1

            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed)
            time.sleep(sleep_time)

    def _decide_action(self, state: MCWorldState) -> Optional:
        goal = self.goal_planner.current_goal

        if self.use_llm and self.llm_client and self._frame_count % 50 == 0:
            try:
                system_prompt = self.prompt_builder.build_system_prompt(goal.value)
                user_prompt = self.prompt_builder.build_user_prompt(
                    state, self.perception.get_decision_history()
                )
                llm_response = self.llm_client.call_sync(system_prompt, user_prompt)
                decision = self.prompt_builder.parse_decision(llm_response)

                strategy_action_map = {
                    'ATTACK': lambda: self._combat_action(state),
                    'FLEE': lambda: self._flee_action(state),
                    'MINE': lambda: self._mine_action(state),
                    'CRAFT': lambda: self._craft_action(state),
                    'BUILD': lambda: self._build_action(state),
                    'EAT': lambda: self._eat_action(state),
                    'SLEEP': lambda: self._sleep_action(state),
                    'EXPLORE': lambda: self._explore_action(state),
                    'FARM': lambda: self._explore_action(state),
                    'TRADE': lambda: self._explore_action(state),
                }

                mapped_action = strategy_action_map.get(
                    decision['strategy'],
                    self._explore_action
                )
                return mapped_action(state)

            except Exception as e:
                logger.warning(f"LLM decision failed, falling back to policy: {e}")

        state_vector = state.to_state_vector()
        return self._policy_action(state, state_vector, goal.value)

    def _policy_action(self, state: MCWorldState, state_vector, goal_name: str):
        try:
            return self.policy.get_action(state_vector, goal_name)
        except Exception as e:
            logger.error(f"Policy inference failed: {e}")
            return self._manual_action(state)

    def _manual_action(self, state: MCWorldState):
        actions = self.goal_planner.get_goal_actions(state)
        return self._actions_to_vector(actions)

    def _combat_action(self, state: MCWorldState):
        return self._actions_to_vector(['ATTACK', 'LMB'])

    def _flee_action(self, state: MCWorldState):
        return self._actions_to_vector(['KEY_W', 'KEY_SPACE', 'KEY_SHIFT'])

    def _mine_action(self, state: MCWorldState):
        return self._actions_to_vector(['LMB', 'KEY_W'])

    def _craft_action(self, state: MCWorldState):
        return self._actions_to_vector(['CRAFT', 'KEY_E'])

    def _build_action(self, state: MCWorldState):
        return self._actions_to_vector(['RMB', 'KEY_SPACE'])

    def _eat_action(self, state: MCWorldState):
        return self._actions_to_vector(['RMB'])

    def _sleep_action(self, state: MCWorldState):
        return self._actions_to_vector(['RMB'])

    def _explore_action(self, state):
        actions = self.goal_planner.get_goal_actions(state)
        return self._actions_to_vector(actions)

    def _actions_to_vector(self, actions: List[str]) -> bytes:
        import numpy as np
        action_vec = np.zeros(40, dtype=np.float32)
        action_vec[4] = 0.6

        action_map = {
            'LMB': 4, 'RMB': 5, 'MMB': 6,
            'KEY_W': 7, 'KEY_A': 8, 'KEY_S': 9, 'KEY_D': 10,
            'KEY_SPACE': 11, 'KEY_SHIFT': 12, 'KEY_CTRL': 13,
            'KEY_TAB': 14, 'KEY_ESC': 15,
            'KEY_1': 16, 'KEY_2': 17, 'KEY_3': 18,
            'KEY_4': 19, 'KEY_5': 20,
            'KEY_Q': 26, 'KEY_E': 27, 'KEY_R': 28, 'KEY_F': 29,
            'KEY_G': 30, 'KEY_H': 31, 'KEY_Z': 32,
            'KEY_X': 33, 'KEY_C': 34, 'KEY_V': 35,
            'KEY_B': 36, 'KEY_N': 37, 'KEY_T': 38, 'KEY_M': 39,
        }

        for action in actions:
            if action == 'EXPLORE':
                action_vec[7] = 1.0
                action_vec[10] = 1.0
            elif action == 'ATTACK':
                action_vec[4] = 1.0
                action_vec[7] = 1.0
            elif action == 'MINE':
                action_vec[4] = 1.0
            elif action == 'CRAFT':
                action_vec[27] = 1.0
            elif action == 'PLACE':
                action_vec[5] = 1.0
            elif action == 'DROP':
                action_vec[26] = 1.0
            elif action == 'EAT':
                action_vec[5] = 1.0
            elif action in action_map:
                action_vec[action_map[action]] = 1.0

        return action_vec

    def _print_stats(self):
        if not self._stats['start_time']:
            return
        elapsed = time.time() - self._stats['start_time']
        logger.info(f"=== AI Bot Stats ===")
        logger.info(f"  Runtime: {elapsed:.1f}s")
        logger.info(f"  Frames: {self._stats['frames']}")
        logger.info(f"  Actions: {self._stats['actions_sent']}")
        logger.info(f"  Milestones: {self._stats['milestones']}")
        logger.info(f"===================")

    def get_status(self) -> Dict:
        state = self._latest_state
        return {
            'running': self._running,
            'paused': self._paused,
            'frames': self._frame_count,
            'health': state.health if state else 0,
            'goal': state.current_goal if state else 'NONE',
            'stats': self._stats,
            'goal_planner_status': self.goal_planner.get_status_string(),
        }
