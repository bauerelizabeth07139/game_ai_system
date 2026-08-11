#!/usr/bin/env python3
"""
Test suite for Minecraft AI modules
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.minecraft.mc_perception import MCWorldState, MCPerception
from src.minecraft.mc_policy import MCPolicy, MCActionSpace
from src.minecraft.mc_goals import MCGoalPlanner, MCGoal, PROGRESSION_CHAIN
from src.minecraft.mc_environment import MCEnvironment


class TestMCWorldState(unittest.TestCase):
    def test_state_from_json(self):
        json_str = '{"health":18.0,"maxHealth":20.0,"hunger":15.0,"posX":100.5,"posY":64.0,"posZ":200.3,"dimension":"minecraft:overworld","inventory":[{"slot":0,"item":"minecraft:oak_log","count":4}]}'
        state = MCWorldState.from_json(json_str)
        self.assertEqual(state.health, 18.0)
        self.assertEqual(state.pos_x, 100.5)
        self.assertEqual(len(state.inventory), 1)

    def test_state_vector_shape(self):
        state = MCWorldState()
        vec = state.to_state_vector()
        self.assertEqual(vec.shape, (141,))
        self.assertEqual(vec.dtype, np.float32)

    def test_state_vector_values(self):
        state = MCWorldState()
        state.health = 10.0
        state.hunger = 15.0
        vec = state.to_state_vector()
        self.assertAlmostEqual(vec[0], 0.5, places=1)
        self.assertAlmostEqual(vec[1], 0.75, places=1)


class TestMCPerception(unittest.TestCase):
    def test_process_state(self):
        perception = MCPerception()
        state = MCWorldState()
        entry = perception.process_state(state)
        self.assertIn('state_vector', entry)
        self.assertIn('semantic_data', entry)
        self.assertEqual(entry['state_vector'].shape, (141,))

    def test_frame_counter(self):
        perception = MCPerception()
        state = MCWorldState()
        e1 = perception.process_state(state)
        e2 = perception.process_state(state)
        self.assertEqual(e2['frame_id'], e1['frame_id'] + 1)


class TestMCPolicy(unittest.TestCase):
    def test_model_creation(self):
        policy = MCPolicy()
        self.assertIsNotNone(policy)
        total_params = sum(p.numel() for p in policy.parameters())
        self.assertGreater(total_params, 1000)

    def test_forward_pass(self):
        policy = MCPolicy()
        batch_states = torch.randn(2, 141 * 3 + 32 + 1)
        batch_strategies = torch.LongTensor([0, 1])
        game_type = torch.LongTensor([5, 5])

        batch_size = 2
        state_tensor = torch.randn(batch_size, 141 * 3)
        strategy_tensor = torch.LongTensor([0] * batch_size)

    def test_get_action_shape(self):
        policy = MCPolicy()
        state_vec = np.random.randn(141).astype(np.float32)
        action = policy.get_action(state_vec, 'GATHER_WOOD')
        self.assertEqual(action.shape, (40,))
        self.assertTrue(np.all(action >= 0))
        self.assertTrue(np.all(action <= 1))

    def test_save_load(self):
        import tempfile
        policy = MCPolicy()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            policy.save(path)
            loaded = MCPolicy()
            loaded.load_pretrained(path)
        finally:
            os.unlink(path)


class TestMCActionSpace(unittest.TestCase):
    def test_to_minecraft_command(self):
        action_vec = np.zeros(40, dtype=np.float32)
        action_vec[4] = 1.0
        action_vec[7] = 1.0
        commands = MCActionSpace.to_minecraft_command(action_vec)
        self.assertTrue(any('LMB' in c or 'PRESS:W' in c for c in commands))

    def test_move_command(self):
        action_vec = np.zeros(40, dtype=np.float32)
        action_vec[7] = 1.0
        action_vec[10] = 1.0
        commands = MCActionSpace.to_minecraft_command(action_vec)
        self.assertGreater(len(commands), 0)


class TestMCGoalPlanner(unittest.TestCase):
    def test_initial_goal(self):
        planner = MCGoalPlanner()
        self.assertEqual(planner.current_goal, MCGoal.GATHER_WOOD)

    def test_progression_chain(self):
        self.assertGreater(len(PROGRESSION_CHAIN), 5)
        self.assertEqual(PROGRESSION_CHAIN[0][0], MCGoal.GATHER_WOOD)
        self.assertEqual(PROGRESSION_CHAIN[-1][0], MCGoal.KILL_ENDER_DRAGON)

    def test_goal_actions(self):
        planner = MCGoalPlanner()
        state = MCWorldState()
        for goal in MCGoal:
            if goal != MCGoal.DONE and goal != MCGoal.NONE:
                planner.current_goal = goal
                actions = planner.get_goal_actions(state)
                self.assertGreater(len(actions), 0, f"No actions for goal {goal}")

    def test_has_item_detection(self):
        planner = MCGoalPlanner()
        state = MCWorldState()
        state.hotbar = [
            {'slot': 0, 'item': 'minecraft:oak_log', 'count': 5}
        ]
        self.assertTrue(planner._has_item(state, ['log'], 1))

    def test_update_goal_progression(self):
        planner = MCGoalPlanner()
        state = MCWorldState()
        state.hotbar = [
            {'slot': 0, 'item': 'minecraft:oak_log', 'count': 8}
        ]
        new_goal = planner.update_goal(state)
        self.assertIn(new_goal, [MCGoal.GATHER_WOOD, MCGoal.CRAFT_TOOLS])


class TestMCEnvironment(unittest.TestCase):
    def test_reset(self):
        env = MCEnvironment()
        obs = env.reset()
        expected_dim = 141 * 4 + 16
        self.assertEqual(obs.shape, (expected_dim,))

    def test_step(self):
        env = MCEnvironment()
        state = MCWorldState()
        obs = env.reset(state)
        action = np.ones(40, dtype=np.float32) * 0.5
        obs2, reward, done, info = env.step(action, state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('health', info)

    def test_milestone_detection(self):
        env = MCEnvironment()
        state = MCWorldState()
        state.hotbar = [
            {'slot': 0, 'item': 'minecraft:oak_log', 'count': 8}
        ]
        env.reset(state)
        action = np.zeros(40, dtype=np.float32)
        obs, reward, done, info = env.step(action, state)
        self.assertGreaterEqual(reward, -1)


if __name__ == '__main__':
    import torch
    unittest.main(verbosity=2)
