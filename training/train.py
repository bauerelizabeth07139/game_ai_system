#!/usr/bin/env python3
"""
Minecraft AI Training Script
Uses PPO (Proximal Policy Optimization) with curriculum learning
to train an AI agent to beat Minecraft.

Training Stages:
  1. Gather Wood (500 episodes)
  2. Craft Tools (500 episodes)
  3. Mine Stone (500 episodes)
  4. Mine Iron (800 episodes)
  5. Mine Diamond (1000 episodes)
  6. Beat Ender Dragon (2000 episodes)
"""

import argparse
import logging
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ppo_agent import PPOAgent, CurriculumTrainer
from training.minecraft_env import TrainingMinecraftEnv
from src.minecraft.mc_policy import MCPolicy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('Training')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Minecraft AI Agent')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Total number of episodes to train (default: 5000)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save model every N episodes (default: 100)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log stats every N episodes (default: 10)')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load pre-trained model')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to train on (cpu/cuda)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluation mode (no training)')
    return parser.parse_args()


def save_policy(policy: MCPolicy, path: str):
    policy.save(path)


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = TrainingMinecraftEnv(headless=True, max_steps=1000)

    agent = PPOAgent(
        input_dim=env.environment._observation_space_shape[0],
        action_dim=env.environment._action_space_shape[0],
        hidden_dim=256,
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,
        c2=0.01,
        batch_size=64,
        n_epochs=10,
        device=args.device,
    )

    if args.load:
        agent.load(args.load)
        logger.info(f"Loaded model from {args.load}")

    curriculum = CurriculumTrainer(agent)

    if args.evaluate:
        evaluate(agent, env, args)
        return

    logger.info("=" * 60)
    logger.info("  Minecraft AI Training")
    logger.info(f"  Episodes: {args.episodes} | Device: {args.device}")
    logger.info(f"  Save dir: {args.save_dir}")
    logger.info(f"  Curriculum stages: {len(curriculum.curriculum)}")
    logger.info("=" * 60)

    total_rewards = []
    best_reward = float('-inf')
    milestone_counts = {}

    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        episode_start = time.time()

        current_goal = curriculum.get_current_goal()
        env.goal_planner.current_goal = current_goal
        env.environment.current_goal = current_goal

        ep_result = env.run_episode(agent, max_steps=env.max_steps)
        total_rewards.append(ep_result['episode_reward'])

        for milestone in ep_result.get('milestones', []):
            milestone_counts[milestone] = milestone_counts.get(milestone, 0) + 1

        if curriculum.should_advance():
            curriculum.advance_stage()

        avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
        if ep_result['episode_reward'] > best_reward:
            best_reward = ep_result['episode_reward']
            agent.save(os.path.join(args.save_dir, 'best_model.pt'))
            save_policy(env.policy, os.path.join(args.save_dir, 'best_policy.pt'))

        if episode % args.save_interval == 0:
            agent.save(os.path.join(args.save_dir, f'checkpoint_{episode}.pt'))
            save_policy(env.policy, os.path.join(args.save_dir, f'policy_{episode}.pt'))

        if episode % args.log_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            logger.info(
                f"Ep {episode}/{args.episodes} | "
                f"Reward: {ep_result['episode_reward']:.1f} | "
                f"Avg100: {avg_reward:.1f} | "
                f"Best: {best_reward:.1f} | "
                f"Goal: {current_goal} | "
                f"Steps: {ep_result['steps']} | "
                f"Milestones: {len(ep_result.get('milestones', []))} | "
                f"Speed: {eps_per_sec:.1f} eps/s | "
                f"Learn: {ep_result['learn_info']}"
            )

    agent.save(os.path.join(args.save_dir, 'final_model.pt'))
    save_policy(env.policy, os.path.join(args.save_dir, 'final_policy.pt'))

    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.info(f"Best reward: {best_reward:.1f}")
    logger.info(f"Milestones achieved:")
    for name, count in sorted(milestone_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  - {name}: {count} times")


def evaluate(agent, env, args):
    logger.info("Starting evaluation...")
    total_reward = 0
    for episode in range(10):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _, _ = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        logger.info(f"  Episode {episode + 1}: Reward = {ep_reward:.1f}, "
                    f"Milestones: {info.get('milestones', [])}")
    logger.info(f"Average reward over 10 episodes: {total_reward / 10:.1f}")


if __name__ == '__main__':
    main()
