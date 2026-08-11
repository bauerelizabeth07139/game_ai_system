import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class PPOMemory:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def sample(self) -> Tuple:
        if len(self.states) < self.batch_size:
            return None

        indices = random.sample(range(len(self.states)), self.batch_size)
        return (
            torch.FloatTensor(np.array([self.states[i] for i in indices])),
            torch.FloatTensor(np.array([self.actions[i] for i in indices])),
            torch.FloatTensor(np.array([self.log_probs[i] for i in indices]).reshape(-1, 1)),
            torch.FloatTensor(np.array([self.values[i] for i in indices]).reshape(-1, 1)),
            torch.FloatTensor(np.array([self.rewards[i] for i in indices]).reshape(-1, 1)),
            torch.FloatTensor(np.array([self.dones[i] for i in indices]).reshape(-1, 1)),
        )

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared(x)
        action_mean = self.actor_mean(shared)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic(shared)
        return action_mean, action_std, value

    def sample_action(self, state: torch.Tensor):
        action_mean, action_std, value = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value


class PPOAgent:
    def __init__(self,
                 input_dim: int = 580,
                 action_dim: int = 40,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 c1: float = 0.5,
                 c2: float = 0.01,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        self.actor_critic = ActorCritic(input_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        self.memory = PPOMemory(batch_size)
        self.training_step = 0

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.sample_action(state_tensor)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.memory.store(state, action, log_prob, value, reward, done)

    def learn(self) -> Dict:
        if len(self.memory) < self.memory.batch_size:
            self.memory.clear()
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(
            np.array(self.memory.log_probs).reshape(-1, 1)
        ).to(self.device)
        rewards = torch.FloatTensor(
            np.array(self.memory.rewards).reshape(-1, 1)
        ).to(self.device)
        dones = torch.FloatTensor(
            np.array(self.memory.dones).reshape(-1, 1)
        ).to(self.device)

        advantages, returns = self._compute_gae(rewards, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            dataset_size = len(states)
            indices = list(range(dataset_size))
            random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                action_mean, action_std, values = self.actor_critic(batch_states)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
                self.training_step += 1

        self.memory.clear()

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def _compute_gae(self, rewards: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(
                        np.array(self.memory.states[t + 1])
                    ).unsqueeze(0).to(self.device)
                    _, _, next_value_tensor = self.actor_critic.forward(state_t)
                    next_value = next_value_tensor.item()

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        return advantages, returns

    def save(self, path: str):
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        logger.info(f"Model loaded from {path} (step {self.training_step})")


class CurriculumTrainer:
    def __init__(self, agent: PPOAgent):
        self.agent = agent
        self.curriculum = [
            {'goal': 'GATHER_WOOD', 'episodes': 500, 'reward_threshold': 10.0},
            {'goal': 'CRAFT_TOOLS', 'episodes': 500, 'reward_threshold': 15.0},
            {'goal': 'MINE_STONE', 'episodes': 500, 'reward_threshold': 20.0},
            {'goal': 'MINE_IRON', 'episodes': 800, 'reward_threshold': 25.0},
            {'goal': 'MINE_DIAMOND', 'episodes': 1000, 'reward_threshold': 30.0},
            {'goal': 'KILL_ENDER_DRAGON', 'episodes': 2000, 'reward_threshold': 100.0},
        ]
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_rewards: List[float] = []

    def should_advance(self) -> bool:
        if len(self.stage_rewards) < 100:
            return False
        avg_reward = np.mean(self.stage_rewards[-100:])
        threshold = self.curriculum[self.current_stage]['reward_threshold']
        return avg_reward >= threshold

    def advance_stage(self):
        if self.current_stage < len(self.curriculum) - 1:
            self.current_stage += 1
            self.stage_episodes = 0
            self.stage_rewards.clear()
            logger.info(f"Advanced to curriculum stage {self.current_stage}: "
                       f"{self.curriculum[self.current_stage]['goal']}")

    def get_current_goal(self) -> str:
        return self.curriculum[self.current_stage]['goal']

    def get_max_episodes(self) -> int:
        return self.curriculum[self.current_stage]['episodes']
