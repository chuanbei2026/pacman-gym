"""
DQN training for Pac-Man.
Usage:
    python3 -m pacman_gym.train               # Train the agent
    python3 -m pacman_gym.train --play        # Watch the trained agent play
    python3 -m pacman_gym.train --play-human  # Render during training
"""

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .gym_env import PacManEnv

MODEL_PATH = Path(__file__).parent / "dqn_model.pt"


class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network."""

    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 80000,
        buffer_size: int = 200000,
        batch_size: int = 128,
        target_update: int = 500,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.policy_net = DQN(obs_size, n_actions).to(self.device)
        self.target_net = DQN(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.total_episodes = 0

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.steps_done / self.epsilon_decay)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: Path = MODEL_PATH):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "total_episodes": self.total_episodes,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: Path = MODEL_PATH):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.total_episodes = checkpoint.get("total_episodes", 0)
        print(f"Model loaded from {path} (episodes: {self.total_episodes})")


def train(episodes: int = 2000, render: bool = False):
    """Train the DQN agent."""
    env = PacManEnv(render_mode="human" if render else None, max_steps=2000)
    obs, _ = env.reset()
    obs_size = obs.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_size, n_actions)

    # Load existing model if available
    if MODEL_PATH.exists():
        agent.load()
        print("Resumed training from saved model")

    best_score = 0
    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)

    print(f"Training DQN agent for {episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Obs size: {obs_size}, Actions: {n_actions}")
    print("-" * 60)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.buffer.push(obs, action, reward, next_obs, done or truncated)
            loss = agent.train_step()

            total_reward += reward
            total_loss += loss
            steps += 1
            obs = next_obs

            if render:
                env.render()

            if done or truncated:
                break

        score = info["score"]
        recent_scores.append(score)
        recent_rewards.append(total_reward)

        if score > best_score:
            best_score = score
            agent.save()

        if (ep + 1) % 20 == 0:
            avg_score = np.mean(recent_scores)
            avg_reward = np.mean(recent_rewards)
            print(
                f"Ep {ep + 1:5d} | "
                f"Score: {score:5.0f} | Avg Score: {avg_score:6.1f} | "
                f"Best: {best_score:5.0f} | "
                f"Reward: {total_reward:7.1f} | Avg Reward: {avg_reward:7.1f} | "
                f"Eps: {agent.epsilon:.3f} | Steps: {steps:4d}"
            )

        # Save periodically
        if (ep + 1) % 200 == 0:
            agent.save()

    agent.save()
    env.close()
    print(f"\nTraining complete! Best score: {best_score}")


def play(model_path: Path = MODEL_PATH):
    """Watch the trained agent play."""
    env = PacManEnv(render_mode="human", max_steps=3000)
    obs, _ = env.reset()

    agent = DQNAgent(obs.shape[0], env.action_space.n)
    if model_path.exists():
        agent.load(model_path)
    else:
        print(f"No model found at {model_path}. Playing with random agent.")

    import pygame

    # Initialize pygame and render first frame so the window exists
    env.render()

    print("Watching trained agent play. Press ESC to quit, SPACE to restart.")

    while True:
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            # Handle events after render (window is guaranteed to exist)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return

            if done or truncated:
                print(f"Game over! Score: {info['score']}, Won: {info['won']}")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                env.close()
                                return
                            if event.key == pygame.K_SPACE:
                                waiting = False

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Pac-Man DQN Training")
    parser.add_argument("--play", action="store_true", help="Watch trained agent play")
    parser.add_argument("--play-human", action="store_true", help="Train with rendering")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    args = parser.parse_args()

    if args.play:
        play()
    else:
        train(episodes=args.episodes, render=args.play_human)


if __name__ == "__main__":
    main()
