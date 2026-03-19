"""
Q*bert DQN Agent using Gymnasium + ALE.

CNN-based Deep Q-Network with frame stacking, experience replay, and
epsilon-greedy exploration. Trains on 84x84 grayscale frames.

Usage:
    python qbert_agent.py train        # Train (resumes from checkpoint if exists)
    python qbert_agent.py play         # Watch trained agent play
    python qbert_agent.py random       # Watch random actions (baseline)

Model checkpoint: qbert_dqn.pt (saved automatically during training)
Requires: gymnasium, ale-py, numpy, torch
"""

import sys
import random
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent / "qbert_dqn.pt"

# --- Hyperparameters ---
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100_000  # steps over which epsilon decays
LEARNING_RATE = 1e-4
MEMORY_SIZE = 100_000
TARGET_UPDATE = 1000  # sync target net every N steps
TRAIN_EPISODES = 1000
FRAME_STACK = 4
FRAME_SKIP = 4  # ALE default is already 4 for v5, but we handle it explicitly


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert 210x160x3 RGB frame to 84x84 grayscale float."""
    # Grayscale via luminance
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    # Downsample to 84x84 using simple slicing/averaging
    # Crop to 160x160 (remove top score area and bottom padding), then resize
    cropped = gray[30:190, :]  # 160x160
    # Simple resize via strided indexing (fast, good enough)
    h, w = cropped.shape
    new_h, new_w = 84, 84
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    resized = cropped[np.ix_(row_idx, col_idx)]
    return resized / 255.0


class FrameStacker:
    """Stacks N preprocessed frames into a single observation."""

    def __init__(self, n=FRAME_STACK):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(frame)
        for _ in range(self.n):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(frame))
        return np.stack(self.frames, axis=0)


class ReplayMemory:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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


class DQN(nn.Module):
    """CNN that maps stacked frames -> Q-values for each action."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, FRAME_STACK, 84, 84)
            conv_out = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class QbertAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.policy_net = DQN(n_actions).to(DEVICE)
        self.target_net = DQN(n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory()
        self.steps_done = 0

    def select_action(self, state: np.ndarray, eval_mode=False) -> int:
        if eval_mode:
            eps = 0.01
        else:
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(
                -self.steps_done / EPS_DECAY
            )
            self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states_t = torch.from_numpy(states).float().to(DEVICE)
        actions_t = torch.from_numpy(actions).long().to(DEVICE)
        rewards_t = torch.from_numpy(rewards).to(DEVICE)
        next_states_t = torch.from_numpy(next_states).float().to(DEVICE)
        dones_t = torch.from_numpy(dones).to(DEVICE)

        # Current Q-values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (from frozen target network)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + GAMMA * next_q * (1 - dones_t)

        # Clip rewards for stability
        target = target.clamp(-1, 1)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Sync target network
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path=MODEL_PATH):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["policy_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        print(f"Model loaded from {path}")


MAX_STEPS_PER_EPISODE = 10_000


def make_env(render=False):
    return gym.make(
        "ALE/Qbert-v5",
        render_mode="human" if render else None,
        frameskip=1,  # We handle frame skip ourselves for consistency
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )


def train():
    env = make_env(render=False)
    n_actions = env.action_space.n
    agent = QbertAgent(n_actions)
    stacker = FrameStacker()

    if MODEL_PATH.exists():
        agent.load()
        print("Resuming training from checkpoint...")

    best_reward = -float("inf")
    recent_rewards = deque(maxlen=20)

    print(f"Training on {DEVICE} | Actions: {env.unwrapped.get_action_meanings()}")
    print(f"Epsilon: {EPS_START} -> {EPS_END} over {EPS_DECAY} steps")
    print()

    for episode in range(TRAIN_EPISODES):
        obs, _ = env.reset()
        state = stacker.reset(obs)
        total_reward = 0
        losses = []

        while True:
            action = agent.select_action(state)

            # Frame skip: repeat action for N frames, accumulate reward
            reward = 0
            done = False
            for _ in range(FRAME_SKIP):
                obs, r, terminated, truncated, info = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            next_state = stacker.push(obs)
            # Clip reward to [-1, 1] for stable training
            clipped_reward = max(-1.0, min(1.0, reward))
            agent.memory.push(state, action, clipped_reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -agent.steps_done / EPS_DECAY
        )
        avg_loss = np.mean(losses) if losses else 0

        print(
            f"Ep {episode:4d} | "
            f"Reward: {total_reward:7.0f} | "
            f"Avg(20): {avg_reward:7.1f} | "
            f"Eps: {eps:.3f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Steps: {agent.steps_done}"
        )

        # Save best model
        if avg_reward > best_reward and episode >= 20:
            best_reward = avg_reward
            agent.save()

        # Periodic checkpoint
        if (episode + 1) % 50 == 0:
            agent.save()

    env.close()
    agent.save()
    print(f"\nTraining complete. Best avg reward: {best_reward:.1f}")


def play(random_mode=False):
    env = make_env(render=True)
    n_actions = env.action_space.n

    agent = None
    if not random_mode:
        agent = QbertAgent(n_actions)
        if MODEL_PATH.exists():
            agent.load()
        else:
            print("No trained model found. Run 'python qbert_agent.py train' first.")
            return

    stacker = FrameStacker()
    total_reward = 0

    obs, _ = env.reset()
    state = stacker.reset(obs)

    mode = "random" if random_mode else "trained"
    print(f"Playing Q*bert ({mode} mode). Close window to stop.")

    while True:
        if random_mode:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, eval_mode=True)

        reward = 0
        done = False
        for _ in range(FRAME_SKIP):
            obs, r, terminated, truncated, info = env.step(action)
            reward += r
            done = terminated or truncated
            if done:
                break

        state = stacker.push(obs)
        total_reward += reward

        if done:
            print(f"Game over! Score: {total_reward:.0f}")
            total_reward = 0
            obs, _ = env.reset()
            state = stacker.reset(obs)

    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        train()
    elif mode == "play":
        play(random_mode=False)
    elif mode == "random":
        play(random_mode=True)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python qbert_agent.py [train|play|random]")
