import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize a replay buffer for DDPG.

        Args:
            capacity (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Size of the batch to sample
            device: Device to move tensors to

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.cat([exp[0] for exp in experiences])
        actions = torch.cat(
            [
                torch.tensor(exp[1], dtype=torch.float32).unsqueeze(0)
                for exp in experiences
            ]
        ).to(device)
        rewards = (
            torch.tensor([exp[2] for exp in experiences], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        next_states = torch.cat([exp[3] for exp in experiences])
        dones = (
            torch.tensor([exp[4] for exp in experiences], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
