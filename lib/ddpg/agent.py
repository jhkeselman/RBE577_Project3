import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm

from lib.ddpg.model import Actor, Critic
from lib.ddpg.replay_buffer import ReplayBuffer
from helpers.utils import get_screen, make_env, setup_camera, soft_update
from helpers.metrics import MetricsTracker
from helpers.logger import Logger


class DDPGAgent:
    def __init__(self, config, device):
        """
        Initialize DDPG agent.

        Args:
            config (dict): Configuration dictionary
            device: Device to use for tensor operations
        """
        self.config = config
        self.device = device
        self.logger = Logger(config, "DDPG")

        # TODO: Create environment
        # Hint: Use make_env function with config, worker_id=0, and test=False
        self.env = make_env(config, worker_id=0, test=False)

        # TODO: Initialize networks (actor, critic, and their targets)
        # Hint: Create Actor and Critic networks and their target networks
        # Target networks should be copies of the main networks
        state_size = config["network"]["state_size"]
        action_dim = config["network"]["action_dim"]
        
        self.actor = Actor(state_size, action_dim).to(self.device)
        self.critic = Critic(state_size, action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        # TODO: Initialize optimizers
        # Hint: Use Adam optimizer with learning rates from config
        self.actor_optimizer = None  # Replace with your implementation
        self.critic_optimizer = None  # Replace with your implementation

        # TODO: Initialize replay buffer
        # Hint: Use ReplayBuffer class with buffer_size from config
        self.replay_buffer = None  # Replace with your implementation

        # TODO: Initialize metrics tracker
        # Hint: Use MetricsTracker class
        self.metrics = None  # Replace with your implementation

        # TODO: Extract hyperparameters from config
        # Hint: Get gamma, tau, batch_size, exploration_noise, etc.
        self.gamma = None  # Replace with your implementation
        self.tau = None  # Replace with your implementation
        self.batch_size = None  # Replace with your implementation
        self.exploration_noise = None  # Replace with your implementation

        # Create model directory
        os.makedirs(config["logging"]["model_dir"], exist_ok=True)

        # Best reward for model saving
        self.best_eval_reward = float("-inf")

    def select_action(self, state, add_noise=True):
        """
        Select an action using the actor network.

        Args:
            state: Current state
            add_noise (bool): Whether to add exploration noise

        Returns:
            numpy.ndarray: Selected action
        """
        # TODO: Implement action selection
        # Hint: Get action from actor network, add exploration noise if needed,
        # and clip to action space bounds
        
        # Your implementation here
        
        return None  # Replace with your implementation

    def update(self):
        """Update the networks using a batch from the replay buffer."""
        # TODO: Check if replay buffer has enough samples
        # Hint: Return early if buffer size is less than batch_size
        # Your implementation here

        # TODO: Sample a batch from the replay buffer
        # Hint: Use replay_buffer.sample() to get states, actions, rewards, next_states, dones
        # Your implementation here

        # TODO: Update critic
        # Hint: Compute target Q-values using target networks and the Bellman equation
        # Compute current Q-values and the critic loss
        # Your implementation here

        # TODO: Update actor
        # Hint: Compute actor loss as negative of the expected Q-value
        # Your implementation here

        # TODO: Update target networks
        # Hint: Use soft_update to update target networks
        # Your implementation here

        return 0.0, 0.0  # Replace with your implementation

    def train(self):
        """Train the agent."""
        # TODO: Extract training parameters from config
        # Hint: Get max_episodes, max_steps, log_interval, save_interval
        max_episodes = None  # Replace with your implementation
        max_steps = None  # Replace with your implementation
        log_interval = None  # Replace with your implementation
        save_interval = None  # Replace with your implementation

        self.logger.project_logger.info(
            f"Starting DDPG training for Kuka pick and place task..."
        )
        self.logger.project_logger.info(f"Device: {self.device}")

        best_reward = float("-inf")

        # TODO: Implement training loop
        # Hint: For each episode, reset environment, collect experience,
        # update networks, and log progress
        for episode in range(1, max_episodes + 1):
            # TODO: Reset environment and get initial state
            # Hint: Use env.reset(), setup_camera(), and get_screen()
            # Your implementation here

            episode_reward = 0
            episode_critic_loss = 0
            episode_actor_loss = 0

            # TODO: Implement episode loop
            # Hint: For each step, select action, take step in environment,
            # store transition, and update networks
            for step in range(max_steps):
                # Your implementation here
                pass

            # TODO: Track metrics
            # Hint: Add episode reward, length, and loss to metrics tracker
            # Your implementation here

            # TODO: Implement logging
            # Hint: Log metrics periodically and save model checkpoints
            # Your implementation here

        # TODO: Save final model
        # Hint: Save actor, critic, and optimizer states
        # Your implementation here

        self.logger.close()
        self.env.close()

    def evaluate(self):
        """
        Evaluate the agent's performance without exploration noise.

        Returns:
            float: Average reward over evaluation episodes
        """
        # TODO: Implement evaluation
        # Hint: Run several episodes without exploration noise and return average reward
        
        # Your implementation here
        
        return 0.0  # Replace with your implementation
