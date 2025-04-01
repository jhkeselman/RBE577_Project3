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
        action_dim = int(np.prod([1,3])) 
        
        self.actor = Actor(state_size, action_dim).to(self.device)
        self.critic = Critic(state_size, action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        # TODO: Initialize optimizers
        # Hint: Use Adam optimizer with learning rates from config
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["hyperparameters"]["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["hyperparameters"]["critic_lr"])

        # TODO: Initialize replay buffer
        # Hint: Use ReplayBuffer class with buffer_size from config
        self.replay_buffer = ReplayBuffer(config["hyperparameters"]["buffer_size"])

        # TODO: Initialize metrics tracker
        # Hint: Use MetricsTracker class
        self.metrics = MetricsTracker()

        # TODO: Extract hyperparameters from config
        # Hint: Get gamma, tau, batch_size, exploration_noise, etc.
        self.gamma = config["hyperparameters"]["gamma"]
        self.tau = config["hyperparameters"]["tau"]
        self.batch_size = config["hyperparameters"]["batch_size"]
        self.exploration_noise = config["hyperparameters"]["exploration_noise"]

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
        
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().detach().numpy()[0]
        if add_noise:
            action += self.exploration_noise * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

    def update(self):
        """Update the networks using a batch from the replay buffer."""
        # TODO: Check if replay buffer has enough samples
        # Hint: Return early if buffer size is less than batch_size
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        # TODO: Sample a batch from the replay buffer
        # Hint: Use replay_buffer.sample() to get states, actions, rewards, next_states, dones
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # TODO: Update critic
        # Hint: Compute target Q-values using target networks and the Bellman equation
        # Compute current Q-values and the critic loss
        next_actions = self.actor_target(next_states)
        target_Q_values = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)

        current_Q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q_values, target_Q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # TODO: Update actor
        # Hint: Compute actor loss as negative of the expected Q-value
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # TODO: Update target networks
        # Hint: Use soft_update to update target networks
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return critic_loss.item(), actor_loss.item()
    
    def train(self):
        """Train the agent."""
        # TODO: Extract training parameters from config
        # Hint: Get max_episodes, max_steps, log_interval, save_interval
        max_episodes = self.config["hyperparameters"]["max_episodes"]
        max_steps = self.config["hyperparameters"]["max_steps"]
        log_interval = self.config["logging"]["log_interval"]
        save_interval = self.config["logging"]["save_interval"]

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
            state = self.env.reset()
            setup_camera(self.env, self.config)

            episode_reward = 0
            episode_critic_loss = 0
            episode_actor_loss = 0

            # TODO: Implement episode loop
            # Hint: For each step, select action, take step in environment,
            # store transition, and update networks
            for step in range(max_steps):
                # Your implementation here
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                critic_loss, actor_loss = self.update()
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                
                if done:
                    break

            # TODO: Track metrics
            # Hint: Add episode reward, length, and loss to metrics tracker
            self.metrics.add_episode_reward(episode_reward)
            self.metrics.add_episode_length(step)
            self.metrics.add_loss("critic", episode_critic_loss)
            self.metrics.add_loss("actor", episode_actor_loss)

            # TODO: Implement logging
            # Hint: Log metrics periodically and save model checkpoints
            self.logger.log_episode(
                episode,
                episode_reward,
                step,
                episode_critic_loss,
                episode_actor_loss,
            )

            if episode % save_interval == 0:
                self.logger.save_model(episode, {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()})

        # TODO: Save final model
        # Hint: Save actor, critic, and optimizer states

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
        
        eval_episodes = self.config["logging"]["eval_episodes"]
        total_reward = 0.0

        for _ in range(eval_episodes):
            state = self.env.reset()
            episode_reward = 0.0

            while True:
                action = self.select_action(state, add_noise=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_reward += episode_reward
        
        return total_reward / eval_episodes
