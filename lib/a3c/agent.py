import torch
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
import numpy as np
import os
import time
from datetime import timedelta

from lib.a3c.model import ActorCritic
from helpers.utils import get_screen, make_env, setup_camera
from helpers.metrics import MetricsTracker
from helpers.logger import Logger


def worker_process(
    worker_id,
    global_net,
    optimizer,
    global_ep,
    max_episodes,
    lock,
    config,
    device,
    log_dir=None,
    model_dir=None,
):
    """
    Worker function for A3C algorithm.

    Args:
        worker_id (int): Worker ID
        global_net (ActorCritic): Global network
        optimizer (torch.optim.Optimizer): Optimizer
        global_ep (mp.Value): Global episode counter
        max_episodes (int): Maximum number of episodes
        lock (mp.Lock): Lock for synchronization
        config (dict): Configuration dictionary
        device (torch.device): Device to use
        log_dir (str): Directory for logs (only for worker 0)
        model_dir (str): Directory for models (only for worker 0)
    """
    # TODO: Create environment
    # Hint: Use make_env function with config, worker_id, and test=False
    env = make_env(config=config, worker_id=worker_id, test=False)

    # TODO: Create logger if this is worker 0
    # Hint: Only worker 0 should have a logger
    logger = Logger(config, "a3c") if worker_id == 0 else None

    # TODO: Create local network
    # Hint: Create a copy of the ActorCritic network with the same parameters as global_net
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    local_net = ActorCritic(config["network"]["state_size"], action_dim).to(device)
    # TODO: Create metrics tracker
    # Hint: Use MetricsTracker class
    metrics = MetricsTracker()

    # TODO: Extract hyperparameters from config
    # Hint: Get gamma, t_max, entropy_coef, value_loss_coef, grad_clip, etc.
    gamma = config["hyperparameters"]["gamma"]
    t_max = config["hyperparameters"]["t_max"]
    entropy_coef = config["hyperparameters"]["entropy_coef"]
    value_loss_coef = config["hyperparameters"]["value_loss_coef"]
    grad_clip = config["hyperparameters"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    # Training loop
    while True:
        # TODO: Sync local network with global network
        # Hint: Copy the state_dict from global_net to local_net
        local_net.load_state_dict(global_net.state_dict())

        # TODO: Reset gradients
        # Hint: Zero out the gradients in the optimizer
        optimizer.zero_grad()

        # TODO: Initialize episode variables
        # Hint: Create empty lists for log_probs, values, rewards, entropies
        log_probs = []
        values = []
        rewards = []
        entropies = []

        # TODO: Reset environment and get initial state
        # Hint: Use env.reset(), setup_camera(), and get_screen()
        state = env.reset()
        setup_camera(env)
        state = get_screen(env).to(device)

        episode_reward = 0
        done = False

        # TODO: Collect experience for t_max steps or until done
        # Hint: Run the environment for t_max steps, collecting experience
        for t in range(t_max):
            # TODO: Get action and value from local network
            # Hint: Pass state through local_net and process the outputs
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            policy, value = local_net(state_tensor)
            action_dist = torch.distributions.Categorical(policy)

            # TODO: Sample action from the policy distribution
            # Hint: Use multinomial sampling from the action probabilities
            action = action_dist.sample()

            # TODO: Convert discrete action to continuous action space
            # Hint: Map discrete actions to continuous actions for the Kuka environment
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            # TODO: Take action in environment and observe next state and reward
            # Hint: Use env.step() and get_screen()
            next_state, reward, done, _ = env.step(action.item())

            # TODO: Store experience
            # Hint: Append log_prob, value, reward, and entropy to their respective lists
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
        
            # TODO: Update state and check if episode is done
            episode_reward += reward
            state = next_state

            if done:
                break

        # TODO: Bootstrap value for incomplete episode
        # Hint: If episode is not done, estimate the value of the last state
        R = 0 if done else local_net(torch.FloatTensor(state).to(device).unsqueeze(0))[1].detach()

        # TODO: Calculate returns and advantages
        # Hint: Compute discounted returns and advantages
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # TODO: Convert lists to tensors
        # Hint: Use torch.cat to concatenate tensors
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)

        # TODO: Calculate losses (actor, critic, entropy)
        # Hint: Compute actor loss using log probs and advantages
        # Compute critic loss using returns and values
        # Compute entropy loss using entropies'
        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        # TODO: Calculate total loss
        # Hint: Combine actor, critic, and entropy losses with their coefficients
        total_loss = actor_loss + value_loss_coef * critic_loss + entropy_coef * entropy_loss

        # TODO: Calculate gradients and update global network
        # Hint: Call backward() on total_loss, clip gradients, and update global parameters
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), grad_clip)
        optimizer.step()

        # TODO: Update metrics
        # Hint: Add episode reward and loss to metrics tracker
        metrics.add_episode_reward(episode_reward)
        metrics.add_loss(total_loss.item())

        # TODO: Update episode counter
        # Hint: Use lock to safely increment global_ep
        with lock:
            global_ep.value += 1

        # TODO: Implement logging for worker 0
        # Hint: Log metrics periodically and save model checkpoints
        if logger and global_ep.value % log_interval == 0:
            logger.log_episode(global_ep.value, episode_reward)
            logger.save_model(global_ep.value, {"actor_critic": global_net.state_dict()})

        # TODO: Check if training is complete
        # Hint: Break the loop if global_ep >= max_episodes
        if global_ep.value >= max_episodes:
            break

    env.close()
