import torch
from lib.ddpg.agent import DDPGAgent
from helpers.config import load_config


def train_ddpg():
    """Train the DDPG agent."""
    # Load configuration
    config = load_config("ddpg")
    device = torch.device(config["device"])

    # Create and train agent
    agent = DDPGAgent(config, device)
    agent.train()
