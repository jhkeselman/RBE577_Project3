import os
import yaml
import torch


def load_config(algorithm):
    """
    Load configuration from YAML files.

    Args:
        algorithm (str): Algorithm name ('a3c' or 'ddpg')

    Returns:
        dict: Combined configuration from common and algorithm-specific files
    """
    # Load common config
    with open(os.path.join("config", "common.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Load algorithm-specific config
    with open(os.path.join("config", f"{algorithm.lower()}.yaml"), "r") as f:
        algo_config = yaml.safe_load(f)

    # Merge configs
    config.update(algo_config)

    # Set device
    if config["device"] == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"
        print("CUDA not available, using CPU instead.")

    return config
