import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pybullet as p
from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

# Image processing
resize = T.Compose(
    [T.ToPILImage(), T.Resize((40, 40), interpolation=Image.BICUBIC), T.ToTensor()]
)


def get_screen(env, device):
    """
    Obtain a processed screen image (a torch tensor) from the environment.
    The raw observation (H x W x 3) is normalized, transposed (CHW), resized, and moved to device.

    Args:
        env: The environment
        device: The device to move the tensor to

    Returns:
        torch.Tensor: Processed screen image
    """
    if not hasattr(env, "_view_matrix"):
        raise AttributeError("Environment doesn't have camera view matrix set up!")

    screen = env._get_observation()  # raw observation (likely 84x84x3)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen).permute(2, 0, 1)  # convert to CHW
    screen = resize(screen).unsqueeze(0).to(device)  # add batch dimension
    return screen


def make_env(config, worker_id=0, test=False):
    """
    Creates and initializes an instance of the KukaDiverseObjectEnv.

    Args:
        config (dict): Configuration dictionary
        worker_id (int): Worker ID (0 renders GUI)
        test (bool): Whether this is a test environment

    Returns:
        env: The initialized environment
    """
    renders = config["env"]["renders"] if worker_id == 0 else False
    env = KukaDiverseObjectEnv(
        renders=renders,
        isDiscrete=config["env"]["isDiscrete"],
        removeHeightHack=config["env"]["removeHeightHack"],
        maxSteps=config["env"]["maxSteps"],
    )

    env.reset()

    # Set up camera
    setup_camera(env, config)

    # TODO:Define observation and action spaces
    env.observation_space = None
    env.action_space = None

    return env


def setup_camera(env, config):
    """
    Set up the camera view and projection matrices for the environment.
    This needs to be called after each reset.

    Args:
        env: The environment
        config (dict): Configuration dictionary
    """
    camera_config = config["camera"]

    env._view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_config["target_pos"],
        distance=camera_config["distance"],
        yaw=camera_config["yaw"],
        pitch=camera_config["pitch"],
        roll=0,
        upAxisIndex=2,
    )

    env._proj_matrix = p.computeProjectionMatrixFOV(
        fov=camera_config["fov"],
        aspect=camera_config["aspect"],
        nearVal=camera_config["nearVal"],
        farVal=camera_config["farVal"],
    )

    return env


def build_hidden_layer(input_dim, hidden_layers):
    """
    Build a sequence of fully connected layers.

    Args:
        input_dim (int): Input dimension
        hidden_layers (list): List of hidden layer sizes

    Returns:
        nn.ModuleList: List of linear layers
    """
    import torch.nn as nn

    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers) > 1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden


def soft_update(target, source, tau):
    """
    Soft-update target network parameters.

    Args:
        target: Target network
        source: Source network
        tau (float): Update rate
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
