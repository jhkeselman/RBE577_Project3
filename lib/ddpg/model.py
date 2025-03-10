import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_out_size(size, kernel_size=5, stride=2):
    """Calculate output size of a convolutional layer."""
    return (size - (kernel_size - 1) - 1) // stride + 1


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim):
        """
        Actor network for DDPG.

        Args:
            state_shape (tuple): Height and width of the processed image
            action_dim (int): Dimensionality of the action space
        """
        super(Actor, self).__init__()
        
        # TODO: Create convolutional layers for processing image input
        # Hint: Use Conv2d layers with appropriate kernel size, stride, and BatchNorm
        self.conv1 = None  # Replace with your implementation
        self.bn1 = None    # Replace with your implementation
        self.conv2 = None  # Replace with your implementation
        self.bn2 = None    # Replace with your implementation
        self.conv3 = None  # Replace with your implementation
        self.bn3 = None    # Replace with your implementation

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_out_size function to calculate the output dimensions
        convw = None  # Replace with your implementation
        convh = None  # Replace with your implementation
        linear_input_size = None  # Replace with your implementation

        # TODO: Create fully connected layers
        # Hint: Create a network with two hidden layers (e.g., 128 and 64 units)
        # and an output layer with action_dim units
        self.fc1 = None  # Replace with your implementation
        self.fc2 = None  # Replace with your implementation
        self.fc3 = None  # Replace with your implementation

    def forward(self, x):
        """Forward pass through the network."""
        # TODO: Implement the forward pass through the network
        # Hint: Process through conv layers with ReLU and batch norm,
        # flatten the output, then process through FC layers
        # The final output should use tanh to bound actions to [-1, 1]
        
        # Your implementation here
        
        return None  # Replace with your implementation


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        """
        Critic network for DDPG.

        Args:
            state_shape (tuple): Height and width of the processed image
            action_dim (int): Dimensionality of the action space
        """
        super(Critic, self).__init__()
        
        # TODO: Create convolutional layers for processing image input
        # Hint: Use Conv2d layers with appropriate kernel size, stride, and BatchNorm
        self.conv1 = None  # Replace with your implementation
        self.bn1 = None    # Replace with your implementation
        self.conv2 = None  # Replace with your implementation
        self.bn2 = None    # Replace with your implementation
        self.conv3 = None  # Replace with your implementation
        self.bn3 = None    # Replace with your implementation

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_out_size function to calculate the output dimensions
        convw = None  # Replace with your implementation
        convh = None  # Replace with your implementation
        linear_input_size = None  # Replace with your implementation

        # TODO: Create fully connected layers
        # Hint: Process state and action separately, then combine them
        # The state should go through one FC layer, the action through another,
        # then combine them and process through additional FC layers
        self.fc1 = None  # Replace with your implementation
        self.fc_action = None  # Replace with your implementation
        self.fc2 = None  # Replace with your implementation
        self.fc3 = None  # Replace with your implementation

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q-value
        """
        # TODO: Implement the forward pass through the network
        # Hint: Process state through conv layers and first FC layer
        # Process action through fc_action
        # Concatenate state and action features, then process through remaining FC layers
        
        # Your implementation here
        
        return None  # Replace with your implementation
