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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_out_size function to calculate the output dimensions
        convw = conv2d_out_size(conv2d_out_size(conv2d_out_size(state_shape[0])))
        convh = conv2d_out_size(conv2d_out_size(conv2d_out_size(state_shape[1])))
        linear_input_size = convw * convh * 64

        # TODO: Create fully connected layers
        # Hint: Create a network with two hidden layers (e.g., 128 and 64 units)
        # and an output layer with action_dim units
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """Forward pass through the network."""
        # TODO: Implement the forward pass through the network
        # Hint: Process through conv layers with ReLU and batch norm,
        # flatten the output, then process through FC layers
        # The final output should use tanh to bound actions to [-1, 1]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        return x


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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_out_size function to calculate the output dimensions
        convw = conv2d_out_size(conv2d_out_size(conv2d_out_size(state_shape[0])))
        convh = conv2d_out_size(conv2d_out_size(conv2d_out_size(state_shape[1])))
        linear_input_size = convw * convh * 64

        # TODO: Create fully connected layers
        # Hint: Process state and action separately, then combine them
        # The state should go through one FC layer, the action through another,
        # then combine them and process through additional FC layers
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.fc_action = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

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
        
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = torch.flatten(state, start_dim=1)
        state = F.relu(self.fc1(state))

        action = F.relu(self.fc_action(action))
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
