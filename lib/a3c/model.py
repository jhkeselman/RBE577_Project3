import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.utils import build_hidden_layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        shared_layers,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
        init_type=None,
    ):
        """
        Neural network that outputs both the policy (actor) and value (critic) estimates.

        Args:
            state_size (tuple): Height and width of the processed image
            action_size (int): Dimensionality of the action space
            shared_layers (list): List of shared layer sizes
            critic_hidden_layers (list): List of critic hidden layer sizes
            actor_hidden_layers (list): List of actor hidden layer sizes
            seed (int): Random seed
            init_type (str): Weight initialization type
        """
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.shared_layers = shared_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.actor_hidden_layers = actor_hidden_layers
        self.seed = seed
        self.init_type = init_type
        torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        # TODO: Create convolutional layers for processing image input
        # Hint: Use Conv2d layers with appropriate kernel size, stride, and BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=8, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_size_out function to calculate the output dimensions
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convw * convh * 64

        # TODO: Create shared layers using the build_hidden_layer helper function
        # Hint: The input dimension should be the flattened size from convolutions
        self.shared = build_hidden_layer(linear_input_size, shared_layers) if shared_layers else nn.Identity()

        # TODO: Create critic network layers
        # Hint: Use build_hidden_layer for hidden layers and a final Linear layer for the value output
        # If critic_hidden_layers is empty, connect directly from shared layers to output
        self.critic_hidden = build_hidden_layer(shared_layers[-1], critic_hidden_layers) if critic_hidden_layers else nn.Identity()
        self.critic = nn.Linear(critic_hidden_layers[-1] if critic_hidden_layers else (shared_layers[-1] if shared_layers else linear_input_size), 1)

        # TODO: Create actor network layers
        # Hint: Use build_hidden_layer for hidden layers and a final Linear layer for the action output
        # If actor_hidden_layers is empty, connect directly from shared layers to output
        self.actor_hidden = build_hidden_layer(shared_layers[-1], actor_hidden_layers) if actor_hidden_layers else nn.Identity()
        self.actor = nn.Linear(actor_hidden_layers[-1] if actor_hidden_layers else (shared_layers[-1] if shared_layers else linear_input_size), action_size)

        self.tanh = nn.Tanh()

        # TODO: Initialize network weights if init_type is provided
        # Hint: Apply the _initialize method to all layers
        if self.init_type:
            self.apply(self._initialize)

    def _initialize(self, n):
        """Initialize network weights based on the specified initialization type."""
        if isinstance(n, nn.Linear):
            # TODO: Implement different weight initialization methods based on self.init_type
            # Hint: Use nn.init functions like xavier_uniform_, kaiming_normal_, etc.
            if self.init_type == "xavier":
                nn.init.xavier_uniform_(n.weight)
            elif self.init_type == "kaiming-normal":
                nn.init.kaiming_normal_(n.weight)
            elif self.init_type == "orthogonal":
                nn.init.orthogonal_(n.weight)
            elif n.bias is not None:
                nn.init.constant_(n.bias, 0)

    def forward(self, state):
        """
        Forward pass mapping state -> (action_mean, value).

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            tuple: (action_mean, value)
        """
        # TODO: Implement the forward pass through the network
        # Hint: Process the state through conv layers, then shared layers,
        # then split into actor and critic paths
        
        def apply_multi_layer(layers, x, f=F.leaky_relu):
            # Helper function to apply multiple layers with activation
            for layer in layers:
                x = f(layer(x))
            return x

        # TODO: Process through convolutional layers
        # Hint: Use ReLU activation with batch normalization
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # TODO: Flatten the output and pass through shared layers
        # Hint: Use view to flatten and apply_multi_layer for the shared layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))

        # TODO: Compute value (critic output)
        # Hint: Process through critic_hidden if it exists, then through critic layer
        value = apply_multi_layer(self.critic_hidden, x) if self.critic_hidden else x
        value = self.critic(value)

        # TODO: Compute action mean (actor output)
        # Hint: Process through actor_hidden if it exists, then through actor layer with tanh
        action_mean = apply_multi_layer(self.actor_hidden, x) if self.actor_hidden else x
        action_mean = torch.tanh(self.actor(action_mean))

        return action_mean, value
