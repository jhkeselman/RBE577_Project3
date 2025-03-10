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
        self.init_type = init_type
        torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        # TODO: Create convolutional layers for processing image input
        # Hint: Use Conv2d layers with appropriate kernel size, stride, and BatchNorm
        self.conv1 = None  # Replace with your implementation
        self.bn1 = None    # Replace with your implementation
        self.conv2 = None  # Replace with your implementation
        self.bn2 = None    # Replace with your implementation
        self.conv3 = None  # Replace with your implementation
        self.bn3 = None    # Replace with your implementation

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # TODO: Calculate the size of the flattened features after convolution
        # Hint: Use the conv2d_size_out function to calculate the output dimensions
        convw = None  # Replace with your implementation
        convh = None  # Replace with your implementation
        linear_input_size = None  # Replace with your implementation

        # TODO: Create shared layers using the build_hidden_layer helper function
        # Hint: The input dimension should be the flattened size from convolutions
        self.shared_layers = None  # Replace with your implementation

        # TODO: Create critic network layers
        # Hint: Use build_hidden_layer for hidden layers and a final Linear layer for the value output
        # If critic_hidden_layers is empty, connect directly from shared layers to output
        self.critic_hidden = None  # Replace with your implementation
        self.critic = None  # Replace with your implementation

        # TODO: Create actor network layers
        # Hint: Use build_hidden_layer for hidden layers and a final Linear layer for the action output
        # If actor_hidden_layers is empty, connect directly from shared layers to output
        self.actor_hidden = None  # Replace with your implementation
        self.actor = None  # Replace with your implementation

        self.tanh = nn.Tanh()

        # TODO: Initialize network weights if init_type is provided
        # Hint: Apply the _initialize method to all layers
        # Your implementation here

    def _initialize(self, n):
        """Initialize network weights based on the specified initialization type."""
        if isinstance(n, nn.Linear):
            # TODO: Implement different weight initialization methods based on self.init_type
            # Hint: Use nn.init functions like xavier_uniform_, kaiming_normal_, etc.
            pass  # Replace with your implementation

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
        x = None  # Replace with your implementation

        # TODO: Flatten the output and pass through shared layers
        # Hint: Use view to flatten and apply_multi_layer for the shared layers
        x = None  # Replace with your implementation

        # TODO: Compute value (critic output)
        # Hint: Process through critic_hidden if it exists, then through critic layer
        value = None  # Replace with your implementation

        # TODO: Compute action mean (actor output)
        # Hint: Process through actor_hidden if it exists, then through actor layer with tanh
        action_mean = None  # Replace with your implementation

        return action_mean, value
