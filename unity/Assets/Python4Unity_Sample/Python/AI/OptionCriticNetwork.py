import torch
import torch.nn as nn
from typing import Tuple

class OptionCriticNetwork(nn.Module):
    """
    Neural network architecture for the Option-Critic agent.
    Outputs logits for option type, joint target (x,y) coordinates,
    termination probability, and state value.
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, input_channels: int, grid_size_x: int, grid_size_y: int, num_action_types: int, hidden_size: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_action_types = num_action_types

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Depthwise separable convolution block 1
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels), # Depthwise
            nn.Conv2d(input_channels, 32, 1), # Pointwise
            nn.GELU(),
            nn.MaxPool2d(2), # Reduces spatial dimensions by 2x

            # Depthwise separable convolution block 2
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.GELU(),
            nn.MaxPool2d(2), # Reduces spatial dimensions by 2x

            # Depthwise separable convolution block 3
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1) # Ensures output is 1x1 regardless of input size, resulting in (batch_size, 128, 1, 1)
        )

        # Fully connected layers to process features before heads
        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),
            nn.GELU()
        )

        # Output heads for Option-Critic
        self.option_type_head = nn.Linear(64, num_action_types) # Logits for action type (water, firebreak etc.)

        # This head now outputs logits for all possible (x,y) combinations for each action type.
        self.target_xy_head = nn.Linear(64, num_action_types * grid_size_x * grid_size_y)

        self.termination_head = nn.Linear(64, 1)              # Logit for termination probability (beta)
        self.value_head = nn.Linear(64, 1)                    # State value V(s)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes weights for linear layers using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observation tensor (batch_size, channels, H, W).

        Returns:
            Tuple[torch.Tensor, ...]:
                - option_type_logits (batch_size, num_action_types)
                - target_xy_logits (batch_size, num_action_types, grid_size_x, grid_size_y)
                - termination_logit (batch_size, 1)
                - value (batch_size, 1)
        """
        conv_out = self.conv_layers(x)

        # Flatten from (batch_size, 128, 1, 1) to (batch_size, 128)
        flat_out = conv_out.view(conv_out.size(0), -1)
        features = self.fc_layers(flat_out)

        option_type_logits = self.option_type_head(features)

        # Reshape to (batch_size, num_action_types, grid_size_x, grid_size_y)
        target_xy_logits = self.target_xy_head(features).view(
            x.size(0), self.num_action_types, self.grid_size_x, self.grid_size_y
        )

        termination_logit = self.termination_head(features)
        value = self.value_head(features)

        # The return signature has changed from 5 to 4 elements.
        return option_type_logits, target_xy_logits, termination_logit, value