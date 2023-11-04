import torch
from metnet3.model import MetNet
from torch.nn.functional import affine_grid

# Initialize the MetNet model with the specified dimensions and classes
met_net_model = MetNet(
    dim=64, n_channels=857, n_classes=1
)

# Example tensors for high-resolution and low-resolution inputs
high_res_inputs = torch.randn(1, 793, 624, 624)
low_res_inputs = torch.randn(1, 64, 1248, 1248)

# Generate a grid of (x, y) coordinates for grid_sample
batch_size, _, height, width = high_res_inputs.size()
theta = torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1)  # Identity transformation matrix
coords = affine_grid(theta, [batch_size, 1, height, width])  # Generates a normalized grid of coordinates
print(f"Coords Shape: {coords.shape}")

# Forward pass through the MetNet model using the correctly formatted 'coords'
output_predictions = met_net_model(high_res_inputs, low_res_inputs, coords)
print(f"Output Predictions Shape: {output_predictions.shape}")
