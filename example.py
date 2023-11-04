import torch
from metnet3.model import MetNet

# Assuming the TopographicalEmbeddingLayer and MaxViT are defined elsewhere
# Initialize the MetNet model
met_net_model = MetNet(
    dim=64, n_channels=4, n_classes=1, 
)

# Example tensors for high-resolution and low-resolution inputs and coordinates
high_res_inputs = torch.randn(1, 793, 624, 624)
low_res_inputs = torch.randn(1, 64, 1248, 1248)
coords = torch.randn(1, 1, 624, 624)

# Forward pass through the MetNet model
output_predictions = met_net_model(high_res_inputs, low_res_inputs, coords)
print(f"Output Predictions Shape: {output_predictions.shape}")
