import torch
import torch.nn as nn
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine

print("=== MNC Optimization Diagnostics ===")
torch.manual_seed(42)

class ScaleDistances(torch.nn.Module):
    def __init__(self, shift, scale):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, x):
        return (x + self.shift) / self.scale

# 1. Instantiate network
model = nn.Sequential(
    MNCLinear(384, 128),
    ScaleDistances(22.0, 2.0),
    nn.Tanh(),
    MNCLinear(128, 10),
    ScaleDistances(90.0, 10.0)
)

engine = MESUEngine(model, lr=0.5, sigma_prior=0.1, alpha_decay=0.0)

# 2. Simulate standard single-shot ingestion pass
x = torch.randn(1, 384)
# Unit-normalize x to match sentence embeddings
x = x / x.norm(2)
target = torch.tensor([1])

# Capture baseline forward pass
logits = model(x)
print(f"Initial raw logits: {logits.data.numpy()[0]}")

# Compute loss and check backward trajectory
loss = -logits[0, 1]
loss.backward()

print("\nComputed Parameter Gradients:")
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"  {name:15} -> Grad Mean: {p.grad.abs().mean().item():.6f}")

# Execute update step
engine.step(loss.item())
engine.zero_grad()

# Check parameter drift post-step
print("\nPost-Update State Check:")
print(f"  Engine Variance Floor Check: {engine.variances['0.W'].min().item():.6f}")