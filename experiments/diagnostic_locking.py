import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class ScaleDistances(nn.Module):
    def __init__(self, shift, scale):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, x):
        return (x + self.shift) / self.scale

def autocalibrate_scale_distances(in_features, num_samples=1000):
    set_seed(42)
    x = torch.randn(num_samples, in_features)
    x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w = torch.randn(num_samples, in_features)
    w = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)
    neg_l1 = -torch.norm(x - w, p=1, dim=1)
    mean_val = neg_l1.mean().item()
    std_val = neg_l1.std().item()
    shift = -mean_val
    scale = 2.0 * std_val if std_val > 1e-5 else 1.0
    return shift, scale

SHIFT_384, SCALE_384 = autocalibrate_scale_distances(384)

def test_locking(width, use_rescaled):
    set_seed(42)
    shift_w, scale_w = autocalibrate_scale_distances(width)
    
    # Simple model with one hidden layer
    # We use a standard input size and a wide/narrow layer
    model = nn.Sequential(
        MNCLinear(384, width),
        ScaleDistances(SHIFT_384, SCALE_384),
        nn.Tanh(),
        MNCLinear(width, 10),
        ScaleDistances(shift_w, scale_w)
    )
    
    # Engine setup
    engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.0)
    
    # We will hack the engine's step function temporarily if we want to test unscaled
    original_step = engine.step
    
    if not use_rescaled:
        # Hack the step function to use unscaled gradient for locking
        @torch.no_grad()
        def unscaled_step(self, current_loss=None):
            g = 0.05
            for name, param in self.model.named_parameters():
                if param.grad is None: continue
                raw_grad = param.grad.data
                grad_norm = raw_grad.norm(2)
                if grad_norm > 1e-5:
                    scale = (param.numel() ** 0.5) / (grad_norm + 1e-4)
                    raw_grad = raw_grad * scale
                var = self.variances[name]
                effective_lr = self.lr * var 
                param.data.sub_(effective_lr * raw_grad)
                # Unscaled gradient for locking:
                var.sub_(var * torch.clamp(param.grad.data.abs() * 0.2, max=0.25))
                var.add_(self.alpha_decay * ((self.sigma_prior**2) - var))
                var.clamp_(min=1e-4, max=self.sigma_prior ** 2)
        import types
        engine.step = types.MethodType(unscaled_step, engine)
        
    loss_fn = nn.CrossEntropyLoss()
    
    x = torch.randn(1, 384)
    x = x / x.norm(p=2, dim=1, keepdim=True)  # Unit norm input
    
    target = torch.tensor([3])
    
    variances_layer0 = []
    
    rescaled_mean_abs = 0.0
    
    for _ in range(15):
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        
        # Amplify gradients to ensure they are realistic and trigger rescaling
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data *= 1000.0  # Make them realistically large
        
        # Record raw grad mean abs just for info
        grad_mean_abs = model[0].W.grad.data.abs().mean().item()
        grad_norm_val = model[0].W.grad.data.norm(2).item()
        
        engine.step()
        var_mean = engine.variances['0.W'].mean().item()
        variances_layer0.append(var_mean)
        
        # Calculate rescaled mean abs inside step implicitly
        raw_grad = model[0].W.grad.data
        if raw_grad.norm(2) > 1e-5:
            scale = (raw_grad.numel() ** 0.5) / (raw_grad.norm(2) + 1e-4)
            rescaled_mean_abs = (raw_grad * scale).abs().mean().item()
        
    return variances_layer0, grad_mean_abs, grad_norm_val, rescaled_mean_abs

print("--- Diagnostic: Variance Locking Speed vs Width ---")
print("We compare the variance drop over 15 steps for Width 32 vs Width 256.")

print("\n1. Using UNSCALED parameter gradient (param.grad.data.abs()):")
var32_unscaled, grad32, norm32, resc32_un = test_locking(32, use_rescaled=False)
var256_unscaled, grad256, norm256, resc256_un = test_locking(256, use_rescaled=False)

print(f"Width  32 | Raw Grad Norm: {norm32:.2e} | Mean Abs: {grad32:.2e} | Variances: {[f'{v:.5f}' for v in var32_unscaled]}")
print(f"Width 256 | Raw Grad Norm: {norm256:.2e} | Mean Abs: {grad256:.2e} | Variances: {[f'{v:.5f}' for v in var256_unscaled]}")

print("\n2. Using RESCALED gradient (raw_grad.abs()):")
var32_rescaled, _, _, resc32 = test_locking(32, use_rescaled=True)
var256_rescaled, _, _, resc256 = test_locking(256, use_rescaled=True)

print(f"Width  32 | Rescaled Mean Abs: {resc32:.4f} | Variances: {[f'{v:.5f}' for v in var32_rescaled]}")
print(f"Width 256 | Rescaled Mean Abs: {resc256:.4f} | Variances: {[f'{v:.5f}' for v in var256_rescaled]}")
