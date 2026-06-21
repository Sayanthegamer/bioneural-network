import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

FACTS = [
    "The blue folder is in the third drawer.",
    "The access code for the main server is 7734.",
    "Sarah's strategy meeting was moved to Tuesday at 3 PM.",
    "The backup generator requires unleaded fuel.",
    "Sector 4 security cameras undergo maintenance at midnight."
]

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

def test_locking_real_data(width, use_rescaled):
    set_seed(42)
    shift_w, scale_w = autocalibrate_scale_distances(width)
    
    model = nn.Sequential(
        MNCLinear(384, width),
        ScaleDistances(SHIFT_384, SCALE_384),
        nn.Tanh(),
        MNCLinear(width, 10),
        ScaleDistances(shift_w, scale_w)
    )
    
    engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.0)
    
    if not use_rescaled:
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
    pipeline = JournalPipeline()
    
    variances_layer0 = []
    grad_means = []
    grad_norms = []
    resc_means = []
    
    for i, text in enumerate(FACTS):
        # 1 step per fact to avoid accumulating changes too fast, but we can do a few steps
        vec = pipeline.embed_sentence(text)
        target = torch.tensor([i])
        
        model.zero_grad()
        logits = model(vec)
        loss = loss_fn(logits, target)
        loss.backward()
        
        grad_means.append(model[0].W.grad.data.abs().mean().item())
        grad_norms.append(model[0].W.grad.data.norm(2).item())
        
        engine.step()
        
        variances_layer0.append(engine.variances['0.W'].mean().item())
        
        raw_grad = model[0].W.grad.data
        if raw_grad.norm(2) > 1e-5:
            scale = (raw_grad.numel() ** 0.5) / (raw_grad.norm(2) + 1e-4)
            resc_means.append((raw_grad * scale).abs().mean().item())
        else:
            resc_means.append(0.0)
            
    return variances_layer0, np.mean(grad_means), np.mean(grad_norms), np.mean(resc_means)

print("--- Diagnostic V2: Real Data Locking Speed vs Width ---")

print("\n1. Using UNSCALED parameter gradient (param.grad.data.abs()):")
var32_un, grad32_un, norm32_un, resc32_un = test_locking_real_data(32, use_rescaled=False)
var256_un, grad256_un, norm256_un, resc256_un = test_locking_real_data(256, use_rescaled=False)

print(f"Width  32 | Avg Norm: {norm32_un:.2e} | Avg Mean Abs: {grad32_un:.2e} | End Var: {var32_un[-1]:.6f}")
print(f"Width 256 | Avg Norm: {norm256_un:.2e} | Avg Mean Abs: {grad256_un:.2e} | End Var: {var256_un[-1]:.6f}")

print("\n2. Using RESCALED gradient (raw_grad.abs()):")
var32_res, grad32_res, norm32_res, resc32_res = test_locking_real_data(32, use_rescaled=True)
var256_res, grad256_res, norm256_res, resc256_res = test_locking_real_data(256, use_rescaled=True)

print(f"Width  32 | Avg Resc Mean Abs: {resc32_res:.4f} | End Var: {var32_res[-1]:.6f}")
print(f"Width 256 | Avg Resc Mean Abs: {resc256_res:.4f} | End Var: {var256_res[-1]:.6f}")
