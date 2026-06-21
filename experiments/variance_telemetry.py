import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

SHIFT0, SCALE0 = autocalibrate_scale_distances(384)
SHIFT1, SCALE1 = autocalibrate_scale_distances(32)

FACTS = [
    {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "answer": "third drawer", "label": 0},
    {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "answer": "7734", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "answer": "Tuesday at 3 PM", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "answer": "unleaded fuel", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "answer": "midnight", "label": 4}
]

INTERFERENCE = [
    "The red folder is resting on the top desk.",
    "Someone left a coffee mug in the breakroom.",
    "The access code for the guest wifi is 9912.",
    "It is raining heavily outside today."
]

def run_variance_telemetry_sweep(seed=42):
    set_seed(seed)
    pipeline = JournalPipeline()
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'variance_telemetry.csv')
    
    headers = ["alpha_decay", "step", "phase", "var_mean", "var_median", "var_min", "var_max"]
    telemetry_rows = []
    
    alpha_values = [0.001, 0.01, 0.1]
    
    for alpha in alpha_values:
        print(f"\n[*] Running telemetry for alpha_decay = {alpha}")
        
        model = nn.Sequential(
            MNCLinear(384, 32),
            ScaleDistances(SHIFT0, SCALE0),
            nn.Tanh(),
            MNCLinear(32, 10),
            ScaleDistances(SHIFT1, SCALE1)
        )
        
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=alpha)
        loss_fn = nn.CrossEntropyLoss()
        
        global_step = 0
        
        # Phase 1: Sequential Fact Training (5 facts, 15 steps each)
        for i, item in enumerate(FACTS):
            text = item["statement"]
            label = item["label"]
            vec = pipeline.embed_sentence(text)
            target = torch.tensor([label])
            
            for step_idx in range(15):
                global_step += 1
                noise = torch.randn_like(vec) * 0.05
                noisy_vec = vec + noise
                logits = model(noisy_vec)
                loss = loss_fn(logits, target)
                loss.backward()
                engine.step(loss.item())
                engine.zero_grad()
                
                # Gather variance stats
                var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
                telemetry_rows.append({
                    "alpha_decay": alpha,
                    "step": global_step,
                    "phase": f"training_fact_{i+1}_step_{step_idx+1}",
                    "var_mean": var_all.mean().item(),
                    "var_median": var_all.median().item(),
                    "var_min": var_all.min().item(),
                    "var_max": var_all.max().item()
                })
        
        # Phase 2: Rest Phase (30 steps of zero-gradient relaxation)
        for step_idx in range(30):
            global_step += 1
            # Ensure grads are zero tensors instead of None to allow relaxation step to run
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
                else:
                    p.grad = torch.zeros_like(p.data)
            engine.step(current_loss=0.0) # passes 0.0 to check decay recovery
            
            var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
            telemetry_rows.append({
                "alpha_decay": alpha,
                "step": global_step,
                "phase": f"rest_step_{step_idx+1}",
                "var_mean": var_all.mean().item(),
                "var_median": var_all.median().item(),
                "var_min": var_all.min().item(),
                "var_max": var_all.max().item()
            })
            
        # Phase 3: Interference Phase (4 interference sentences, 15 steps each)
        for i, text in enumerate(INTERFERENCE):
            vec = pipeline.embed_sentence(text)
            target = torch.tensor([5]) # arbitrary interference class label
            
            for step_idx in range(15):
                global_step += 1
                noise = torch.randn_like(vec) * 0.05
                noisy_vec = vec + noise
                logits = model(noisy_vec)
                loss = loss_fn(logits, target)
                loss.backward()
                engine.step(loss.item())
                engine.zero_grad()
                
                var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
                telemetry_rows.append({
                    "alpha_decay": alpha,
                    "step": global_step,
                    "phase": f"interference_{i+1}_step_{step_idx+1}",
                    "var_mean": var_all.mean().item(),
                    "var_median": var_all.median().item(),
                    "var_min": var_all.min().item(),
                    "var_max": var_all.max().item()
                })
        
        # Print summary for this alpha value
        final_row = telemetry_rows[-1]
        print(f"Finished alpha_decay = {alpha}:")
        print(f"  Final Mean Variance:   {final_row['var_mean']:.6f}")
        print(f"  Final Median Variance: {final_row['var_median']:.6f}")
        print(f"  Final Min Variance:    {final_row['var_min']:.6f}")
        print(f"  Final Max Variance:    {final_row['var_max']:.6f}")
        
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(telemetry_rows)
        
    print(f"\n[SUCCESS] Variance telemetry CSV saved to: {csv_path}")

if __name__ == "__main__":
    run_variance_telemetry_sweep()
