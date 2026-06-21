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
    {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "label": 0},
    {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "label": 4}
]

INTERFERENCE = [
    "The red folder is resting on the top desk.",
    "Someone left a coffee mug in the breakroom.",
    "The access code for the guest wifi is 9912.",
    "It is raining heavily outside today."
]

def evaluate(model, pipeline):
    model.eval()
    correct = 0
    with torch.no_grad():
        for f in FACTS:
            qvec = pipeline.embed_sentence(f["query"])
            logits = model(qvec)
            pred = logits.argmax(dim=-1).item()
            if pred == f["label"]:
                correct += 1
    return correct / len(FACTS)

def run_drift_analysis(seeds, csv_path):
    pipeline = JournalPipeline()
    results_dir = os.path.dirname(csv_path)
    os.makedirs(results_dir, exist_ok=True)
    
    headers = [
        "seed", "experiment", "config", "alpha_decay", "u2_enabled", "replay_buffer_size", "num_facts",
        "recall", "forgetting", "drift_active_from_w0", "drift_active_from_w_post_train",
        "drift_dormant_from_w0", "drift_layer0", "variance_mean", "variance_min", "variance_max", "runtime_sec"
    ]
    
    data_points = []
    
    import time
    
    for u2_enabled in [True, False]:
        print(f"\n[*] Evaluating drift with u2_enabled = {u2_enabled}")
        
        for seed in seeds:
            t_start = time.time()
            set_seed(seed)
            
            model = nn.Sequential(
                MNCLinear(384, 32),
                ScaleDistances(SHIFT0, SCALE0),
                nn.Tanh(),
                MNCLinear(32, 10),
                ScaleDistances(SHIFT1, SCALE1)
            )
            
            W0_init = torch.clone(model[3].W.data) # Layer 3 initial templates
            W0_layer0 = torch.clone(model[0].W.data) # Layer 0 initial templates
            
            engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, u2_enabled=u2_enabled)
            loss_fn = nn.CrossEntropyLoss()
            
            # Phase 1: Facts Training
            for item in FACTS:
                vec = pipeline.embed_sentence(item["statement"])
                target = torch.tensor([item["label"]])
                for _ in range(15):
                    noise = torch.randn_like(vec) * 0.05
                    noisy_vec = vec + noise
                    logits = model(noisy_vec)
                    loss = loss_fn(logits, target)
                    loss.backward()
                    engine.step(loss.item())
                    engine.zero_grad()
                    
            initial_acc = evaluate(model, pipeline)
            W_post_train = torch.clone(model[3].W.data)
            
            # Phase 2: Rest Phase
            for _ in range(30):
                model.zero_grad()
                engine.step(current_loss=0.0)
                
            # Phase 3: Interference Phase
            for text in INTERFERENCE:
                vec = pipeline.embed_sentence(text)
                target = torch.tensor([5]) # interference class
                for _ in range(15):
                    noise = torch.randn_like(vec) * 0.05
                    noisy_vec = vec + noise
                    logits = model(noisy_vec)
                    loss = loss_fn(logits, target)
                    loss.backward()
                    engine.step(loss.item())
                    engine.zero_grad()
                    
            final_acc = evaluate(model, pipeline)
            forgetting = initial_acc - final_acc
            
            # Drift metrics
            W_final = model[3].W.data
            
            # Active templates (classes 0-4)
            drift_active_w0 = torch.norm(W_final[0:5] - W0_init[0:5], p=2, dim=1).mean().item()
            drift_active_post = torch.norm(W_final[0:5] - W_post_train[0:5], p=2, dim=1).mean().item()
            
            # Dormant templates (classes 6-9 - completely unused)
            drift_dormant_w0 = torch.norm(W_final[6:10] - W0_init[6:10], p=2, dim=1).mean().item()
            
            # Layer 0 templates drift
            drift_layer0 = torch.norm(model[0].W.data - W0_layer0, p=2).item()
            
            # Variance statistics
            var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
            
            runtime = time.time() - t_start
            
            row = {
                "seed": seed,
                "experiment": "drift_analysis",
                "config": f"u2_enabled={u2_enabled}",
                "alpha_decay": 0.02,
                "u2_enabled": u2_enabled,
                "replay_buffer_size": 0,
                "num_facts": len(FACTS),
                "recall": final_acc,
                "forgetting": forgetting,
                "drift_active_from_w0": drift_active_w0,
                "drift_active_from_w_post_train": drift_active_post,
                "drift_dormant_from_w0": drift_dormant_w0,
                "drift_layer0": drift_layer0,
                "variance_mean": var_all.mean().item(),
                "variance_min": var_all.min().item(),
                "variance_max": var_all.max().item(),
                "runtime_sec": runtime
            }
            data_points.append(row)
            print(f"  Seed {seed:3} | Recall: {final_acc:.2f} | Forgetting: {forgetting:.2f} | Active Drift (post-train): {drift_active_post:.4f} | Dormant Drift: {drift_dormant_w0:.4f}")
            
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_points)
        
    print(f"\n[SUCCESS] Drift analysis CSV saved to: {csv_path}")
    
    # Calculate statistics & Pearson correlation
    for u2_enabled in [True, False]:
        subset = [d for d in data_points if d["u2_enabled"] == u2_enabled]
        forg_vals = [d["forgetting"] for d in subset]
        drift_w0_vals = [d["drift_active_from_w0"] for d in subset]
        drift_post_vals = [d["drift_active_from_w_post_train"] for d in subset]
        
        mean_forg = np.mean(forg_vals)
        std_forg = np.std(forg_vals)
        mean_drift_w0 = np.mean(drift_w0_vals)
        std_drift_w0 = np.std(drift_w0_vals)
        mean_drift_post = np.mean(drift_post_vals)
        std_drift_post = np.std(drift_post_vals)
        
        print(f"\nSummary for u2_enabled = {u2_enabled}:")
        print(f"  Forgetting:                     {mean_forg:.4f} +/- {std_forg:.4f}")
        print(f"  Active Drift (from W0):         {mean_drift_w0:.4f} +/- {std_drift_w0:.4f}")
        print(f"  Active Drift (from post-train): {mean_drift_post:.4f} +/- {std_drift_post:.4f}")
        
        # Pearson correlation
        if std_forg > 1e-6 and std_drift_post > 1e-6:
            r_post = np.corrcoef(drift_post_vals, forg_vals)[0, 1]
            print(f"  Correlation rho(drift_active_post, forgetting): {r_post:.4f}")
        else:
            print(f"  Correlation rho(drift_active_post, forgetting): N/A (zero variance)")
            
        if std_forg > 1e-6 and std_drift_w0 > 1e-6:
            r_w0 = np.corrcoef(drift_w0_vals, forg_vals)[0, 1]
            print(f"  Correlation rho(drift_active_w0, forgetting): {r_w0:.4f}")
        else:
            print(f"  Correlation rho(drift_active_w0, forgetting): N/A (zero variance)")

if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    run_drift_analysis(seeds, "experiments/results/drift_analysis.csv")
