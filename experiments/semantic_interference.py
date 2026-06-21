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

FACT = {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "label": 1}

INTERFERENCE_CASES = {
    "related": "Server room is on floor 2.",
    "neutral": "The cafeteria opens at 8.",
    "conflicting": "The access code for the main server is 9912."
}

def run_evaluation(seed, interference_type):
    set_seed(seed)
    pipeline = JournalPipeline()
    
    model = nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(SHIFT0, SCALE0),
        nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(SHIFT1, SCALE1)
    )
    
    engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()
    
    # Capture initial weights to measure drift
    W0_init = torch.clone(model[0].W.data)
    W3_init = torch.clone(model[3].W.data)
    
    # 1. Ingest baseline Fact 1 (15 steps)
    vec_fact = pipeline.embed_sentence(FACT["statement"])
    target_fact = torch.tensor([FACT["label"]])
    
    for _ in range(15):
        noise = torch.randn_like(vec_fact) * 0.05
        noisy_vec = vec_fact + noise
        logits = model(noisy_vec)
        loss = loss_fn(logits, target_fact)
        loss.backward()
        engine.step(loss.item())
        engine.zero_grad()
        
    # Check baseline accuracy (should be 100% on the fact)
    model.eval()
    qvec = pipeline.embed_sentence(FACT["query"])
    with torch.no_grad():
        pred_init = model(qvec).argmax(dim=-1).item()
    initial_acc = 1.0 if pred_init == FACT["label"] else 0.0
    
    # 2. Rest phase: 30 steps of zero-gradient relaxation (allows variance to recover)
    model.train()
    for _ in range(30):
        # Pass dummy update step to trigger alpha_decay prior relaxation
        model.zero_grad()
        engine.step(current_loss=0.0)
        
    # 3. Ingest Interference (15 steps)
    interference_text = INTERFERENCE_CASES[interference_type]
    vec_inter = pipeline.embed_sentence(interference_text)
    target_inter = torch.tensor([5]) # Arbitrary class label 5
    
    for _ in range(15):
        noise = torch.randn_like(vec_inter) * 0.05
        noisy_vec = vec_inter + noise
        logits = model(noisy_vec)
        loss = loss_fn(logits, target_inter)
        loss.backward()
        engine.step(loss.item())
        engine.zero_grad()
        
    # 4. Final recall evaluation
    model.eval()
    with torch.no_grad():
        pred_final = model(qvec).argmax(dim=-1).item()
    final_acc = 1.0 if pred_final == FACT["label"] else 0.0
    
    forgetting = initial_acc - final_acc
    
    # Drift
    drift_0 = torch.norm(model[0].W.data - W0_init, p=2).item()
    drift_3 = torch.norm(model[3].W.data - W3_init, p=2).item()
    mean_drift = (drift_0 + drift_3) / 2.0
    
    # Variance
    var_all = torch.cat([engine.variances['0.W'].view(-1), engine.variances['3.W'].view(-1)])
    
    return {
        "recall": final_acc,
        "forgetting": forgetting,
        "drift": mean_drift,
        "variance_mean": var_all.mean().item(),
        "variance_min": var_all.min().item(),
        "variance_max": var_all.max().item()
    }

if __name__ == "__main__":
    print("====================================================")
    print("  MESU Semantic Interference & Recovery Study")
    print("====================================================")
    
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'semantic_interference.csv')
    
    headers = [
        "seed", "experiment", "config", "recall", "forgetting", 
        "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for case in ["related", "neutral", "conflicting"]:
            print(f"\n[*] Running case: {case}")
            
            recalls = []
            forgettings = []
            
            for seed in seeds:
                res = run_evaluation(seed, case)
                row = {
                    "seed": seed,
                    "experiment": "semantic_interference",
                    "config": f"interference={case}",
                    "recall": res["recall"],
                    "forgetting": res["forgetting"],
                    "drift": res["drift"],
                    "variance_mean": res["variance_mean"],
                    "variance_min": res["variance_min"],
                    "variance_max": res["variance_max"],
                    "runtime_sec": 0.0
                }
                writer.writerow(row)
                recalls.append(res["recall"])
                forgettings.append(res["forgetting"])
                print(f"  Seed {seed:3} -> Recall: {res['recall']*100:.0f}%, Forgetting: {res['forgetting']*100:.0f}%, Drift: {res['drift']:.4f}")
                
            print(f"--> [MEAN] Recall: {np.mean(recalls)*100:.1f}%, Forgetting: {np.mean(forgettings)*100:.1f}%")
            
    print(f"\n[SUCCESS] Saved results to: {csv_path}")
