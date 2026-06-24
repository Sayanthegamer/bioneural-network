import sys
import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

# Standard setup
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

JOURNAL_LINES = []
for f in FACTS:
    JOURNAL_LINES.append({"text": f["statement"], "label": f["label"]})
for idx, text in enumerate(INTERFERENCE):
    JOURNAL_LINES.append({"text": text, "label": 5 + idx})

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

def run_evaluation(seed, u2_enabled):
    t_start = time.time()
    set_seed(seed)
    
    pipeline = JournalPipeline()
    
    # 32 hidden nodes = Shared Bottleneck
    model = nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(SHIFT0, SCALE0),
        nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(SHIFT1, SCALE1)
    )
    
    # Capture initial weights to measure drift
    W0_init = torch.clone(model[0].W.data)
    W3_init = torch.clone(model[3].W.data)
    
    engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, u2_enabled=u2_enabled)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train sequentially over 9 lines
    initial_acc = 0.0
    for i, item in enumerate(JOURNAL_LINES):
        text = item["text"]
        label = item["label"]
        vec = pipeline.embed_sentence(text)
        target = torch.tensor([label])
        
        # Symmetric training protocol (15 steps for both facts and interference)
        num_steps = 15
        for _ in range(num_steps):
            noise = torch.randn_like(vec) * 0.05
            noisy_vec = vec + noise
            logits = model(noisy_vec)
            loss = loss_fn(logits, target)
            loss.backward()
            engine.step(loss.item())
            engine.zero_grad()
            
        # Immediately after Day 5 (ingesting all 5 facts), measure the baseline accuracy
        if i == 4:
            initial_acc = evaluate(model, pipeline)
            
    # Final Day 10 evaluation
    final_acc = evaluate(model, pipeline)
    forgetting = initial_acc - final_acc
    
    # Calculate template drift
    drift_0 = torch.norm(model[0].W.data - W0_init, p=2).item()
    drift_3 = torch.norm(model[3].W.data - W3_init, p=2).item()
    mean_drift = (drift_0 + drift_3) / 2.0
    
    # Variance statistics
    var_all = torch.cat([engine.variances['0.W'].view(-1), engine.variances['3.W'].view(-1)])
    var_mean = var_all.mean().item()
    var_min = var_all.min().item()
    var_max = var_all.max().item()
    
    runtime = time.time() - t_start
    
    return {
        "seed": seed,
        "recall": final_acc,
        "forgetting": forgetting,
        "drift": mean_drift,
        "variance_mean": var_mean,
        "variance_min": var_min,
        "variance_max": var_max,
        "runtime_sec": runtime
    }

if __name__ == "__main__":
    print("====================================================")
    print("  MESU U2 Cascade Ablation Study (Symmetric Steps)")
    print("====================================================")
    
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    results_dir = os.path.join(os.path.dirname(__file__), 'results/diagnostic_telemetry')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'u2_ablation.csv')
    
    headers = [
        "seed", "experiment", "config", "recall", "forgetting", 
        "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for u2 in [True, False]:
            config_str = f"u2_enabled={u2}"
            print(f"\n[*] Evaluating configuration: {config_str}")
            
            recalls = []
            forgettings = []
            drifts = []
            
            for seed in seeds:
                res = run_evaluation(seed, u2_enabled=u2)
                row = {
                    "seed": seed,
                    "experiment": "u2_ablation",
                    "config": config_str,
                    "recall": res["recall"],
                    "forgetting": res["forgetting"],
                    "drift": res["drift"],
                    "variance_mean": res["variance_mean"],
                    "variance_min": res["variance_min"],
                    "variance_max": res["variance_max"],
                    "runtime_sec": res["runtime_sec"]
                }
                writer.writerow(row)
                recalls.append(res["recall"])
                forgettings.append(res["forgetting"])
                drifts.append(res["drift"])
                print(f"  Seed {seed:3} -> Recall: {res['recall']*100:.0f}%, Forgetting: {res['forgetting']*100:.0f}%, Drift: {res['drift']:.4f}")
                
            mean_rec = np.mean(recalls) * 100
            mean_forg = np.mean(forgettings) * 100
            mean_drift = np.mean(drifts)
            
            print(f"--> [MEAN] Recall: {mean_rec:.1f}%, Forgetting: {mean_forg:.1f}%, Drift: {mean_drift:.4f}")
            
    print(f"\n[SUCCESS] Saved results to: {csv_path}")
